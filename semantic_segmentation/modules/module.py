import os
import pdb
from typing import Any, Dict, List, Optional, Tuple

import oyaml as yaml
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from .losses import get_div_loss_weight, js_div_loss, gjs_div_loss, kl_div_loss

# from modules.deeplab.modeling import deeplabv3plus_resnet50
# from modules.unet import UNet

class SegmentationNetwork(pl.LightningModule):

  def __init__(self, network: nn.Module, 
                     criterion: nn.Module, 
                     learning_rate: float, 
                     weight_decay: float, 
                     train_step_settings: Optional[List[str]] = None,
                     val_step_settings: Optional[List[str]] = None,
                     test_step_settings: Optional[List[str]] = None,
                     predict_step_settings: Optional[List[str]] = None,
                     ckpt_path=None):
    super().__init__()

    self.network = network
    self.criterion = criterion
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    
    self.last_logits = None
    self.validation_step_outputs: List = []

    self.save_hyperparameters("learning_rate", "weight_decay")

    # evaluation metrics for all classes
    self.metric_train_iou = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=self.network.num_classes, average=None
    )
    self.metric_val_iou = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=self.network.num_classes, average=None
    )
    self.metric_test_iou = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=self.network.num_classes, average=None
    )

    if train_step_settings is not None:
      self.train_step_settings = train_step_settings
    else:
      self.train_step_settings = []

    if val_step_settings is not None:
      self.val_step_settings = val_step_settings
    else:
      self.val_step_settings = []

    if test_step_settings is not None:
      self.test_step_settings = test_step_settings
    else:
      self.test_step_settings = []

    if predict_step_settings is not None:
      self.predict_step_settings = predict_step_settings
    else:
      self.predict_step_settings = []

    if ckpt_path is not None:
      print("Load pretrained weights of backbone.")
      ckpt_dict = torch.load(ckpt_path)
      self.load_state_dict(ckpt_dict['state_dict'], strict=False)

  def compute_xentropy_loss(self, logits: torch.Tensor, y: torch.Tensor, mode: str, mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ Compute cross entropy loss based on logits and ground-truths.

    Args:
        logits (torch.Tensor): logits [B x num_classes x H x W]
        y (torch.Tensor): ground-truth [B x H x W]
        mode (str): train, val, or test
        mask_keep (Optional[torch.Tensor], optional): 1 := consider annotation, 0 := do not consider annotation [B x H x W]. Defaults to None.

    Returns:
        torch.Tensor: loss
    """

    return self.criterion(logits, y, mode=mode, mask_keep=mask_keep)

  def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
    """ Forward pass of backbone network.

    Args:
        img_batch (torch.Tensor): input image(s) [B x C x H x W]

    Returns:
        torch.Tensor, torch.Tensor: predictions of segmentation head [B x num_classes x H x W]
    """
    output = self.network.forward(img_batch)
    
    return output

  def training_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
    if not self.train_step_settings:
      raise ValueError('You need to specify the settings for the training step.')

    processed_steps = []
    if 'regular' in self.train_step_settings:
      logits_regular = self.forward(batch['input_image'])

      mask_keep_regular = batch['anno'] != 255
      loss_regular = self.compute_xentropy_loss(logits_regular, batch['anno'], mode='train', mask_keep=mask_keep_regular)

      processed_steps.append('regular')
      # update training metrics
      pred = torch.argmax(logits_regular.detach(), dim=1)
      self.metric_train_iou(pred , batch['anno'])

    if (self.trainer.current_epoch == 0) and (batch_idx == 0):
      print("Processed steps during training: ", processed_steps)

    # accumulate all losses
    my_local_loss_values = {var_name: var_value for var_name, var_value in locals().items() if var_name.startswith('loss')}
    loss = torch.sum(torch.stack(list(my_local_loss_values.values())))
    
    self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    self.metric_train_iou.update(pred, batch['anno'])
    
    out_dict = {'loss': loss, 'logits': logits_regular, 'anno': batch['anno']}
    out_dict.update(my_local_loss_values)
    del my_local_loss_values
    
    return out_dict
   
  def on_train_epoch_end(self) -> None:
      # ---- IoU (epoch-level, Lightning-visible) ----
      mIoU = self.metric_train_iou.compute()
      self.log(
          "train_mIoU",
          mIoU.mean(),
          on_epoch=True,
          prog_bar=True,
          sync_dist=True
      )
      self.metric_train_iou.reset()

  def validation_step(self, batch: dict, batch_idx: int):
      if not self.val_step_settings:
          raise ValueError("You need to specify the settings for the validation step.")
      assert len(self.val_step_settings) == 1

      # forward
      logits = self.forward(batch["input_image"])

      # loss
      loss = self.compute_xentropy_loss(
          logits, batch["anno"], mode="val"
      )

      # Lightning-visible logging (this is what matters)
      self.log(
          "val_loss",
          loss,
          on_step=False,
          on_epoch=True,
          prog_bar=True,
          sync_dist=True
      )

      # update metric state (NO compute here)
      preds = torch.argmax(logits, dim=1)
      self.metric_val_iou.update(preds, batch["anno"])
      
      self.last_logits = logits.detach()

      return loss

  def on_validation_epoch_end(self) -> None:
      # ---- IoU aggregation ----
      iou_per_class = self.metric_val_iou.compute()
      mIoU = iou_per_class.mean()

      # Lightning-visible metric (checkpoint-safe)
      self.log(
          "val_mIoU",
          mIoU,
          on_epoch=True,
          prog_bar=True,
          sync_dist=True
      )

      # optional: per-class IoU (logger-only diagnostics)
      epoch = self.trainer.current_epoch
      for class_index, iou_class in enumerate(iou_per_class):
          self.logger.experiment.log_metrics(
              {f"val/iou_class_{class_index}": iou_class},
              epoch
          )

      self.logger.experiment.log_metrics(
          {"val/mIoU": mIoU},
          epoch
      )

      # reset metric state
      self.metric_val_iou.reset()

      # optional artifact saving
      path = os.path.join(
          self.trainer.log_dir,
          "val",
          "evaluation",
          "iou-classwise",
          f"epoch-{epoch:06d}",
      )
      save_iou_metric(iou_per_class, path)

  def test_step(self, batch: dict, batch_idx: int):
      if not self.test_step_settings:
          raise ValueError("You need to specify the settings for the test step.")
      assert len(self.test_step_settings) == 1

      # forward
      
      logits = self.forward(batch["input_image"])

      preds = torch.argmax(logits, dim=1)
      
      self.last_logits = logits.detach()

      # update metric state (ONLY update, never compute)
      self.metric_test_iou.update(preds, batch["anno"])

  def on_test_epoch_end(self) -> None:
      iou_per_class = self.metric_test_iou.compute()
      mIoU = float(iou_per_class.mean())

      # Lightning-visible logging (optional but correct)
      self.log(
          "test_mIoU",
          mIoU,
          prog_bar=True,
          sync_dist=True
      )
      for class_index, iou_class in enumerate(iou_per_class):
          self.logger.experiment.log_metrics(
              {f"test/iou_class_{class_index}": iou_class},
              self.trainer.current_epoch
          )

      print(f"Test mIoU: {mIoU:.3f}")

      self.metric_test_iou.reset()

      epoch = self.trainer.current_epoch
      path = os.path.join(
          self.trainer.log_dir,
          "evaluation",
          "iou-classwise",
          f"epoch-{epoch:06d}",
      )
      save_iou_metric(iou_per_class, path)

  def lr_scaling(self, current_epoch: int) -> float:
    warm_up_epochs = 16
    if current_epoch <= warm_up_epochs:
      lr_scale = current_epoch / warm_up_epochs
    else:
      lr_scale = pow((1 - ((current_epoch - (warm_up_epochs + 1)) / (self.trainer.max_epochs - (warm_up_epochs + 1)))), 3.0)

    return lr_scale

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.lr_scaling(epoch))

    scheduler_config = {
        "scheduler": scheduler,
        "interval": "epoch",   
        "frequency": 1,
        "name": "learning_rate",
    }

    return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
  
  def lr_scheduler_step(self, scheduler, optimizer_idx):
    scheduler.step()

def save_iou_metric(metrics: torch.Tensor, path_to_dir: str) -> None:
  if not os.path.exists(path_to_dir):
    os.makedirs(path_to_dir)

  iou_info = {}
  for cls_index, iou_metric in enumerate(metrics):
    iou_info[f'class_{cls_index}'] = round(float(iou_metric), 5)

  iou_info['mIoU'] = round(float(metrics.mean()), 5)

  fpath = os.path.join(path_to_dir, "iou.yaml")
  with open(fpath, 'w') as ostream:
    yaml.dump(iou_info, ostream)
