from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
import torchmetrics
import torchmetrics
from torchmetrics import Accuracy


class Classifier(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001
    ) -> None:
        super().__init__()
        self.model = net
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
        self.lr = lr
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        # No need to apply softmax here, as it's likely already applied in the model
        loss = F.cross_entropy(outputs.logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        # No need to apply softmax here, as it's likely already applied in the model
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)


        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer