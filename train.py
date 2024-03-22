from src.crack_dataset import CrackDataset
from scripts.inception import InceptionCustom
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from src.crack_datamodule import CrackDataModule
from scripts.crack_module import Classifier

# Inisialisasi CrackDataModule
data_module = CrackDataModule(data_dir="/kaggle/input/surface-crack-detection", batch_size=64)

# Inisialisasi model InceptionV3
num_classes = 2
inception_model = InceptionCustom(num_classes=num_classes)

# Inisialisasi Classifier
classifier = Classifier(net=inception_model, lr=0.001)

# Inisialisasi PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=10)  

# Melatih model
trainer.fit(classifier, data_module)

