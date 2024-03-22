import os
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from .crack_dataset import CrackDataset

class CrackDataModule(LightningDataModule):
    def __init__(
    self,
    data_dir,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    validation_split: float = 0.2,
    shuffle_dataset: bool = True,
    random_seed = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.validation_split = validation_split
        self.shuffle_dataset = shuffle_dataset
        self.random_seed = random_seed
        
        self.transforms = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ]
        )
        
        self.batch_size = batch_size
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
#         self.data_dir = data_dir
        all_files = []
        labels = []
        
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            for file in os.listdir(class_dir):
                all_files.append(os.path.join(class_dir, file))
                labels.append(class_name)
                
        train_files, val_files, train_labels, val_labels = train_test_split(
            all_files, labels, test_size=self.validation_split, random_state=self.random_seed
        )
        
        train_df = pd.DataFrame({'file_path': train_files, 'label': train_labels})
        val_df = pd.DataFrame({'file_path': val_files, 'label': val_labels})
        
        self.train_dataset = CrackDataset(dataframe=train_df, transform=self.transforms)
        self.val_dataset = CrackDataset(dataframe=val_df, transform=self.transforms)     
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_dataset
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )
    
    def test_dataloader(self):
        pass