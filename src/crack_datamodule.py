import torch
from pytorch_lightning import LighntingDataModule
from crack_dataset import CrackDataset

class CrackDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            validation_split: float = 0.2,
            shuffle_dataset: bool = True,
            random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.validation_split = validation_split
        self.shuffle_dataset = shuffle_dataset
        self.random_seed = random_seed

        self.transforms = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                
            ]
        )