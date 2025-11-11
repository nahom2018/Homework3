import csv
from pathlib import Path
from torchvision import transforms
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


class SuperTuxDataset(Dataset):
    """
    SuperTux dataset for classification
    """

    def __init__(
        self,
        dataset_path: str,
        transform_pipeline: str = "default",
    ):
        self.transform = self.get_transform(transform_pipeline)
        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    img_path = Path(dataset_path, fname)
                    label_id = LABEL_NAMES.index(label)

                    self.data.append((img_path, label_id))

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def get_transform(pipeline: str):
        pipeline = (pipeline or "basic").lower()
        if pipeline in ("train", "aug", "strong"):
            return T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(64, padding=4),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
                T.RandomErasing(p=0.25, scale=(0.02, 0.08), value="random"),
            ])
        # val / test
        return T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Pairs of images and labels (int) for classification
        """
        img_path, label_id = self.data[idx]
        img = Image.open(img_path)
        data = (self.transform(img), label_id)

        return data


def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 2,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader | Dataset:
    """
    Constructs the dataset/dataloader.
    The specified transform_pipeline must be implemented in the SuperTuxDataset class.

    Args:
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        return_dataloader (bool): returns either DataLoader or Dataset
        num_workers (int): data workers, set to 0 for VSCode debugging
        batch_size (int): batch size
        shuffle (bool): should be true for train and false for val

    Returns:
        DataLoader or Dataset
    """
    dataset = SuperTuxDataset(dataset_path, transform_pipeline=transform_pipeline)

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
