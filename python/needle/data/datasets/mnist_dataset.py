from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as f:
            f.read(16)
            file_content = f.read()
            self.images = (np.frombuffer(file_content, dtype=np.uint8).astype(np.float32)/255).reshape(-1, 784)
        with gzip.open(label_filename, 'rb') as f:
            f.read(8)
            file_content = f.read()
            self.labels = np.frombuffer(file_content, dtype=np.uint8)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = self.apply_transforms(self.images[index].reshape(28, 28, -1))
        return images.reshape(-1, 784), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION