import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class ProductDataset(Dataset):
    
    def __init__(self,num_samples=100):
        self.num_samples=num_samples
        self.vocab={"<PAD>":0,"great":1,"good":2,"bad":3,"terrible":4}
        self.labels=np.random.randint(0,2,num_samples)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,idx):
        label=self.labels[idx]
        mock_image_np = np.random.randn(3, 224, 224).astype(np.float32)
        image_tensor=torch.tensor(mock_image_np)
        mock_text_list = [1, 2, 0, 0]
        text_tensor=torch.tensor(mock_text_list,dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, text_tensor, label_tensor
    
if __name__ == "__main__":
    dataset = ProductDataset(num_samples=10)
    print(f"Dataset initialization successful! Length: {len(dataset)}")
    
    img, txt, lbl = dataset[0]
    print("✓ Image Tensor Shape:", img.shape)  
    print("✓ Text Tensor:", txt)
    print("✓ Label Tensor:", lbl)
        