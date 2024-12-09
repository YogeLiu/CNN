import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['cat', 'dog']
        self.images = []
        self.labels = []
        
        # 加载所有图片路径和标签
        for img_name in os.listdir(root_dir):
            self.images.append(os.path.join(root_dir, img_name))
            self.labels.append(0 if 'cat' in img_name else 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label 
    

if __name__ == '__main__':
    dataset = CatDogDataset(root_dir='data/train')
    print(len(dataset))
