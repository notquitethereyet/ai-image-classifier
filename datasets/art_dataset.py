import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple


class ArtDataset(Dataset):
    """Dataset for AI vs Human Art Classification"""
    
    def __init__(self, 
                 csv_file: str, 
                 root_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 is_test: bool = False):
        """
        Args:
            csv_file: Path to the csv file with annotations
            root_dir: Directory with all the images
            transform: Optional transform to be applied on a sample
            is_test: If True, dataset is used for testing (no labels)
        """
        # Read CSV file with proper types
        self.data_frame = pd.read_csv(csv_file)
        
        # Check CSV structure
        print(f"CSV columns: {self.data_frame.columns.tolist()}")
        
        # Handle test CSV (assumes single column with image paths)
        if is_test:
            if len(self.data_frame.columns) == 1:
                # If the column name isn't 'id', rename it
                if self.data_frame.columns[0] != 'id':
                    self.data_frame.columns = ['id']
            else:
                print(f"Warning: Test CSV has {len(self.data_frame.columns)} columns, expected 1")
        
        # Handle training CSV
        else:
            # Make sure file_name and label columns exist
            if 'file_name' not in self.data_frame.columns or 'label' not in self.data_frame.columns:
                # Check if using default pandas index column naming
                if self.data_frame.columns[0] == 'Unnamed: 0':
                    # Rename columns to expected format
                    self.data_frame.columns = ['index', 'file_name', 'label']
                    print("Renamed columns to ['index', 'file_name', 'label']")
                else:
                    print(f"Warning: Training CSV missing expected columns. Available: {self.data_frame.columns.tolist()}")
        
        # Convert file_name column to string if it exists
        if 'file_name' in self.data_frame.columns:
            self.data_frame['file_name'] = self.data_frame['file_name'].astype(str)
        
        if is_test and 'id' in self.data_frame.columns:
            self.data_frame['id'] = self.data_frame['id'].astype(str)
        
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self) -> int:
        return len(self.data_frame)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor] or torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name using column names if available, or positions if not
        if self.is_test:
            if 'id' in self.data_frame.columns:
                img_name = self.data_frame.loc[idx, 'id']
            else:
                img_name = self.data_frame.iloc[idx, 0]
        else:
            if 'file_name' in self.data_frame.columns:
                img_name = self.data_frame.loc[idx, 'file_name']
            else:
                img_name = self.data_frame.iloc[idx, 1]  # Assuming second column is file_name
        
        # Convert img_name to string if it's not already (handles pandas int64 type)
        img_name = str(img_name) if not isinstance(img_name, str) else img_name
            
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            image = Image.new('RGB', (224, 224))
            
            # Try alternative path (remove or add folder prefix)
            if 'test_data_v2' in img_path:
                alt_path = img_path.replace('test_data_v2', 'test_data')
                try:
                    print(f"Trying alternative path: {alt_path}")
                    image = Image.open(alt_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading from alternative path: {e}")
            elif 'test_data' in img_path and not 'test_data_v2' in img_path:
                alt_path = img_path.replace('test_data', 'test_data_v2')
                try:
                    print(f"Trying alternative path: {alt_path}")
                    image = Image.open(alt_path).convert('RGB')
                except Exception as e:
                    print(f"Error loading from alternative path: {e}")
                    
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image
        else:
            # Get label using column name if available, or position if not
            if 'label' in self.data_frame.columns:
                label = self.data_frame.loc[idx, 'label']
            else:
                label = self.data_frame.iloc[idx, 2]  # Assuming third column is label
                
            label = torch.tensor(float(label), dtype=torch.float32)
            return image, label


def get_data_loaders(data_dir: str, 
                     batch_size: int = 32,
                     num_workers: int = 4,
                     img_size: int = 224) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and test data loaders
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        img_size: Size to resize images to
        
    Returns:
        train_loader, test_loader: DataLoader objects
    """
    # Define transformations for training data
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transformations for test data
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test_v2.csv')
    
    train_dataset = ArtDataset(csv_file=train_csv, 
                               root_dir=data_dir, 
                               transform=train_transform,
                               is_test=False)
    
    test_dataset = ArtDataset(csv_file=test_csv,
                              root_dir=data_dir,
                              transform=test_transform,
                              is_test=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True)
    
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    
    return train_loader, test_loader