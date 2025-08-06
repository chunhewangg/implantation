import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import imageio
import torchvision.transforms.functional as F

class imagedatasets(Dataset):
    def __init__(self,root,type = 'train',resize= (512,512),color = 'RGB'):
        self.root = root
        self.type = type
        self.resize = resize
        self.color = color
        
        # Load file paths during initialization
        self.data_paths = self._load_paths()
    
    def resize_with_padding(self, img_tensor, target_size):
        """Resize image while maintaining aspect ratio using padding"""
        _, h, w = img_tensor.shape
        target_h, target_w = target_size
        
        # Calculate scaling factor to fit within target size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize the image
        img_resized = F.resize(img_tensor, (new_h, new_w))
        
        # Calculate padding
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Add padding
        img_padded = F.pad(img_resized, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        return img_padded
    
    def _load_paths(self):
        """Load file paths based on dataset type"""
        if self.type == 'train':
            image_path = os.path.join(self.root, 'stage1_train')
            dir_names = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]
            length = len(dir_names)
            indices = list(range(length))
            split = int(np.floor(0.95 * length))
            np.random.shuffle(indices)
            train_indices = indices[:split]
            return [os.path.join(image_path, dir_names[idx]) for idx in train_indices]
        elif self.type == 'val':
            image_path = os.path.join(self.root, 'stage1_train')
            dir_names = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]
            length = len(dir_names)
            indices = list(range(length))
            split = int(np.floor(0.95 * length))
            np.random.shuffle(indices)
            val_indices = indices[split:]
            return [os.path.join(image_path, dir_names[idx]) for idx in val_indices]
        elif self.type == 'test':
            image_path = os.path.join(self.root, 'stage1_test')
            dir_names = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]
            return [os.path.join(image_path, dir_name) for dir_name in dir_names]
    
    def __len__(self):
        return len(self.data_paths)
        
        
    
    def __getitem__(self, idx):
        """Get a single item by index"""
        filename = self.data_paths[idx]
        
        if self.type in ['train', 'val']:
            im_path = os.path.join(filename, 'images')
            mask_path = os.path.join(filename, 'masks')
            
            if self.color == "RGB":
                img_array = imageio.imread(os.path.join(im_path, os.listdir(im_path)[0]))
                img_tensor = torch.from_numpy(img_array).permute(2,0,1)[:3].float()
                
                # Create binary mask - combine all nucleus regions
                mask_files = os.listdir(mask_path)
                if len(mask_files) > 0:
                    first_mask = imageio.imread(os.path.join(mask_path, mask_files[0]))
                    combined_mask = np.zeros(first_mask.shape[:2], dtype=np.uint8)
                    
                    for mask_file in mask_files:
                        mask = imageio.imread(os.path.join(mask_path, mask_file))
                        if len(mask.shape) == 3:
                            mask = mask[:,:,0]  # Take first channel
                        mask_binary = (mask > 0).astype(np.uint8)
                        combined_mask = np.maximum(combined_mask, mask_binary)  # Union of all masks
                    
                    mask_tensor = torch.from_numpy(combined_mask).unsqueeze(0).long()
                else:
                    mask_tensor = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2])).long()
                
            elif self.color == "gray":
                img_array = Image.open(os.path.join(im_path, os.listdir(im_path)[0])).convert('L')
                img_array = np.array(img_array)
                
                # Create binary mask - combine all nucleus regions
                mask_files = os.listdir(mask_path)
                combined_mask = np.zeros(img_array.shape, dtype=np.uint8)
                
                for mask_file in mask_files:
                    mask = np.array(Image.open(os.path.join(mask_path, mask_file)).convert('L'))
                    mask_binary = (mask > 0).astype(np.uint8)
                    combined_mask = np.maximum(combined_mask, mask_binary)  # Union of all masks
                
                # Normalize image to [0, 1]
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
                mask_tensor = torch.from_numpy(combined_mask).unsqueeze(0).long()
            
            # Resize with padding
            img_resized = self.resize_with_padding(img_tensor, self.resize)
            mask_resized = self.resize_with_padding(mask_tensor.float(), self.resize).long()
            
            return img_resized, mask_resized
            
        elif self.type == 'test':
            im_path = os.path.join(filename, 'images')
            
            if self.color == "RGB":
                img_array = imageio.imread(os.path.join(im_path, os.listdir(im_path)[0]))
                img_tensor = torch.from_numpy(img_array).permute(2,0,1)[:3].float() / 255.0
            elif self.color == "gray":
                img_array = Image.open(os.path.join(im_path, os.listdir(im_path)[0])).convert('L')
                img_array = np.array(img_array)
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
            
            img_resized = self.resize_with_padding(img_tensor, self.resize)
            return img_resized        






if __name__ == "__main__":
    root = '/Users/chunhewang/Desktop/Jesus_implemention/U-net/data/data-science-bowl-2018/'
    
    # Create train and validation datasets
    train_dataset = imagedatasets(root, type='train', resize=(512, 512), color='gray')
    val_dataset = imagedatasets(root, type='val', resize=(512, 512), color='gray')
    test_dataset = imagedatasets(root, type='test', resize=(512, 512), color='gray')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test single sample access
    img, mask = train_dataset[0]
    print(f"\nSample 0: image shape {img.shape}, mask shape {mask.shape}")
    print(f"Image dtype: {img.dtype}, Mask dtype: {mask.dtype}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"Unique mask values: {torch.unique(mask)}")
    
    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Example with batches
    print("\nWith DataLoader batching:")
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Train batch {batch_idx}: images shape {images.shape}, masks shape {masks.shape}, Unique mask values: {torch.unique(masks)}")
        print(f"Batch mask dtype: {masks.dtype}")
        if batch_idx >= 1:
            break
    
    for batch_idx, images in enumerate(test_loader):
        print(f"Test batch {batch_idx}: images shape {images.shape}")
        if batch_idx >= 1:
            break
        

        