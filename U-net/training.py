from loader import imagedatasets
from torch.utils.data import DataLoader
from Cont_Exp import UNet
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, f1_score
import logging
from datetime import datetime
from torch.amp import GradScaler, autocast


# Data augmentation transforms
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=10.0, p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
    A.OpticalDistortion(distort_limit=0.2, p=0.2),
])

val_transform = A.Compose([])

def Args(cuda_device=None):
    class args:
        def __init__(self):
            self.root = '/Users/chunhewang/Desktop/Jesus_implemention/U-net/data/data-science-bowl-2018'
            self.type = 'train'
            self.resize = (512, 512)
            self.color = 'gray'  # Changed to gray for single channel
            self.batch_size = 8  # Reduced for better memory usage
            self.lr = 0.001
            self.epochs = 100
            self.model_save_path = './checkpoints/'
            self.use_amp = True  # Enable mixed precision training
            
            # Configurable CUDA device selection
            if cuda_device is not None:
                if torch.cuda.is_available() and cuda_device < torch.cuda.device_count():
                    self.device = torch.device(f'cuda:{cuda_device}')
                else:
                    print(f"Warning: CUDA device {cuda_device} not available. Using CPU.")
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.save_every = 10  # Save checkpoint every N epochs
            self.log_every = 10   # Log every N batches
    return args()

# Custom dataset wrapper with augmentations
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        
        # Convert to numpy for albumentations (expects HWC format)
        image_np = image.squeeze(0).numpy()  # Remove channel dim for grayscale
        mask_np = mask.squeeze(0).numpy().astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']
        
        # Convert back to tensors
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).long()
        
        # Normalize image to [0, 1] if not already
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
            
        return image_tensor, mask_tensor

# Dice Loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, predictions, targets):
        dice_loss = self.dice_loss(predictions, targets.float())
        bce_loss = self.bce_loss(predictions, targets.float())
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

# Metrics calculation
def calculate_metrics(predictions, targets):
    predictions = torch.sigmoid(predictions)
    pred_binary = (predictions > 0.5).float()
    targets_binary = targets.float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = targets_binary.view(-1).cpu().numpy()
    
    accuracy = accuracy_score(target_flat, pred_flat)
    f1 = f1_score(target_flat, pred_flat, average='binary', zero_division=0)
    
    # IoU (Intersection over Union)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = intersection / union if union > 0 else 0
    
    return accuracy, f1, iou

# Setup logging
def setup_logging(args):
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device, logger, epoch, log_every, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_iou = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            accuracy, f1, iou = calculate_metrics(outputs, masks)
        
        total_loss += loss.item()
        total_accuracy += accuracy
        total_f1 += f1
        total_iou += iou
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.4f}',
            'IoU': f'{iou:.4f}'
        })
        
        # Log every N batches
        if (batch_idx + 1) % log_every == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx+1}/{num_batches}, '
                       f'Loss: {loss.item():.4f}, Acc: {accuracy:.4f}, '
                       f'F1: {f1:.4f}, IoU: {iou:.4f}')
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_f1 = total_f1 / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_accuracy, avg_f1, avg_iou

# Validation function
def validate_epoch(model, val_loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_f1 = 0.0
    total_iou = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images, masks = images.to(device), masks.to(device)
            
            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            accuracy, f1, iou = calculate_metrics(outputs, masks)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            total_f1 += f1
            total_iou += iou
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    avg_f1 = total_f1 / num_batches
    avg_iou = total_iou / num_batches
    
    return avg_loss, avg_accuracy, avg_f1, avg_iou

# Save checkpoint
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, args, is_best=False):
    os.makedirs(args.model_save_path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    # Regular checkpoint
    checkpoint_path = os.path.join(args.model_save_path, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Best model checkpoint
    if is_best:
        best_path = os.path.join(args.model_save_path, 'best_model.pth')
        torch.save(checkpoint, best_path)

# Main training function
def train_model(cuda_device=None):
    args = Args(cuda_device=cuda_device)
    logger = setup_logging(args)
    
    logger.info(f"Training configuration:")
    logger.info(f"Device: {args.device}")
    logger.info(f"Mixed Precision: {args.use_amp}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Image size: {args.resize}")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset_base = imagedatasets(args.root, type='train', resize=args.resize, color=args.color)
    val_dataset_base = imagedatasets(args.root, type='val', resize=args.resize, color=args.color)
    
    # Apply augmentations
    train_dataset = AugmentedDataset(train_dataset_base, transform=train_transform)
    val_dataset = AugmentedDataset(val_dataset_base, transform=val_transform)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Model, loss, optimizer
    model = UNet(input_dim=1).to(args.device)  # 1 channel for grayscale
    criterion = CombinedLoss(dice_weight=0.6, bce_weight=0.4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Mixed precision training
    scaler = GradScaler('cuda') if args.use_amp else None
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_ious = []
    val_ious = []
    
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Training
        train_loss, train_acc, train_f1, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, args.device, logger, epoch, args.log_every,
            scaler=scaler, use_amp=args.use_amp
        )
        
        # Validation
        val_loss, val_acc, val_f1, val_iou = validate_epoch(
            model, val_loader, criterion, args.device, use_amp=args.use_amp
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_ious.append(train_iou)
        val_ious.append(val_iou)
        
        # Log epoch results
        logger.info(f"Epoch {epoch}/{args.epochs}:")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, IoU: {train_iou:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, IoU: {val_iou:.4f}")
        logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoints
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"New best model! Val loss: {val_loss:.4f}")
        
        if epoch % args.save_every == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, args, is_best)
        
        # Plot training curves every 10 epochs
        if epoch % 10 == 0:
            plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, 
                                train_ious, val_ious, epoch)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/3600:.2f} hours")
    
    # Final model save
    final_path = os.path.join(args.model_save_path, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")
    
    return model, train_losses, val_losses

# Plot training curves
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, train_ious, val_ious, epoch):
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Val Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # IoU plot
    plt.subplot(1, 3, 3)
    plt.plot(train_ious, label='Train IoU', color='blue')
    plt.plot(val_ious, label='Val IoU', color='red')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/training_curves_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage:
    # For GPU 0: python training.py
    # For GPU 1: model, train_losses, val_losses = train_model(cuda_device=1)
    # For GPU 2: model, train_losses, val_losses = train_model(cuda_device=2)
    # For CPU: model, train_losses, val_losses = train_model(cuda_device=None) or set cuda_device to invalid number
    
    model, train_losses, val_losses = train_model(cuda_device=2)  # Use GPU 2