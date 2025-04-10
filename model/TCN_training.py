import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Import your custom modules
from TCN_model import TCN
from TCN_embedding import VideoFeatureExtractor
from Data_utls.Datasets import VideoDataset


class CombinedModel(nn.Module):
    """Combined model with feature extraction and TCN"""

    def __init__(self, feature_extractor, tcn_model):
        super(CombinedModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.tcn_model = tcn_model

    def forward(self, x):
        # Extract features
        features = self.feature_extractor.extract_features_for_tcn(x)

        # Apply TCN model
        outputs, encoder_features, decoder_features = self.tcn_model(features)

        return outputs, encoder_features, decoder_features


def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        num_epochs=25,
        device='cuda',
        save_dir='checkpoints'
):
    """Train the combined model with duplicated per-frame labels"""
    os.makedirs(save_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for frames, labels in tqdm(dataloader):
                frames = frames.to(device)
                labels = labels.to(device)  # labels shape: [batch] (video-level)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Get per-frame outputs: [batch, time, num_classes]
                    outputs, _, _ = model(frames)

                    # Duplicate the video-level label to get per-frame labels.
                    # Let T be the number of time steps.
                    T = outputs.size(1)
                    expanded_labels = labels.unsqueeze(1).expand(-1, T)  # Shape: [batch, T]

                    # Flatten outputs and labels to compute loss frame by frame
                    loss = criterion(outputs.view(-1, outputs.size(-1)),
                                     expanded_labels.reshape(-1))

                    # For predictions we aggregate the per-frame predictions via majority vote.
                    _, frame_preds = torch.max(outputs, dim=2)  # [batch, T]
                    preds, _ = torch.mode(frame_preds, dim=1)   # [batch]

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * frames.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'acc': epoch_acc,
                    }, os.path.join(save_dir, 'best_model.pth'))
                    print(f"Saved new best model with validation accuracy: {epoch_acc:.4f}")

        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(save_dir, f'epoch_{epoch + 1}.pth'))
        print()

    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(save_dir, 'final_model.pth'))
    print(f'Best val Acc: {best_val_acc:4f}')
    return model, history



def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test data using majority vote aggregation"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in tqdm(test_loader):
            frames = frames.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(frames)
            T = outputs.size(1)
            expanded_labels = labels.unsqueeze(1).expand(-1, T)
            _, frame_preds = torch.max(outputs, dim=2)
            preds, _ = torch.mode(frame_preds, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    return metrics


def plot_training_history(history, save_path='training_history.png'):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train combined feature extraction + TCN model')
    parser.add_argument('--data_dir', type=str, default='E:/TCN/DATA', help='Data directory')
    parser.add_argument('--train_csv', type=str, default='videos_train_shuffled.csv', help='Training CSV file')
    parser.add_argument('--val_csv', type=str, default='videos_val_shuffled.csv', help='Validation CSV file')
    parser.add_argument('--test_csv', type=str, default='videos_test_shuffled.csv', help='Test CSV file')
    parser.add_argument('--feature_type', type=str, default='resnet', choices=['cnn', 'resnet','resnet18','resnet34','resnet50','resnet101'],
                        help='Feature extractor type')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--kernel_size', type=int, default=3, help='TCN kernel size')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_csv_path = os.path.join(args.data_dir, args.train_csv)
    val_csv_path = os.path.join(args.data_dir, args.val_csv)
    test_csv_path = os.path.join(args.data_dir, args.test_csv)
    train_dataset = VideoDataset(train_csv_path, num_frames=args.num_frames, base_path=args.data_dir)
    val_dataset = VideoDataset(val_csv_path, num_frames=args.num_frames, base_path=args.data_dir)
    test_dataset = VideoDataset(test_csv_path, num_frames=args.num_frames, base_path=args.data_dir)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Number of classes
    num_classes = len(np.unique(train_dataset.encoded_labels))
    print(f"Number of classes: {num_classes}")

    # Initialize feature extractor
    print(f"Initializing {args.feature_type} feature extractor...")
    feature_extractor = VideoFeatureExtractor(
        extractor_type=args.feature_type,
        output_dim=args.feature_dim,
        device=device
    )

    # Initialize TCN model
    print("Initializing TCN model...")
    kernel_size_int = int(args.kernel_size)
    tcn_model = TCN(
        input_channels=args.feature_dim,
        output_channels=args.feature_dim,
        num_classes=num_classes,
        num_channels=[32, 64, 96],
        kernel_size=kernel_size_int,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Create combined model
    combined_model = CombinedModel(feature_extractor, tcn_model).to(device)

    # Initialize loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()

    if args.feature_type == 'cnn':
        optimizer = optim.AdamW(
            combined_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.AdamW(
        tcn_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'min',
        patience=5,
        factor=0.5
    )

    # Train model
    print("Starting training...")
    start_time = time.time()
    trained_model, history = train_model(
        model=combined_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time / 60:.2f} minutes")

    # Plot training history
    plot_training_history(history, os.path.join(args.save_dir, 'training_history.png'))

    # Evaluate on test set
    print("Evaluating on test set...")
    metrics = evaluate_model(trained_model, test_loader, device)

    # Save evaluation results
    with open(os.path.join(args.save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Test Precision: {metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {metrics['recall']:.4f}\n")
        f.write(f"Test F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Training Time: {training_time / 60:.2f} minutes\n")


if __name__ == "__main__":
    main()