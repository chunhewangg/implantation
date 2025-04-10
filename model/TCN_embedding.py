import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class CNNFeatureExtractor(nn.Module):
    """
    Custom CNN for feature extraction from video frames using ModuleList
    """

    def __init__(self, output_dim=256, channels=[16, 32, 64, 128, 256]):
        super(CNNFeatureExtractor, self).__init__()

        # Input channels
        in_channels = 3

        # Create CNN blocks using ModuleList
        self.conv_blocks = nn.ModuleList()
        for out_channels in channels:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        self.fc = None
        self.output_dim = output_dim

        # Transformation for input normalization
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        # Apply normalization
        x = self.transform(x)

        # Apply CNN blocks
        for block in self.conv_blocks:
            x = block(x)

        # Dynamically create FC layer if not created yet
        if self.fc is None:
            self.feature_size = x.size(1) * x.size(2) * x.size(3)
            self.fc = nn.Linear(self.feature_size, self.output_dim).to(x.device)
           # print(f"Created FC layer with input size: {self.feature_size}")
        # Flatten
        x = x.view(x.size(0), -1)

        # Apply FC layer
        x = self.fc(x)

        return x


class ResNetFeatureExtractor(nn.Module):
    """
    Pre-trained ResNet feature extractor
    """

    def __init__(self, output_dim=128, model_name='resnet50', pretrained=True):
        super(ResNetFeatureExtractor, self).__init__()

        # Load pre-trained model
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Model {model_name} not supported")

        # Remove classification layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])

        # Add projection layer if needed
        if output_dim != self.feature_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = nn.Identity()

        # Transformation for input normalization (ImageNet stats)
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        # Apply normalization
        x = self.transform(x)

        # Extract features
        x = self.features(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Apply projection if needed
        x = self.projection(x)

        return x


class VideoFeatureExtractor:
    """
    Utility class for extracting features from video frames
    """

    def __init__(self, extractor_type='resnet', output_dim=128, device=None):
        """
        Initialize video feature extractor

        Args:
            extractor_type: Type of extractor ('cnn' or 'resnet')
            output_dim: Dimension of output features
            device: Device to run the model on
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Create feature extractor
        if extractor_type.lower() == 'cnn':
            self.model = CNNFeatureExtractor(output_dim=output_dim).to(self.device)
        elif extractor_type.lower() == 'resnet':
            self.model = ResNetFeatureExtractor(
                output_dim=output_dim,
                model_name='resnet50',
                pretrained=True
            ).to(self.device)
        elif extractor_type.lower() == 'resnet18' or 'resnet34' or 'resnet50' or 'resnet101':
            self.model = ResNetFeatureExtractor(
                output_dim=output_dim,
                model_name=extractor_type.lower(),
                pretrained=True
            ).to(self.device)
        else:
            raise ValueError(f"Extractor type {extractor_type} not supported")

        # Set model to evaluation mode
        self.model.eval()

    def extract_features(self, frames):
        """
        Extract features from video frames

        Args:
            frames: Tensor of shape [num_frames, channels, height, width]
                  or [batch_size, num_frames, channels, height, width]

        Returns:
            Tensor of shape [num_frames, output_dim]
                  or [batch_size, num_frames, output_dim]
        """
        # Check if batch dimension exists
        if len(frames.shape) == 5:
            batch_size, num_frames, channels, height, width = frames.shape
            is_batch = True
        elif len(frames.shape) == 4:
            num_frames, channels, height, width = frames.shape
            batch_size = 1
            is_batch = False
            # Add batch dimension
            frames = frames.unsqueeze(0)
        else:
            raise ValueError("Input frames should be 4D or 5D tensor")

        # Process each video
        all_features = []
        with torch.no_grad():
            for b in range(batch_size):
                # Process each frame
                frame_features = []
                for f in range(num_frames):
                    # Get frame and move to device
                    frame = frames[b, f].to(self.device)

                    # Extract features (add batch dimension)
                    features = self.model(frame.unsqueeze(0))
                    frame_features.append(features.squeeze(0))

                # Stack frame features
                video_features = torch.stack(frame_features)
                all_features.append(video_features)

        # Stack features for all videos
        if is_batch:
            result = torch.stack(all_features)
        else:
            result = all_features[0]

        return result

    def extract_features_batch(self, frames, batch_size=32):
        """
        Extract features from video frames using batch processing

        Args:
            frames: Tensor of shape [num_frames, channels, height, width]
                  or [batch_size, num_frames, channels, height, width]
            batch_size: Batch size for frame processing

        Returns:
            Tensor of shape [num_frames, output_dim]
                  or [batch_size, num_frames, output_dim]
        """
        # Check if batch dimension exists
        if len(frames.shape) == 5:
            video_batch_size, num_frames, channels, height, width = frames.shape
            is_batch = True
        elif len(frames.shape) == 4:
            num_frames, channels, height, width = frames.shape
            video_batch_size = 1
            is_batch = False
            # Add batch dimension
            frames = frames.unsqueeze(0)
        else:
            raise ValueError("Input frames should be 4D or 5D tensor")

        # Reshape to process all frames at once
        frames_flat = frames.view(video_batch_size * num_frames, channels, height, width)

        # Process frames in batches
        all_features = []
        with torch.no_grad():
            for i in range(0, len(frames_flat), batch_size):
                # Get batch of frames
                batch = frames_flat[i:i + batch_size].to(self.device)

                # Extract features
                features = self.model(batch)
                all_features.append(features.cpu())

        # Concatenate all batches
        all_features = torch.cat(all_features, dim=0)

        # Reshape back to original batch structure
        if is_batch:
            all_features = all_features.view(video_batch_size, num_frames, -1)
        else:
            all_features = all_features.view(num_frames, -1)

        return all_features

    def extract_features_for_tcn(self, frames, batch_size=32):
        """
        Extract features from video frames and reshape for TCN input

        Args:
            frames: Tensor of shape [num_frames, channels, height, width]
                  or [batch_size, num_frames, channels, height, width]
            batch_size: Batch size for frame processing

        Returns:
            Tensor of shape [output_dim, num_frames]
                  or [batch_size, output_dim, num_frames]
        """
        # Extract features
        features = self.extract_features_batch(frames, batch_size)

        # Transpose for TCN input (time dimension last)
        if len(features.shape) == 3:
            # [batch, frames, features] -> [batch, features, frames]
            features = features.transpose(1, 2)
        else:
            # [frames, features] -> [features, frames]
            features = features.t()

        return features


# Example usage
if __name__ == "__main__":
    # Create dummy video data
    num_frames = 16
    batch_size = 2
    frames = torch.randn(batch_size, num_frames, 3, 224, 224)

    # Initialize feature extractors
    cnn_extractor = VideoFeatureExtractor(extractor_type='cnn', output_dim=512)
    resnet_extractor = VideoFeatureExtractor(extractor_type='resnet18', output_dim=512)

    # Extract features using CNN
    print("Extracting features using CNN...")
    cnn_features = cnn_extractor.extract_features_for_tcn(frames)
    print(f"CNN features shape: {cnn_features.shape}")  # Expected: [batch_size, 128, num_frames]

    # Extract features using ResNet
    print("Extracting features using ResNet...")
    resnet_features = resnet_extractor.extract_features_for_tcn(frames)
    print(f"ResNet features shape: {resnet_features.shape}")  # Expected: [batch_size, 128, num_frames]