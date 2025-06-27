import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import os


class VideoDataset(Dataset):
    def __init__(self, csv_path, num_frames=16, resize_shape=(224, 224),
                 csv_video_path_col='clip_path', csv_label_col='label', base_path=None):
        self.num_frames = int(num_frames)
        self.resize_shape = (int(resize_shape[0]), int(resize_shape[1]))

        # Load CSV data
        if csv_path is not None:
            video_csv = pd.read_csv(csv_path)
        else:
            raise ValueError("You must provide a csv file with the video paths and labels")

        self.video_paths_adjusted = []
        self.video_paths = video_csv[csv_video_path_col].values
        for video_path in self.video_paths:
            if video_path.startswith('/'):
                video_path = video_path[1:]
            self.video_paths_adjusted.append(video_path)



        self.labels = video_csv[csv_label_col].values


        if base_path is not None:
            self.video_paths_adjusted = [os.path.join(base_path, path) for path in self.video_paths_adjusted]

        # Create label encoder once during initialization
        self.label_encoder_dict = self._create_label_encoder(self.labels)
        self.encoded_labels = np.array([self.label_encoder_dict[label] for label in self.labels])

    def __len__(self):
        return len(self.video_paths_adjusted)

    def _create_label_encoder(self, labels):
        """Create a mapping from text labels to integers"""
        unique_labels = np.unique(labels)
        return {label: i for i, label in enumerate(unique_labels)}

    def __getitem__(self, idx):
        video_path = self.video_paths_adjusted[idx]



        # Open video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            # Return a placeholder frame array instead of None
            frames_array = np.zeros((self.num_frames, 3, self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)
            return torch.from_numpy(frames_array), self.encoded_labels[idx]

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame indices to sample
        if total_frames >= self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            frame_indices = np.arange(0, self.num_frames) % total_frames

        # Extract frames
        frames = []
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, self.resize_shape)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize pixel values to [0, 1]
                frame = frame / 255.0
                frames.append(frame)
            else:
                # Add a blank frame if reading fails
                frames.append(np.zeros((self.resize_shape[1], self.resize_shape[0], 3)))

        cap.release()

        # Convert to numpy array and transpose
        frames_array = np.array(frames)
        # Convert from [num_frames, height, width, channels] to [num_frames, channels, height, width]
        frames_array = np.transpose(frames_array, (0, 3, 1, 2)).astype(np.float32)

        # Convert to tensor
        frames_tensor = torch.from_numpy(frames_array)
        label_tensor = torch.tensor(self.encoded_labels[idx], dtype=torch.long)

        return frames_tensor, label_tensor





if __name__ ==  "__main__":
    base_path = r"E:\TCN\DATA"
    csv_paths = r"E:\TCN\DATA\videos_train_shuffled.csv"

    dataset = VideoDataset(csv_paths,base_path=base_path)

    for i in range(100):
        frames, label = dataset[i]
        print(f"Frames tensor shape: {frames.shape}")
        print(f"Label tensor shape: {label.shape}")
        print(f"Label value: {label.item()}")

        # Optionally, check value ranges
        print(f"Frame min value: {frames.min()}")
        print(f"Frame max value: {frames.max()}")













































