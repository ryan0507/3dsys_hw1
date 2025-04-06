import numpy as np
import torch
from torch.utils.data import Dataset
import os
from rotations import (
    rotmat_to_euler,
    rotmat_to_quaternion,
    rotmat_to_sixd,
    rotmat_to_nined,
)


class PointCloudAlignmentDataset(Dataset):
    def __init__(self, mode, device, representation="quaternion"):
        """
        Dataset for point cloud alignment task.

        Args:
            mode (str): 'train' or 'test'
            device (torch.device): Device to load the data on
            representation (str): Rotation representation to use ('euler', 'quaternion', 'sixd', 'nined')
        """
        path = os.path.join("data", "pcd_dataset")
        pcd_path = os.path.join(path, f"{mode}_point_cloud.npy")  # N, Npcd, 3
        rotated_pcd_path = os.path.join(path, f"rotated_{mode}_point_cloud.npy")
        out_rot_path = os.path.join(path, f"{mode}_rotations.npy")

        if (
            not os.path.exists(pcd_path)
            or not os.path.exists(rotated_pcd_path)
            or not os.path.exists(out_rot_path)
        ):
            raise FileNotFoundError(f"Dataset files not found in {path}")

        self.device = device
        self.representation = representation

        # Load point clouds
        pcd, rotated_pcd = np.load(pcd_path), np.load(rotated_pcd_path)
        self.feature_input = torch.from_numpy(
            np.concatenate((pcd, rotated_pcd), axis=-1)
            .transpose((0, 2, 1))
            .astype(np.float32)
        ).to(self.device)

        # Load rotation matrices
        rot_matrices = torch.from_numpy(np.load(out_rot_path).astype(np.float32)).to(
            self.device
        )

        # Convert rotation matrices to the desired representation
        if representation == "euler":
            self.targets = rotmat_to_euler(rot_matrices)
        elif representation == "quaternion":
            self.targets = rotmat_to_quaternion(rot_matrices)
        elif representation == "sixd":
            self.targets = rotmat_to_sixd(rot_matrices)
        elif representation == "nined":
            self.targets = rotmat_to_nined(rot_matrices)
        else:
            raise ValueError(f"Unknown representation: {representation}")

        self.N = len(self.feature_input)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.feature_input[idx], self.targets[idx]
