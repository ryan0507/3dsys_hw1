import torch
import os


class Config:
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset configuration
        self.data_path = os.path.join("data", "pcd_dataset")
        self.batch_size = 32

        # Training configuration
        self.epochs = 2
        self.learning_rate = 0.001

        # Model configuration
        self.input_dim = (6, 3000)  # (channels, points)

        # Rotation representations
        self.representations = {
            "euler": 3,  # 3D Euler angles
            "quaternion": 4,  # 4D Quaternion
            "sixd": 6,  # 6D representation (first two columns of rotation matrix)
            "nined": 9,  # 9D representation (full rotation matrix)
        }

        # Metrics for evaluation
        self.metrics = ["l1", "l2", "chordal", "geodesic"]

        # Visualization
        self.save_plots = True
        self.plot_dir = "plots"

    def get_output_size(self, representation):
        return self.representations[representation]
