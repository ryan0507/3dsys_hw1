import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.spatial.distance import cdist
import torch
import torch.optim as optim

# Load MNIST dataset used sklearn.datasets
mnist = fetch_openml('mnist_784')

# print(mnist)
y_data = mnist.target.to_numpy().astype(int)
X_data = mnist.data.to_numpy()

# Sample selection: select at least 100 sample per digit
sample_size = 200
class_indices_per_digit = []
for digit in range(10):
    digit_indices = np.where(y_data == digit)[0][:sample_size]
    class_indices_per_digit.extend(digit_indices)
sampled_indices = np.array(class_indices_per_digit)

X_sample = X_data[sampled_indices]
y_sample = y_data[sampled_indices]

# Preprocessing : Normalize pixel values to [0, 1]
X_sample = X_sample / 255.0
n_samples = X_sample.shape[0]

# Compute the pairwise Euclidean distance matrix D
D = cdist(X_sample, X_sample, metric='euclidean')

# Classical MDS

H_matrix = np.eye(n_samples) - (1/n_samples) * np.ones((n_samples,1)) @ np.ones((n_samples,1)).T
D_squared = D**2
B_matrix = -0.5 * H_matrix.dot(D_squared).dot(H_matrix)

# Eigen Decomposition process for the classical MDS
eigvals, eigvecs = np.linalg.eigh(B_matrix)
idx_sorted = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx_sorted]
eigvecs = eigvecs[:, idx_sorted]
lambda_1, lambda_2 = eigvals[0], eigvals[1]
v1, v2 = eigvecs[:, 0], eigvecs[:, 1]

# Compute the 2D embedding
X_mds_classical = np.column_stack((np.sqrt(lambda_1) * v1, np.sqrt(lambda_2) * v2))

dist_torch = torch.tensor(D, dtype=torch.float32)  
X_torch = torch.randn(n_samples, 2, requires_grad=True)  # Random init for 2D embedding
optimizer = optim.Adam([X_torch], lr=0.01)
# Iteration settings for training gradient descent methods
num_iterations = 2000  


for iter in range(num_iterations):
    optimizer.zero_grad()
    diff = X_torch.unsqueeze(1) - X_torch.unsqueeze(0)
    
    # Add epsilon for the stability
    dist_est = torch.sqrt(torch.sum(diff**2, dim=2) + 1e-9) 
    loss = torch.sum((dist_est - dist_torch)**2)
    loss.backward()
    optimizer.step()

X_mds_metric = X_torch.detach().numpy()

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
scatter1 = axs[0].scatter(X_mds_classical[:, 0], X_mds_classical[:, 1],
                          c=y_sample, cmap='tab10', s=5)
axs[0].set_title("Classical MDS")
axs[0].axis('equal')
scatter2 = axs[1].scatter(X_mds_metric[:, 0], X_mds_metric[:, 1],
                          c=y_sample, cmap='tab10', s=5)
axs[1].set_title("Metric MDS")
axs[1].axis('equal')

plt.colorbar(scatter2, ax=axs[1], fraction=0.03)
plt.tight_layout()
plt.savefig('prob2.png')