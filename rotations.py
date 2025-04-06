import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation
import roma

# ----------------------
# Rotation conversions
# ----------------------


def rotmat_to_euler(rotmat):
    """
    Convert rotation matrices to Euler angles.

    Args:
        rotmat (torch.Tensor): Batch of rotation matrices of shape (B, 3, 3)

    Returns:
        torch.Tensor: Batch of Euler angles of shape (B, 3)
    """
    # TODO: Implement conversion from rotation matrices to Euler angles
    batch_size = rotmat.shape[0]
    rotmat_np = rotmat.detach().cpu().numpy()
    r = Rotation.from_matrix(rotmat_np)
    euler_np = r.as_euler('xyz', degrees=False)
    euler = torch.from_numpy(euler_np).to(rotmat.device).float()
    
    return euler


def rotmat_to_quaternion(rotmat):
    """
    Convert rotation matrices to quaternions.

    Args:
        rotmat (torch.Tensor): Batch of rotation matrices of shape (B, 3, 3)

    Returns:
        torch.Tensor: Batch of quaternions of shape (B, 4)
    """

    batch_size = rotmat.shape[0]
    U, _, Vt = torch.linalg.svd(rotmat)               # U, S, V^T each (B, 3, 3)
    det_uvt = torch.det(U @ Vt)                       # (B,)
    diag_correction = torch.eye(3, device=rotmat.device).unsqueeze(0).repeat(batch_size, 1, 1)
    diag_correction[:, 2, 2] = det_uvt
    R = U @ diag_correction @ Vt                      # (B, 3, 3)
    R_np = R.detach().cpu().numpy()

    # Use scipy's Rotation to get quaternion (x, y, z, w)
    r = Rotation.from_matrix(R_np)
    quat_np_xyzw = r.as_quat()                        # shape (B, 4)

    # Step 4: Reorder to (w, x, y, z)
    quat_np_wxyz = np.concatenate([
        quat_np_xyzw[:, 3:4],
        quat_np_xyzw[:, 0:1],
        quat_np_xyzw[:, 1:2],
        quat_np_xyzw[:, 2:3]
    ], axis=1)

    quaternion = torch.from_numpy(quat_np_wxyz).to(rotmat.device).float()

    return quaternion


def rotmat_to_sixd(rotmat):
    """
    Convert rotation matrices to 6D representation (first two columns).

    Args:
        rotmat (torch.Tensor): Batch of rotation matrices of shape (B, 3, 3)

    Returns:
        torch.Tensor: Batch of 6D representations of shape (B, 6)
    """


    # Make SVD each matrix in the batch U, S, V^T each (B, 3, 3)
    U, _, Vt = torch.linalg.svd(rotmat)  
    # Shape: (B)
    det_uvt = torch.det(U @ Vt)
    diag_correction = torch.eye(3, device=rotmat.device).unsqueeze(0).repeat(rotmat.size(0), 1, 1)
    diag_correction[:, 2, 2] = det_uvt
    # Shape: (B, 3, 3)
    R_svd = U @ diag_correction @ Vt      

    # Extract columns and flatten
    # Shape: (B, 3)
    first_col = R_svd[:, :, 0]              
    second_col = R_svd[:, :, 1]            
    # Shape: (B, 6)
    sixd_mat = torch.cat([first_col, second_col], dim=1)  

    return sixd_mat


def rotmat_to_nined(rotmat):
    """
    Convert rotation matrices to 9D representation (flattened matrix).

    Args:
        rotmat (torch.Tensor): Batch of rotation matrices of shape (B, 3, 3)

    Returns:
        torch.Tensor: Batch of 9D representations of shape (B, 9)
    """
    # TODO: Implement conversion from rotation matrices to 9D representation
    # 회전 행렬을 평탄화
    batch_size = rotmat.shape[0]
    
     # Step 1: SVD
    U, S, Vt = torch.linalg.svd(rotmat)  # U, S, V^T each have shape (B, 3, 3)

    # Step 2: Enforce a proper rotation by fixing the determinant to +1
    det_uvt = torch.det(U @ Vt)          # Shape (B,)
    diag_correction = torch.eye(3, device=rotmat.device).unsqueeze(0).repeat(rotmat.shape[0], 1, 1)
    diag_correction[:, 2, 2] = det_uvt   # diag(1, 1, det(UV^T)) for each matrix in the batch

    # Construct the orthonormal rotation
    R = U @ diag_correction @ Vt         # Shape (B, 3, 3)

    # Step 3: Flatten to 9D
    nined_mat = R.reshape(rotmat.shape[0], 9)
    
    return nined_mat


# ----------------------
# Metric functions
# ----------------------


def l1_distance(pred, target):
    """
    L1 distance between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted values
        target (torch.Tensor): Target values

    Returns:
        torch.Tensor: L1 distance
    """
    # TODO: Implement L1 distance calculation
    # L1 거리 계산 (절대값 차이의 합)
    return torch.abs(pred - target).mean(dim=-1)


def l2_distance(pred, target):
    """
    L2 distance between predictions and targets.

    Args:
        pred (torch.Tensor): Predicted values
        target (torch.Tensor): Target values

    Returns:
        torch.Tensor: L2 distance
    """
    # TODO: Implement L2 distance calculation
    # L2 거리 계산 (유클리드 거리)
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))


def chordal_distance(pred_rotmat, target_rotmat):
    """
    Chordal distance between rotation matrices.

    Args:
        pred_rotmat (torch.Tensor): Predicted rotation matrices
        target_rotmat (torch.Tensor): Target rotation matrices

    Returns:
        torch.Tensor: Chordal distance
    """
    # TODO: Implement chordal distance calculation
    # 예측 회전 행렬과 목표 회전 행렬의 차이
    diff = pred_rotmat - target_rotmat
    fro_norm = torch.sqrt(torch.sum(diff**2, dim=(1, 2)))
    chordal_dist = fro_norm / torch.sqrt(torch.tensor(2.0, device=pred_rotmat.device))
    
    return chordal_dist


def geodesic_distance(pred_rotmat, target_rotmat):
    """
    Geodesic distance between rotation matrices.

    Args:
        pred_rotmat (torch.Tensor): Predicted rotation matrices
        target_rotmat (torch.Tensor): Target rotation matrices

    Returns:
        torch.Tensor: Geodesic distance
    """

    batch_size = pred_rotmat.shape[0]
    angles = torch.zeros(batch_size, device=pred_rotmat.device)
    
    pred_np = pred_rotmat.detach().cpu().numpy()
    target_np = target_rotmat.detach().cpu().numpy()
    
    for i in range(batch_size):
        r_pred = Rotation.from_matrix(pred_np[i])
        r_target = Rotation.from_matrix(target_np[i])
        r_diff = r_pred * r_target.inv()
        angle_np = r_diff.magnitude()
        angles[i] = torch.tensor(angle_np, device=pred_rotmat.device)
    
    return angles


# ----------------------
# Coordinate transformations
# ----------------------


def sixd_to_rotmat(sixd):
    """
    Convert 6D representation to rotation matrices using Gram-Schmidt orthogonalization.

    Args:
        sixd (torch.Tensor): Batch of 6D representations of shape (B, 6)

    Returns:
        torch.Tensor: Batch of rotation matrices of shape (B, 3, 3)
    """
    # TODO: Implement conversion from 6D representation to rotation matrices
    batch_size = sixd.shape[0]
    
    # Extract the first two columns from the 6D representation
    x_raw = sixd[:, 0:3]  
    y_raw = sixd[:, 3:6]  

    x = F.normalize(x_raw, p=2, dim=1)
    
    # Gram-Schmidt process
    z = torch.cross(x, y_raw, dim=1)
    z = F.normalize(z, p=2, dim=1)

    # Cross product of z and x
    y = torch.cross(z, x, dim=1)
    
    # Combine each column to form a rotation matrix
    x = x.view(batch_size, 3, 1)
    y = y.view(batch_size, 3, 1)
    z = z.view(batch_size, 3, 1)
    
    rot_mat = torch.cat([x, y, z], dim=2)  
    
    return rot_mat


def nined_to_rotmat(nined):
    """
    Convert 9D representation to rotation matrices using SVD.

    Args:
        nined (torch.Tensor): Batch of 9D representations of shape (B, 9)

    Returns:
        torch.Tensor: Batch of rotation matrices of shape (B, 3, 3)
    """

    batch_size = nined.shape[0]    
    # Convert 9D representation to 3 3 matrix
    mat = nined.view(batch_size, 3, 3)
    u, s, v = torch.svd(mat)
    # Restore rotation matrix: R = U * V^T
    rotmat = torch.bmm(u, v.transpose(1, 2))
    # Make determinant 1
    det = torch.det(rotmat).view(batch_size, 1, 1)
    rotmat = rotmat * torch.sign(det).view(batch_size, 1, 1)
    
    return rotmat


def quaternion_to_rotmat(quaternion):
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternion (torch.Tensor): Batch of quaternions of shape (B, 4)

    Returns:
        torch.Tensor: Batch of rotation matrices of shape (B, 3, 3)
    """
    # TODO: Implement conversion from quaternions to rotation matrices
    batch_size = quaternion.shape[0]
    device = quaternion.device
    
    # scipy (x,y,z,w) -> (w,x,y,z)
    quat_np = quaternion.detach().cpu().numpy()
    quat_xyzw = np.concatenate([quat_np[:, 1:], quat_np[:, 0:1]], axis=1)
    r = Rotation.from_quat(quat_xyzw)
    rotmat_np = r.as_matrix()

    # Move numpy to tensor values    
    rotmat = torch.from_numpy(rotmat_np).to(device).float()
    
    return rotmat


def euler_to_rotmat(euler):
    """
    Convert Euler angles to rotation matrices.

    Args:
        euler (torch.Tensor): Batch of Euler angles of shape (B, 3)

    Returns:
        torch.Tensor: Batch of rotation matrices of shape (B, 3, 3)
    """
    # TODO: Implement conversion from Euler angles to rotation matrices

    # Set device to the same
    device = euler.device
    euler_np = euler.detach().cpu().numpy()

    # Convert to rotation matrix
    rotmat_np = Rotation.from_euler('xyz', euler_np, degrees=False).as_matrix()
    rotmat = torch.from_numpy(rotmat_np).to(device).float()
    return rotmat


def compute_metrics(pred, target, representation):
    """
    Compute various metrics between predicted and target rotations.

    Args:
        pred (torch.Tensor): Predicted rotation representation
        target (torch.Tensor): Target rotation representation
        representation (str): Type of rotation representation

    Returns:
        dict: Dictionary of computed metrics
    """
    # TODO: Convert representations to rotation matrices first
    # TODO: Compute L1 and L2 metrics on the representations
    # TODO: Compute chordal and geodesic metrics on the rotation matrices
    # TODO: Return metrics as a dictionary

    # Convert representations to rotation matrices
    if representation == 'euler':
        pred_rotmat = euler_to_rotmat(pred)
        target_rotmat = euler_to_rotmat(target)
    elif representation == 'quaternion':
        pred_rotmat = quaternion_to_rotmat(pred)
        target_rotmat = quaternion_to_rotmat(target)
    elif representation == 'sixd':
        pred_rotmat = sixd_to_rotmat(pred)
        target_rotmat = sixd_to_rotmat(target)
    elif representation == 'nined':
        pred_rotmat = nined_to_rotmat(pred)
        target_rotmat = nined_to_rotmat(target)
    elif representation == 'rotmat':
        pred_rotmat = pred
        target_rotmat = target
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    # Compute L1 and L2 metrics on the representations
    l1_dist = l1_distance(pred, target)
    l2_dist = l2_distance(pred, target)
    
    chord_dist = chordal_distance(pred_rotmat, target_rotmat)
    geod_dist = geodesic_distance(pred_rotmat, target_rotmat)
    
    # Return metrics as a dictionary
    metrics = {
        'l1_distance': l1_dist.mean().item(),
        'l2_distance': l2_dist.mean().item(),
        'chordal_distance': chord_dist.mean().item(),
        'geodesic_distance': geod_dist.mean().item(),
    }
    
    return metrics


# Test code comparing with roma

def test_euler_conversion():
    batch_size = 10
    # Generate a batch of random rotation matrices
    R_np = Rotation.random(batch_size).as_matrix()
    R = torch.from_numpy(R_np).float()
    
    # Convert rotation matrices -> Euler angles -> rotation matrices
    euler = rotmat_to_euler(R)
    R_rec = euler_to_rotmat(euler)
    
    # Compute geodesic error between original and recovered rotations
    error = geodesic_distance(R, R_rec)
    assert error.mean() < 1e-5, f"Euler conversion error too high: {error.mean().item()}"
    print("Euler conversion test passed.")


def test_quaternion_conversion():
    batch_size = 10
    R_np = Rotation.random(batch_size).as_matrix()
    R = torch.from_numpy(R_np).float()
    
    # Convert rotation matrices -> quaternion -> rotation matrices
    quat = rotmat_to_quaternion(R)
    R_rec = quaternion_to_rotmat(quat)
    
    error = geodesic_distance(R, R_rec)
    assert error.mean() < 1e-5, f"Quaternion conversion error too high: {error.mean().item()}"
    print("Quaternion conversion test passed.")


def test_sixd_conversion():
    batch_size = 10
    R_np = Rotation.random(batch_size).as_matrix()
    R = torch.from_numpy(R_np).float()
    
    # Convert rotation matrices -> 6D representation -> rotation matrices
    sixd = rotmat_to_sixd(R)
    R_rec = sixd_to_rotmat(sixd)
    
    error = geodesic_distance(R, R_rec)
    assert error.mean() < 1e-5, f"6D conversion error too high: {error.mean().item()}"
    print("6D conversion test passed.")


def test_nined_conversion():
    batch_size = 10
    R_np = Rotation.random(batch_size).as_matrix()
    R = torch.from_numpy(R_np).float()
    
    # Convert rotation matrices -> 9D representation -> rotation matrices
    nined = rotmat_to_nined(R)
    R_rec = nined_to_rotmat(nined)
    
    error = geodesic_distance(R, R_rec)
    assert error.mean() < 1e-5, f"9D conversion error too high: {error.mean().item()}"
    print("9D conversion test passed.")

def get_roma_func(alternatives):
    """
    Select the first available function from the roma module.
    
    Args:
        alternatives (list): List of possible function names.
        
    Returns:
        function: The function from the roma module.
    
    Raises:
        AttributeError: If none of the alternatives exist.
    """
    for alt in alternatives:
        if hasattr(roma, alt):
            return getattr(roma, alt)
    raise AttributeError(f"Roma library does not have any of the following attributes: {alternatives}")

def test_roma_consistency():
    batch_size = 10
    R_np = Rotation.random(batch_size).as_matrix()
    R = torch.from_numpy(R_np).float()

    # --- Euler comparison ---
    # Try alternatives for Euler conversion function.
    euler_func = get_roma_func(['matrix2euler', 'euler_from_matrix', 'rotmat2euler'])
    euler_custom = rotmat_to_euler(R).detach().cpu().numpy()
    # Some functions might require an 'order' parameter.
    try:
        euler_roma = np.array([euler_func(r, order='xyz') for r in R_np])
    except TypeError:
        euler_roma = np.array([euler_func(r) for r in R_np])
    diff_euler = np.abs(euler_custom - euler_roma)
    assert diff_euler.max() < 1e-5, f"Roma Euler conversion mismatch: {diff_euler.max()}"
    print("Roma Euler conversion test passed.")
    
    # --- Quaternion comparison ---
    quat_func = get_roma_func(['matrix2quat', 'quaternion_from_matrix', 'rotmat2quat'])
    quat_custom = rotmat_to_quaternion(R).detach().cpu().numpy()
    quat_roma = np.array([quat_func(r) for r in R_np])
    # Adjust for sign ambiguity
    for i in range(batch_size):
        if np.linalg.norm(quat_custom[i] + quat_roma[i]) < np.linalg.norm(quat_custom[i] - quat_roma[i]):
            quat_roma[i] = -quat_roma[i]
    diff_quat = np.abs(quat_custom - quat_roma)
    assert diff_quat.max() < 1e-5, f"Roma quaternion conversion mismatch: {diff_quat.max()}"
    print("Roma quaternion conversion test passed.")
    
    # --- 6D comparison ---
    sixd_func = get_roma_func(['matrix2six', 'sixd_from_matrix', 'rotmat2six'])
    sixd_custom = rotmat_to_sixd(R).detach().cpu().numpy()
    sixd_roma = np.array([sixd_func(r) for r in R_np])
    diff_sixd = np.abs(sixd_custom - sixd_roma)
    assert diff_sixd.max() < 1e-5, f"Roma 6D conversion mismatch: {diff_sixd.max()}"
    print("Roma 6D conversion test passed.")
    
    # --- 9D comparison ---
    nined_func = get_roma_func(['matrix2nine', 'nined_from_matrix', 'rotmat2nined'])
    nined_custom = rotmat_to_nined(R).detach().cpu().numpy()
    nined_roma = np.array([nined_func(r) for r in R_np])
    diff_nined = np.abs(nined_custom - nined_roma)
    assert diff_nined.max() < 1e-5, f"Roma 9D conversion mismatch: {diff_nined.max()}"
    print("Roma 9D conversion test passed.")


def test_compute_metrics():
    batch_size = 5
    R_np = Rotation.random(batch_size).as_matrix()
    R = torch.from_numpy(R_np).float()
    
    # Create a slightly perturbed target by adding small noise and re-orthonormalizing
    noise = 0.01 * torch.randn_like(R)
    R_noisy = R + noise
    U, _, Vt = torch.linalg.svd(R_noisy)
    R_target = torch.bmm(U, Vt.transpose(1, 2))
    
    # Test using the rotation matrix representation directly
    metrics = compute_metrics(R, R_target, 'rotmat')
    print("Computed metrics (rotmat representation):", metrics)
    
    # (You could similarly test for 'euler', 'quaternion', 'sixd', or 'nined')
    print("Metrics computation test passed.")


if __name__ == '__main__':
    test_euler_conversion()
    test_quaternion_conversion()
    test_sixd_conversion()
    test_nined_conversion()
    test_roma_consistency()
    test_compute_metrics()
    
    print("All tests passed successfully!")
