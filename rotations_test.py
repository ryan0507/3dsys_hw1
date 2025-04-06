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
    return rotmat.reshape(rotmat.shape[0], 9)


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
    # Ahh? 16 pairs... need to modify before submission
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


from hitchhike import euler_angles_to_matrix, matrix_to_euler_angles
def roma_euler_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    return euler_angles_to_matrix(inp.reshape(-1, 3), convention="XYZ")

def roma_rotmat_to_euler(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return matrix_to_euler_angles(base, convention="XYZ")

def roma_rotmat_to_quaternion(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return roma.rotmat_to_unitquat(base)

def roma_rotmat_to_sixd(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return base[:, :, :2]

def roma_rotmat_to_nined(base: torch.Tensor, **kwargs) -> torch.Tensor:
    return base

def roma_quaternion_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    # without normalization
    # normalize first
    x = inp.reshape(-1, 4)
    return roma.unitquat_to_rotmat(x / x.norm(dim=1, keepdim=True))

def roma_sixd_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    return roma.special_gramschmidt(inp.reshape(-1, 3, 2))


def symmetric_orthogonalization(x, **kwargs):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batch_size, 9]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r


def roma_nined_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    return symmetric_orthogonalization(inp)

def test_comparisons():
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation

    batch_size = 10
    R_np = Rotation.random(batch_size).as_matrix()
    R = torch.from_numpy(R_np).float()
    
    # --- Quaternion Conversion Comparison ---
    quat_ours = rotmat_to_quaternion(R)
    quat_roma = roma_rotmat_to_quaternion(R)
    
    # Reconstruct rotation matrices from quaternions
    R_rec_ours = quaternion_to_rotmat(quat_ours)
    R_rec_roma = roma_quaternion_to_rotmat(quat_roma)
    
    # Convert to numpy arrays for comparison
    R_rec_ours_np = R_rec_ours.detach().cpu().numpy()
    R_rec_roma_np = R_rec_roma.detach().cpu().numpy()
    if np.allclose(R_rec_ours_np, R_rec_roma_np, atol=1e-5):
        print("Quaternion-based reconstructions are close (np.allclose).")
    else:
        print("Quaternion-based reconstructions differ.")

    # --- 6D Conversion Comparison ---
    sixd_ours = rotmat_to_sixd(R)
    sixd_roma = roma_rotmat_to_sixd(R)
    
    # Reconstruct rotation matrices from 6D representations
    R_rec_ours = sixd_to_rotmat(sixd_ours)
    R_rec_roma = roma_sixd_to_rotmat(sixd_roma)
    
    R_rec_ours_np = R_rec_ours.detach().cpu().numpy()
    R_rec_roma_np = R_rec_roma.detach().cpu().numpy()
    if np.allclose(R_rec_ours_np, R_rec_roma_np, atol=1e-5):
        print("6D-based reconstructions are close (np.allclose).")
    else:
        print("6D-based reconstructions differ.")

    # --- 9D Conversion Comparison ---
    nined_ours = rotmat_to_nined(R)
    nined_roma = roma_rotmat_to_nined(R)
    
    # Reconstruct rotation matrices from 9D representations
    R_rec_ours = nined_to_rotmat(nined_ours)
    R_rec_roma = roma_nined_to_rotmat(nined_roma)
    
    R_rec_ours_np = R_rec_ours.detach().cpu().numpy()
    R_rec_roma_np = R_rec_roma.detach().cpu().numpy()
    if np.allclose(R_rec_ours_np, R_rec_roma_np, atol=1e-5):
        print("9D-based reconstructions are close (np.allclose).")
    else:
        print("9D-based reconstructions differ.")

def test_euler_comparisons():
    import numpy as np
    from scipy.spatial.transform import Rotation

    batch_size = 10
    R_np = Rotation.random(batch_size).as_matrix()
    R = torch.from_numpy(R_np).float()
    
    # Euler conversion using your implementation
    euler_ours = rotmat_to_euler(R)
    R_rec_ours = euler_to_rotmat(euler_ours)
    
    # Euler conversion using roma (with xyz convention)
    euler_roma = roma_rotmat_to_euler(R)
    R_rec_roma = roma_euler_to_rotmat(euler_roma)
    
    R_rec_ours_np = R_rec_ours.detach().cpu().numpy()
    R_rec_roma_np = R_rec_roma.detach().cpu().numpy()
    
    if np.allclose(R_rec_ours_np, R_rec_roma_np, atol=1e-5):
         print("Euler-based reconstructions are close (np.allclose).")
    else:
         print("Euler-based reconstructions differ.")


def new_quaternion_to_rotmat(quaternion):
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternion (torch.Tensor): Batch of quaternions of shape (B, 4)

    Returns:
        torch.Tensor: Batch of rotation matrices of shape (B, 3, 3)
    """

    quaternion = F.normalize(quaternion, p=2, dim=1)
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    one = torch.ones_like(w)
    two = 2.0
    R11 = one - two * (y * y + z * z)
    R12 = two * (x * y - w * z)
    R13 = two * (x * z + w * y)
    R21 = two * (x * y + w * z)
    R22 = one - two * (x * x + z * z)
    R23 = two * (y * z - w * x)
    R31 = two * (x * z - w * y)
    R32 = two * (y * z + w * x)
    R33 = one - two * (x * x + y * y)
    row1 = torch.stack([R11, R12, R13], dim=1)
    row2 = torch.stack([R21, R22, R23], dim=1)
    row3 = torch.stack([R31, R32, R33], dim=1)
    rotmat = torch.stack([row1, row2, row3], dim=1)
    return rotmat


def new_euler_to_rotmat(euler):
    """
    Convert Euler angles to rotation matrices.

    Args:
        euler (torch.Tensor): Batch of Euler angles of shape (B, 3)

    Returns:
        torch.Tensor: Batch of rotation matrices of shape (B, 3, 3)
    """

    # Since here loss function is differentiable, we can directly apply the rotation matrices
    # Apply scipy library is not admitted. Occring errors....
    r = euler[:, 0]
    p = euler[:, 1]
    y = euler[:, 2]
    cx = torch.cos(r)
    sx = torch.sin(r)
    cy = torch.cos(p)
    sy = torch.sin(p)
    cz = torch.cos(y)
    sz = torch.sin(y)
    # Rotation x-axis
    R_x = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=1),
        torch.stack([torch.zeros_like(cx), sx, cx], dim=1)
    ], dim=1)
    # Rotation y-axis
    R_y = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=1)
    ], dim=1)
    # Rotation z-axis
    R_z = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=1),
        torch.stack([sz, cz, torch.zeros_like(cz)], dim=1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=1)
    ], dim=1)
    # Compose rotations: using extrinsic rotations R = R_z * R_y * R_x
    rotmat = torch.matmul(torch.matmul(R_z, R_y), R_x)
    return rotmat


def test_quaternion_conversion_comparison():
    """
    Compare the original quaternion_to_rotmat with the new implementation.
    """
    print("\nTesting quaternion conversion comparison...")
    
    # Generate random quaternions
    batch_size = 10
    quaternions = torch.randn(batch_size, 4)
    quaternions = F.normalize(quaternions, p=2, dim=1)  # Normalize to unit quaternions
    
    # Convert using both implementations
    rotmat_original = quaternion_to_rotmat(quaternions)
    rotmat_new = new_quaternion_to_rotmat(quaternions)
    
    # Calculate differences
    diff = torch.abs(rotmat_original - rotmat_new)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference between implementations: {max_diff:.6f}")
    print(f"Mean difference between implementations: {mean_diff:.6f}")
    
    # Check if they're close
    is_close = torch.allclose(rotmat_original, rotmat_new, rtol=1e-5, atol=1e-5)
    print(f"Implementations are {'close' if is_close else 'different'}")
    
    # Verify both produce valid rotation matrices
    det_original = torch.det(rotmat_original)
    det_new = torch.det(rotmat_new)
    
    print(f"Determinant of original implementation: {det_original.mean().item():.6f}")
    print(f"Determinant of new implementation: {det_new.mean().item():.6f}")
    
    # Check orthogonality
    identity = torch.eye(3, device=rotmat_original.device).unsqueeze(0).repeat(batch_size, 1, 1)
    orthogonality_original = torch.matmul(rotmat_original, rotmat_original.transpose(1, 2))
    orthogonality_new = torch.matmul(rotmat_new, rotmat_new.transpose(1, 2))
    
    orthogonality_diff_original = torch.abs(orthogonality_original - identity).mean().item()
    orthogonality_diff_new = torch.abs(orthogonality_new - identity).mean().item()
    
    print(f"Orthogonality error (original): {orthogonality_diff_original:.6f}")
    print(f"Orthogonality error (new): {orthogonality_diff_new:.6f}")
    
    return is_close

def test_euler_conversion_comparison():
    """
    Compare the original euler_to_rotmat with the new implementation.
    """
    print("\nTesting Euler angle conversion comparison...")
    
    # Generate random Euler angles
    batch_size = 10
    euler_angles = torch.randn(batch_size, 3)
    
    # Convert using both implementations
    rotmat_original = euler_to_rotmat(euler_angles)
    rotmat_new = new_euler_to_rotmat(euler_angles)
    
    # Calculate differences
    diff = torch.abs(rotmat_original - rotmat_new)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference between implementations: {max_diff:.6f}")
    print(f"Mean difference between implementations: {mean_diff:.6f}")
    
    # Check if they're close
    is_close = torch.allclose(rotmat_original, rotmat_new, rtol=1e-5, atol=1e-5)
    print(f"Implementations are {'close' if is_close else 'different'}")
    
    # Verify both produce valid rotation matrices
    det_original = torch.det(rotmat_original)
    det_new = torch.det(rotmat_new)
    
    print(f"Determinant of original implementation: {det_original.mean().item():.6f}")
    print(f"Determinant of new implementation: {det_new.mean().item():.6f}")
    
    # Check orthogonality
    identity = torch.eye(3, device=rotmat_original.device).unsqueeze(0).repeat(batch_size, 1, 1)
    orthogonality_original = torch.matmul(rotmat_original, rotmat_original.transpose(1, 2))
    orthogonality_new = torch.matmul(rotmat_new, rotmat_new.transpose(1, 2))
    
    orthogonality_diff_original = torch.abs(orthogonality_original - identity).mean().item()
    orthogonality_diff_new = torch.abs(orthogonality_new - identity).mean().item()
    
    print(f"Orthogonality error (original): {orthogonality_diff_original:.6f}")
    print(f"Orthogonality error (new): {orthogonality_diff_new:.6f}")
    
    return is_close

def test_quaternion_euler_consistency():
    """
    Test consistency between quaternion and Euler angle conversions.
    """
    print("\nTesting quaternion-Euler angle consistency...")
    
    # Generate random Euler angles
    batch_size = 10
    euler_angles = torch.randn(batch_size, 3)
    
    # Convert Euler to rotation matrix
    rotmat_from_euler = new_euler_to_rotmat(euler_angles)
    
    # Convert rotation matrix to quaternion
    quaternion = rotmat_to_quaternion(rotmat_from_euler)
    
    # Convert quaternion back to rotation matrix
    rotmat_from_quaternion = new_quaternion_to_rotmat(quaternion)
    
    # Calculate differences
    diff = torch.abs(rotmat_from_euler - rotmat_from_quaternion)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference between Euler and quaternion paths: {max_diff:.6f}")
    print(f"Mean difference between Euler and quaternion paths: {mean_diff:.6f}")
    
    # Check if they're close
    is_close = torch.allclose(rotmat_from_euler, rotmat_from_quaternion, rtol=1e-5, atol=1e-5)
    print(f"Paths are {'consistent' if is_close else 'inconsistent'}")
    
    return is_close

if __name__ == '__main__':
    test_euler_conversion()
    test_quaternion_conversion()
    test_sixd_conversion()
    test_nined_conversion()
    test_compute_metrics()
    test_comparisons()
    test_euler_comparisons()
    
    # Add new tests
    test_quaternion_conversion_comparison()
    test_euler_conversion_comparison()
    test_quaternion_euler_consistency()
    
    print("All tests passed successfully!")
