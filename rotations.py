import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation

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
    # TODO: Implement conversion from rotation matrices to quaternions
    batch_size = rotmat.shape[0]

    # Step 1 & 2: SVD to ensure valid rotation
    U, _, Vt = torch.linalg.svd(rotmat)               # U, S, V^T each (B, 3, 3)
    det_uvt = torch.det(U @ Vt)                       # (B,)
    diag_correction = torch.eye(3, device=rotmat.device).unsqueeze(0).repeat(batch_size, 1, 1)
    diag_correction[:, 2, 2] = det_uvt
    R = U @ diag_correction @ Vt                      # (B, 3, 3)

    # Step 3: Convert to numpy for SciPy
    R_np = R.detach().cpu().numpy()

    # Use scipy's Rotation to get quaternion (x, y, z, w)
    r = R.from_matrix(R_np)
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
    # TODO: Implement conversion from rotation matrices to 6D representation
    # 회전 행렬의 처음 두 열을 가져옴
    U, _, Vt = torch.linalg.svd(rotmat)  # U, S, V^T each (B, 3, 3)
    det_uvt = torch.det(U @ Vt)          # (B,)
    diag_correction = torch.eye(3, device=rotmat.device).unsqueeze(0).repeat(rotmat.size(0), 1, 1)
    diag_correction[:, 2, 2] = det_uvt
    R = U @ diag_correction @ Vt         # (B, 3, 3)

    # Step 2: Extract columns and flatten
    first_col = R[:, :, 0]              # (B, 3)
    second_col = R[:, :, 1]             # (B, 3)
    sixd = torch.cat([first_col, second_col], dim=1)  # (B, 6)

    return sixd


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
    # TODO: Implement geodesic distance calculation
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
    
    # 6D 표현에서 처음 두 열 추출
    x_raw = sixd[:, 0:3]  # 첫 번째 열
    y_raw = sixd[:, 3:6]  # 두 번째 열
    
    # 첫 번째 열 정규화
    x = F.normalize(x_raw, p=2, dim=1)
    
    # Gram-Schmidt 과정
    # 두 번째 열에서 첫 번째 열의 성분 제거
    z = torch.cross(x, y_raw, dim=1)
    z = F.normalize(z, p=2, dim=1)
    
    # 세 번째 열은 첫 번째와 세 번째 열의 외적
    y = torch.cross(z, x, dim=1)
    
    # 각 열을 회전 행렬로 조합
    x = x.view(batch_size, 3, 1)
    y = y.view(batch_size, 3, 1)
    z = z.view(batch_size, 3, 1)
    
    rotmat = torch.cat([x, y, z], dim=2)  # 크기: (B, 3, 3)
    
    return rotmat


def nined_to_rotmat(nined):
    """
    Convert 9D representation to rotation matrices using SVD.

    Args:
        nined (torch.Tensor): Batch of 9D representations of shape (B, 9)

    Returns:
        torch.Tensor: Batch of rotation matrices of shape (B, 3, 3)
    """
    # TODO: Implement conversion from 9D representation to rotation matrices
    batch_size = nined.shape[0]
    
    # 9D 표현을 3x3 행렬로 변환
    mat = nined.view(batch_size, 3, 3)
    
    # SVD 수행
    u, s, v = torch.svd(mat)
    
    # 회전 행렬 복원: R = U * V^T
    rotmat = torch.bmm(u, v.transpose(1, 2))
    
    # 행렬식이 1이 되도록 보정
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
    
    # PyTorch 텐서를 NumPy 배열로 변환
    # scipy는 (x,y,z,w) 순서를 사용하므로 변환 필요
    quat_np = quaternion.detach().cpu().numpy()
    quat_xyzw = np.concatenate([quat_np[:, 1:], quat_np[:, 0:1]], axis=1)
    
    # scipy를 사용하여 변환
    r = Rotation.from_quat(quat_xyzw)
    rotmat_np = r.as_matrix()
    
    # 결과를 다시 PyTorch 텐서로 변환
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
    
    r = Rotation.from_euler('xyz', euler_np, degrees=False)
    rotmat_np = r.as_matrix()
    
    # 결과를 다시 PyTorch 텐서로 변환
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
    # 회전 표현을 회전 행렬로 변환
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
    
    # 원래 표현에 대한 L1 및 L2 거리 계산
    l1_dist = l1_distance(pred, target)
    l2_dist = l2_distance(pred, target)
    
    # 회전 행렬에 대한 Chordal 및 Geodesic 거리 계산
    chord_dist = chordal_distance(pred_rotmat, target_rotmat)
    geod_dist = geodesic_distance(pred_rotmat, target_rotmat)
    
    # 모든 메트릭을 딕셔너리로 반환
    metrics = {
        'l1_distance': l1_dist.mean().item(),
        'l2_distance': l2_dist.mean().item(),
        'chordal_distance': chord_dist.mean().item(),
        'geodesic_distance': geod_dist.mean().item(),
        'max_l1': l1_dist.max().item(),
        'max_l2': l2_dist.max().item(),
        'max_chordal': chord_dist.max().item(),
        'max_geodesic': geod_dist.max().item(),
        'min_l1': l1_dist.min().item(),
        'min_l2': l2_dist.min().item(),
        'min_chordal': chord_dist.min().item(),
        'min_geodesic': geod_dist.min().item(),
    }
    
    return metrics
