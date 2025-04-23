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
    # U, S, V^T each (B, 3, 3)
    U, _, Vt = torch.linalg.svd(rotmat)           
    # Shape (B,)
    det_uvt = torch.det(U @ Vt)                       
    # Shape (B, 3, 3)
    diag_correction = torch.eye(3, device=rotmat.device).unsqueeze(0).repeat(batch_size, 1, 1)
    diag_correction[:, 2, 2] = det_uvt
    R = U @ diag_correction @ Vt                      
    R_np = R.detach().cpu().numpy()

    # Use scipy's Rotation to get quaternion (x, y, z, w)
    r = Rotation.from_matrix(R_np)
    quat_np_xyzw = r.as_quat()                        

    # Reorder to (w, x, y, z)
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
    diff = pred_rotmat - target_rotmat
    chordal_dist = torch.sum(diff**2, dim=(1, 2))
        
    return chordal_dist
    # return torch.sqrt(torch.sum((pred_rotmat - target_rotmat) ** 2, dim=-1))


def geodesic_distance(pred_rotmat, target_rotmat):
    """
    Geodesic distance between rotation matrices.

    Args:
        pred_rotmat (torch.Tensor): Predicted rotation matrices
        target_rotmat (torch.Tensor): Target rotation matrices

    Returns:
        torch.Tensor: Geodesic distance
    """

    # Compute the relative rotation matrix: R_diff = R_pred^T * R_target
    R_diff = torch.matmul(pred_rotmat.transpose(1, 2), target_rotmat)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -0.98, 0.98)
    angle = torch.acos(cos_angle)
    return angle


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
    u, s, v = torch.linalg.svd(mat)
    # Restore rotation matrix: R = U * V^T
    vt = v.transpose(1, 2)
    det = torch.det(u @ vt).view(batch_size, 1, 1)
    VT = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), dim = 1)
    rotmat = u @ VT
    
    return rotmat

def quaternion_to_rotmat(quaternion):
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternion (torch.Tensor): Batch of quaternions of shape (B, 4)

    Returns:
        torch.Tensor: Batch of rotation matrices of shape (B, 3, 3)
    """

    # Really horrible code... but it works with graident flowing
    quaternion = F.normalize(quaternion, p=2, dim=1)
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    one_matrix = torch.ones_like(w)

    R1_1 = one_matrix - 2 * (y*y + z* z)
    R1_2 = 2 * (x*y -w*z)
    R1_3 = 2 * (x*z + w*y)
    R2_1 = 2 * (x*y + w*z)
    R2_2 = one_matrix-2 * (x*x+z*z)
    R2_3 = 2 * (y*z-w*x)
    R3_1 = 2 * (x*z-w*y)
    R3_2 = 2 * (y*z+w*x)
    R3_3 = one_matrix - 2 * (x * x + y * y)
    row1 = torch.stack([R1_1, R1_2, R1_3], dim=1)
    row2 = torch.stack([R2_1, R2_2, R2_3], dim=1)
    row3 = torch.stack([R3_1, R3_2, R3_3], dim=1)
    rotmat = torch.stack([row1, row2, row3], dim=1)
    return rotmat


def euler_to_rotmat(euler):
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
    # Rot_x 
    R_x = torch.stack([
        torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=1),
        torch.stack([torch.zeros_like(cx), cx, -sx], dim=1),
        torch.stack([torch.zeros_like(cx), sx, cx], dim=1)
    ], dim=1)
    # Rot_y
    R_y = torch.stack([
        torch.stack([cy, torch.zeros_like(cy), sy], dim=1),
        torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=1),
        torch.stack([-sy, torch.zeros_like(cy), cy], dim=1)
    ], dim=1)
    # Rot_z
    R_z = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)], dim=1),
        torch.stack([sz, cz, torch.zeros_like(cz)], dim=1),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=1)
    ], dim=1)
    # Compose rotations: using extrinsic rotations R = R_z * R_y * R_x
    rotmat = torch.matmul(torch.matmul(R_z, R_y), R_x)
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
    # We could use with same as loss function
    metrics = {
        'l1': l1_dist.mean(),
        'l2': l2_dist.mean(),
        'chordal': chord_dist.mean(),
        'geodesic': geod_dist.mean(),
    }
    
    return metrics