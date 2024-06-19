import os
import cv2
import numpy as np
import random


def rotation_augumentation(image, keypoint2d):
    width, height = image.shape[:2]
    random.seed(0)
    degree_list = [i * 30 for i in range(0, 12)]
    rotation = random.sample(degree_list, 1)
    
    transmat = get_affine_transmat(width, height, rotation)
    keypoint2d = joint_affine_transform(keypoint2d, transmat)
    
    return image, keypoint2d


def get_affine_transmat(src_w, src_h, rotation, inv=0):
    dst_w, dst_h = src_w, src_h
    rotation_rad = np.pi * rotation / 180
    center = np.zeros((2), dtype=np.float32)
    center[0], center[1] = (src_w - 1) * .5, (src_h - 1) * .5
    
    src_direction = get_direction([0, (src_w - 1) * -.5], rotation_rad) # for symmetric random, we can either let 2rd point rotate + or -
    dst_direction = np.array([0, (dst_w - 1) * -.5], dtype=np.float32)  # point after rotating in dst
    
    # need 3 points in both to calc affine transform matrix
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    # src may be shifted, but we only consider centered dst map
    src[0, :] = center
    src[1, :] = center + src_direction
    
    dst[0, :] = [(dst_w - 1) * .5, (dst_h - 1) * .5]
    dst[1, :] = np.array([ (dst_w - 1) * .5, (dst_h - 1) * .5 ]) + dst_direction

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])        # get 3rd point use same method to ensure symmetrically
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])        # get 3rd point use same method to ensure symmetrically

    # print(src)
    # print(dst)

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    
    # trans (ndarray([2, 3])): transform matrix with bias
    return trans


# methods for joints after affine op
def joint_affine_transform(joint, transmat):
    '''
        joint: [2]
        transmat: [2, 2]
    '''
    new_joint = np.array([ joint[0], joint[1], 1.], dtype=np.float32).T
    new_joint = np.dot(transmat, new_joint)
    
    return new_joint[:2]


# flip joint coordinates
def fliplr_joints(joints, joints_vis, width, joint_pairs):
    # horizontal flip
    joints[:, 0] = width - joints[:, 0] - 1

    for pair in joint_pairs:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :].copy(), joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :].copy(), joints_vis[pair[0], :].copy()

    return joints * joints_vis, joints_vis


def flip_back(output_flipped, joint_pairs):
    """
    Flip back flipped outputs

    Args:
        output_flipped (ndarray([batch_size, num_joints, height, width])): Flipped heatmaps outputed by network
        joint_pairs (list[num_pairs, 2]): Symmetric joint pairs

    Returns:
        output_flipped (ndarray([batch_size, num_joints, height, width])): After flipping back
    """
    assert output_flipped.ndim == 4, 'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    # swap
    for pair in joint_pairs:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


#########################################
# get location after rotating
def get_direction(src_point, rotation_rad):
    SiN, CoS = np.sin(rotation_rad), np.cos(rotation_rad)
    return [
        src_point[0] * CoS - src_point[1] * SiN,
        src_point[0] * SiN + src_point[1] * CoS
    ]


# get location 
def get_3rd_point(a, b):
    return np.array([
        b[0] + b[1] - a[1],
        b[1] + a[0] - b[0]
    ], dtype=np.float32)
