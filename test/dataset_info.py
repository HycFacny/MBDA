import cv2
import os
import json
import numpy as np
from PIL import ImageFile
import torch
import torch.nn as nn
import scipy.io as scio


from tllib.vision.datasets._util import download as download_data, check_exits
from tllib.vision.datasets.keypoint_detection.util import *
from tllib.vision.datasets.keypoint_detection.keypoint_dataset import Body16KeypointDataset
from tllib.vision.transforms.keypoint_detection import *

# from ..utils.sample_generator import samples


ImageFile.LOAD_TRUNCATED_IMAGES = True

def surreal_loads(root, split='train'):
    all_samples = []
    for part in [0, 1, 2]:
        annotation_file = os.path.join(root, 'train', 'run{}.json'.format(part))
        print("loading", annotation_file)
        with open(annotation_file) as f:
            samples = json.load(f)
            for sample in samples:
                sample["image_path"] = os.path.join(root, 'train', 'run{}'.format(part), sample['name'])
            all_samples.extend(samples)
        break
    samples_len = len(all_samples)
    samples_split = min(int(samples_len * 0.2), 3200)
    if split == 'train':
        all_samples = all_samples[samples_split:]
    elif split == 'test':
        all_samples = all_samples[:samples_split]
    joints_index = (7, 4, 1, 2, 5, 8, 0, 9, 12, 15, 20, 18, 13, 14, 19, 21)
    
    average = np.zeros((16, 2), dtype=np.float)
    Asize = np.zeros((2), dtype=np.float)
    for idx in range(samples_split):
        sample = all_samples[idx]
        image_path = sample['image_path']
        keypoint2d = np.array(sample['keypoint2d'])[joints_index, :]
        average += keypoint2d
        image = cv2.imread(image_path)
        Asize[0] += image.shape[0]
        Asize[1] += image.shape[1]

    average /= samples_split
    Asize /= samples_split

    return samples_split, average, Asize


def lsp_loads(root, split='train'):
    samples = []
    annotations = scio.loadmat(os.path.join(root, "joints.mat"))['joints'].transpose((2,1,0))
    for i in range(0, 2000):
        image = "im{0:04d}.jpg".format(i+1)
        annotation = annotations[i]
        samples.append((image, annotation))

    joints_index = (0, 1, 2, 3, 4, 5, 13, 13, 12, 13, 6, 7, 8, 9, 10, 11)
    visible = np.array([1.] * 6 + [0, 0] + [1.] * 8, dtype=np.float32)
    
    num_sample = len(samples)
    average = np.zeros((16, 2), dtype=np.float)
    Asize = np.zeros((2), dtype=np.float)
    for sample in samples:
        image = cv2.imread(os.path.join(root, 'images', sample[0]))
        Asize[0] += image.shape[0]
        Asize[1] += image.shape[1]
        keypoint2d = sample[1][joints_index, :2]
        average += keypoint2d
    
    average /= num_sample
    Asize /= num_sample
    
    return num_sample, average, Asize


if __name__ == '__main__':
    print(surreal_loads('/home/huangyuchao/projects/datasets/SURREAL'))
    # print(lsp_loads('/home/huangyuchao/projects/datasets/LSP'))