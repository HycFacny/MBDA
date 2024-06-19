from pathlib import Path

import os
import json
import numpy as np
from PIL import ImageFile
import torch
import torch.nn as nn
import cv2

import utils.sample_generator.samples as samples


def load_image_with_anno(image_root):
    images_path = list(Path(image_root).rglob("*.jpg"))
    image = cv2.imread(images_path[0])
    # print(images)
    
    for part in [0, 1, 2]:
        annotation_file = os.path.join(root, split, 'run{}.json'.format(part))
        print("loading", annotation_file)
        with open(annotation_file) as f:
            samples = json.load(f)
            for sample in samples:
                sample["image_path"] = os.path.join(root, self.split, 'run{}'.format(part), sample['name'])
            all_samples.extend(samples)
    
    
    
if __name__ == '__main__':
    load_image_with_anno('/home/huangyuchao/projects/datasets/SURREAL/train/run0/01_01', '')
    