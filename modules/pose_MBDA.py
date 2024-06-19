from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tllib.modules.gl import WarmStartGradientLayer
from tllib.utils.metric.keypoint_detection import get_max_preds


# (0, 1, 2, 3, 4, 5, 13, 13, 12, 13, 6,  7,  8,  9,  10, 11) lsp
# (7, 4, 1, 2, 5, 8, 0,  9,  12, 15, 20, 18, 13, 14, 19, 21) surreal

'''
0: Right ankle
1: Right knee
2: Right hip
3: Left hip
4: Left knee
5: Left ankle
6: Head top
7: Head top
8: Neck
9: Head top
10:Right wrist
11:Right elbow
12:Right shoulder
13:Left shoulder
14:Left elbow
15:Left wrist

upper = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
lower = (0, 1, 2, 3, 4, 5)
left  = (3, 4, 5, 6, 7, 8, 9, 13, 14, 15)
right = (0, 1, 2, 6, 7, 8, 9, 10, 11, 12)

arms =  (0, 1, 2, 3, 4, 5) # r ankle, knee, hip; l
legs =  (10, 11, 12, 13, 14, 15) # r wrist, elbow, shoulder; l
heads = (6, 7, 8, 9) # head top, neck

'''


class FastPseudoLabelGenerator2d(nn.Module):
    def __init__(self, sigma=2, m=4):
        super().__init__()
        self.sigma = sigma
        self.upper_body_index = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
        self.lower_body_index = (0, 1, 2, 3, 4, 5)
        self.mask_num = m
    
    def forward(self, heatmap: torch.Tensor):
        heatmap = heatmap.detach()
        height, width = heatmap.shape[-2:]
        idx = heatmap.flatten(-2).argmax(dim=-1) # B, K
        pred_h, pred_w = idx.div(width, rounding_mode='floor'), idx.remainder(width) # B, K
        delta_h = torch.arange(height, device=heatmap.device) - pred_h.unsqueeze(-1) # B, K, H
        delta_w = torch.arange(width, device=heatmap.device) - pred_w.unsqueeze(-1) # B, K, W
        gaussian = (delta_h.square().unsqueeze(-1) + delta_w.square().unsqueeze(-2)).div(-2 * self.sigma * self.sigma).exp() # B, K, H, W
        ground_truth = F.threshold(gaussian, threshold=1e-2, value=0.)

        
        ground_false = (ground_truth.sum(dim=1, keepdim=True) - ground_truth).clamp(0., 1.)
        return ground_truth, ground_false


class PseudoLabelGeneratorBase(nn.Module):
    def __init__(self, num_keypoints, height=64, width=64, sigma=2):
        super(PseudoLabelGeneratorBase, self).__init__()
        self.height = height
        self.width = width
        self.sigma = sigma

        heatmaps = np.zeros((width, height, height, width), dtype=np.float32)

        tmp_size = sigma * 3
        for mu_x in range(width):
            for mu_y in range(height):
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], width)
                img_y = max(0, ul[1]), min(br[1], height)

                heatmaps[mu_x][mu_y][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        self.heatmaps = heatmaps
    
    def forward(self, y):
        raise NotImplementedError


class PseudoLabelGenerator2d(PseudoLabelGeneratorBase):
    def __init__(self, num_keypoints, height=64, width=64, sigma=2):
        super(PseudoLabelGenerator2d, self).__init__(
            num_keypoints, height, width, sigma
        )
        self.false_matrix = 1. - np.eye(num_keypoints, dtype=np.float32)


    def forward(self, y):
        B, K, H, W = y.shape
        y = y.detach()
        preds, max_vals = get_max_preds(y.cpu().numpy())  # B x K x (x, y)
        preds = preds.reshape(-1, 2).astype(np.int)
        ground_truth = self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W).copy()

        ground_false = ground_truth.reshape(B, K, -1).transpose((0, 2, 1))
        ground_false = ground_false.dot(self.false_matrix).clip(max=1., min=0.).transpose((0,    2, 1)).reshape(B, K, H, W).copy()
        return torch.from_numpy(ground_truth).to(y.device), torch.from_numpy(ground_false).to(y.device)


class PseudoLabelGenerator2dHalfbody(PseudoLabelGeneratorBase):
    def __init__(self, num_keypoints, height=64, width=64, sigma=2, o=0):
        super(PseudoLabelGenerator2dHalfbody, self).__init__(
            num_keypoints, height, width, sigma
        )
        self.body_part_index = []       # 0 : upper,   1 : lower, 2: left, 3: right
        self.body_part_index.append((6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
        self.body_part_index.append((0, 1, 2, 3, 4, 5))
        self.body_part_index.append((3, 4, 5, 6, 7, 8, 9, 13, 14, 15))
        self.body_part_index.append((0, 1, 2, 6, 7, 8, 9, 10, 11, 12))
    
        self.false_matrix = np.eye(num_keypoints, dtype=np.float32)
        self.false_matrix = 1 - self.false_matrix
        for x in self.body_part_index[o]:
            for i in range(num_keypoints):
                self.false_matrix[x][i] = 0
                self.false_matrix[i][x] = 0
        

    def forward(self, y):
        B, K, H, W = y.shape
        y = y.detach()
        preds, max_vals = get_max_preds(y.cpu().numpy())  # B x K x (x, y)
        preds = preds.reshape(-1, 2).astype(np.int)
        ground_truth = self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W).copy()

        ground_false = ground_truth.reshape(B, K, -1).transpose((0, 2, 1))
        ground_false = ground_false.dot(self.false_matrix).clip(max=1., min=0.).transpose((0,    2, 1)).reshape(B, K, H, W).copy()
        return torch.from_numpy(ground_truth).to(y.device), torch.from_numpy(ground_false).to(y.device)


class PseudoLabelGenerator2dRandomChoice(PseudoLabelGeneratorBase):
    def __init__(self, num_keypoints, height=64, width=64, sigma=2, o=6):
        super(PseudoLabelGenerator2dRandomChoice, self).__init__(
            num_keypoints, height, width, sigma
        )
    
        self.choice = np.sort(np.random.choice(np.array([i for i in range(num_keypoints)]), num_keypoints - o))
        self.false_matrix = np.eye(num_keypoints, dtype=np.float32)
        self.false_matrix = 1 - self.false_matrix
        for x in self.choice:
            for i in range(num_keypoints):
                self.false_matrix[x][i] = 0
                self.false_matrix[i][x] = 0


    def forward(self, y):
        B, K, H, W = y.shape
        y = y.detach()
        preds, max_vals = get_max_preds(y.cpu().numpy())  # B x K x (x, y)
        preds = preds.reshape(-1, 2).astype(np.int)
        ground_truth = self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W).copy()

        ground_false = ground_truth.reshape(B, K, -1).transpose((0, 2, 1))
        ground_false = ground_false.dot(self.false_matrix).clip(max=1., min=0.).transpose((0, 2, 1)).reshape(B, K, H, W).copy()
        return torch.from_numpy(ground_truth).to(y.device), torch.from_numpy(ground_false).to(y.device)



class PseudoLabelGenerator2dSelected(PseudoLabelGeneratorBase):
    
    # (0, 1, 2, 3, 4, 5, 13, 13, 12, 13, 6,  7,  8,  9,  10, 11) lsp
    # (7, 4, 1, 2, 5, 8, 0,  9,  12, 15, 20, 18, 13, 14, 19, 21) surreal

    '''
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Head top
    7: Head top
    8: Neck
    9: Head top
    10:Right wrist
    11:Right elbow
    12:Right shoulder
    13:Left shoulder
    14:Left elbow
    15:Left wrist

    upper = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    lower = (0, 1, 2, 3, 4, 5)
    left  = (3, 4, 5, 6, 7, 8, 9, 13, 14, 15)
    right = (0, 1, 2, 6, 7, 8, 9, 10, 11, 12)

    arms =  (0, 1, 2, 3, 4, 5) # r ankle, knee, hip; l
    legs =  (10, 11, 12, 13, 14, 15) # r wrist, elbow, shoulder; l
    heads = (6, 7, 8, 9) # head top, neck

    '''

    def __init__(self, num_keypoints, height=64, width=64, sigma=2, o=6):
        super(PseudoLabelGenerator2dSelected, self).__init__(
            num_keypoints, height, width, sigma
        )
    
        self.all =True if o // 1000 > 0 else False
        o = o // 1000
        self.idx = o // 100
        self.ratio = (o % 100) / 100
        
        self.l = [self.ratio * 6 / 100, self.ratio * 6 / 100, self.ratio * 4 / 100]
        
        self.body_part_index = []
        self.body_part_index.append((0, 1, 2, 3, 4, 5))
        self.body_part_index.append((10, 11, 12, 13, 14, 15))
        self.body_part_index.append((6, 7, 8, 9))
        
        self.choice = np.array([])
        if self.all:
            for i in range(3):
                self.choice = np.append(np.sort(np.random.choice(self.body_part_index[i])), self.l[i])
        else:
            self.choice = np.append(np.sort(np.random.choice(self.body_part_index[self.idx])), self.l[self.idx])
        self.false_matrix = 1. - np.eye(num_keypoints, dtype=np.float32)
        for x in self.choice:
            for i in range(num_keypoints):
                self.false_matrix[x][i] = 0
                self.false_matrix[i][x] = 0

    def forward(self, y):
        B, K, H, W = y.shape
        y = y.detach()
        preds, max_vals = get_max_preds(y.cpu().numpy())  # B x K x (x, y)
        preds = preds.reshape(-1, 2).astype(np.int)
        ground_truth = self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W).copy()

        ground_false = ground_truth.reshape(B, K, -1).transpose((0, 2, 1))
        ground_false = ground_false.dot(self.false_matrix).clip(max=1., min=0.).transpose((0, 2, 1)).reshape(B, K, H, W).copy()
        return torch.from_numpy(ground_truth).to(y.device), torch.from_numpy(ground_false).to(y.device)


class PseudoLabelGenerator2dDynamic(PseudoLabelGeneratorBase):
    def __init__(self, num_keypoints, height=64, width=64, sigma=2, o=0):
        super(PseudoLabelGenerator2dDynamic, self).__init__(
            num_keypoints, height, width, sigma
        )
        # self.heatmaps = torch.autograd.Variable(torch.from_numpy(self.heatmaps), requires_grad=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.k = num_keypoints
        self.E = nn.Linear(
            num_keypoints, num_keypoints, bias=False
        ).to(self.device)
        self._init_weight()

    def forward(self, y):
        B, K, H, W = y.shape
        y = y.detach()
        preds, max_vals = get_max_preds(y.cpu().numpy())  # B x K x (x, y)
        preds = preds.reshape(-1, 2).astype(np.int)
        ground_truth = torch.from_numpy(self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W)).to(y.device)

        ground_false = torch.reshape(ground_truth, (B, K, -1)).transpose(1, 2)
        ground_false = self.E(ground_false)
        ground_false = torch.reshape(ground_false.clamp(max=1., min=0.).transpose(1, 2), (B, K, H, W))
        return ground_truth, ground_false

    def _init_weight(self):
        self.E.weight = nn.parameter.Parameter(torch.from_numpy(1. - np.eye(self.k, dtype=np.float32)).to(self.device), requires_grad=True)


class RegressionDisparity(nn.Module):
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator2d, criterion: nn.Module, num_advs=1, branch_tradeoffs=[1.0]):
        super(RegressionDisparity, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator
        self.num_advs = num_advs
        self.branch_tradeoffs = branch_tradeoffs

    def forward(self, y, y_adv, weight=None, mode='min'):
        assert mode in ['min', 'max']
        ground_truth, ground_false = self.pseudo_label_generator(y.detach())
        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            loss = self.criterion(y_adv[0], ground_truth, weight) * self.branch_tradeoffs[0]
            for idx in range(1, self.num_advs):
                loss += self.criterion(y_adv[idx], ground_truth, weight) * self.branch_tradeoffs[idx]
            return loss
        else:
            loss = self.criterion(y_adv[0], ground_false, weight) * self.branch_tradeoffs[0]
            for idx in range(1, self.num_advs):
                loss += self.criterion(y_adv[idx], ground_false, weight) * self.branch_tradeoffs[idx]
            return loss


class PoseResNet2d(nn.Module):
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints, num_advs,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=1):
        super(PoseResNet2d, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = nn.ModuleList([
            self._make_head(num_head_layers, feature_dim, num_keypoints, i) \
                for i in range(num_advs)
        ])

        self.num_advs = num_advs
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints, init_index=0):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if init_index == 0:
                    nn.init.normal_(m.weight, std=0.001)
                elif init_index == 1:
                    nn.init.uniform_(m.weight, a=0, b=1)
                elif init_index == 2:
                    nn.init.xavier_normal_(m.weight, gain=1)
                else:
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = []
        for i in range(self.num_advs):
            y_adv.append(self.head_adv[i](f_adv))

        if self.training:
            return y, y_adv
        else:
            return y

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()

    