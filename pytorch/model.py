# Updated model.py with DGCNN + PointNet (OSR branch) fusion
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder

# --- KNN & Graph Feature ---
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda') if x.is_cuda else torch.device('cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

# --- DGCNN C+SH Branch ---
class DGCNNBranch(nn.Module):
    def __init__(self, args, input_dim=51):
        """
        입력: Centroid + SH, 총 51 차원 (예: 3 + 48)
        """
        super(DGCNNBranch, self).__init__()
        self.k = args.k
        self.emb_dims = args.emb_dims

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.emb_dims),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # x: (B, 51, N)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        return torch.cat((x1, x2), dim=1)

# --- PointNet OSR Branch ---
class PointNetBranch(nn.Module):
    def __init__(self, input_dim=8):
        """
        입력: Opacity + Scale + Rotation, 총 10 차원 (예: 1 + 3 + 4 + 추가 채널이 있을 수 있음)
        여기서는 PointNetEncoder를 활용하여 OSR 특징을 추출함.
        """
        super(PointNetBranch, self).__init__()
        self.encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=input_dim)

    def forward(self, x):
        # x: (B, 10, N)
        x, _, _ = self.encoder(x)
        return x

# --- Feature Fusion + Classifier ---
class DGCNN_PN_Merge(nn.Module):
    def __init__(self, args, csh_dim=51, osr_dim=8, output_channels=40):
        """
        두 branch의 출력을 융합하여 최종 분류를 수행.
        fusion_dim: DGCNNBranch의 출력 (args.emb_dims*2)와 PointNetBranch의 출력 (1024) 합산.
        """
        super(DGCNN_PN_Merge, self).__init__()
        self.dgcnn_branch = DGCNNBranch(args, input_dim=csh_dim)
        self.pointnet_branch = PointNetBranch(input_dim=osr_dim)

        fusion_dim = args.emb_dims * 2 + 1024
        self.fc1 = nn.Linear(fusion_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.fc3 = nn.Linear(256, output_channels)

    def forward(self, csh_input, osr_input):
        # csh_input: (B, 51, N) → DGCNN branch, OSR_input: (B, 10, N) → PointNet branch
        csh_feat = self.dgcnn_branch(csh_input)    # 출력: (B, args.emb_dims*2) from DGCNN branch
        osr_feat = self.pointnet_branch(osr_input)   # 출력: (B, 1024) from PointNet branch
        
        x = torch.cat((csh_feat, osr_feat), dim=1)
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.fc3(x)
        return x
