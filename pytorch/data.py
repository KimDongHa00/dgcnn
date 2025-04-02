
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    pointcloud[:, :3] = np.add(np.multiply(pointcloud[:, :3], xyz1), xyz2).astype('float32')
    return pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def load_data_split(partition):
    """
    기존 62차원 h5 데이터를 불러온 후 내부에서 CSH(51차원)와 OSR(11차원)으로 분리
    - CSH = centroid + SH (3 + 48)
    - OSR = Opacity + Scale + Rotation (1 + 3 + 4) = 8 → PointNet에서는 11차원으로 맞춰서 처리 가능
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data_CSH, all_data_OSR = [], []
    all_label = []

    file_pattern = 'train*.h5' if partition == 'train' else 'test*.h5'
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', file_pattern)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')       # shape: (B, 1024, 62)
            label = f['label'][:].astype('int64')       # shape: (B, 1)

            centroid = data[:, :, :3]                   # (B, 1024, 3)
            sh_coeffs = data[:, :, 6:54]                # (B, 1024, 48)
            opacity = data[:, :, 54:55]                 # (B, 1024, 1)
            scale = data[:, :, 55:58]                   # (B, 1024, 3)
            rotation = data[:, :, 58:62]                # (B, 1024, 4)

            csh = np.concatenate((centroid, sh_coeffs), axis=-1)     # (B, 1024, 51)
            osr = np.concatenate((opacity, scale, rotation), axis=-1)  # (B, 1024, 8)

            all_data_CSH.append(csh)
            all_data_OSR.append(osr)
            all_label.append(label)

    all_data_CSH = np.concatenate(all_data_CSH, axis=0)
    all_data_OSR = np.concatenate(all_data_OSR, axis=0)
    all_label = np.concatenate(all_label, axis=0).squeeze()

    return all_data_CSH, all_data_OSR, all_label


class ModelNet40_Merged(Dataset):
    """
    DGCNN에는 C+SH (51차원), PointNet에는 O+S+R (8차원)을 각각 입력으로 주는 구조.
    - 입력은 기존의 62차원 h5 데이터에서 내부적으로 분리
    """
    def __init__(self, num_points=1024, partition='train'):
        self.data_CSH, self.data_OSR, self.labels = load_data_split(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, idx):
        csh = self.data_CSH[idx][:self.num_points]
        osr = self.data_OSR[idx][:self.num_points]
        label = self.labels[idx]

        if self.partition == 'train':
            csh = translate_pointcloud(csh)  # DGCNN에 들어갈 centroid + SH만 변환
            np.random.shuffle(csh)
            np.random.shuffle(osr)

        return csh, osr, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    train = ModelNet40_Merged(1024, partition='train')
    for csh, osr, label in train:
        print(f"C+SH: {csh.shape}, O+S+R: {osr.shape}, Label: {label}")

