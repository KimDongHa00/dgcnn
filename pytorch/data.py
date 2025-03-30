#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

'''
def load_data(partition, datamodel='xyz'):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        # 추가한 부분 full인 경우 cnetroid를 제외한 나머지 속성을 추가해서 받는 방법으로 진행
        if datamodel == 'full':
            additional_features = np.random.rand(data.shape[0], data.shape[1], 59).astype('float32')  # 62-3=59
            data = np.concatenate((data, additional_features), axis=-1)
        
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label
'''
def load_data(partition, datamodel='xyz'):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    # 파일 검색 패턴을 현재 파일 이름에 맞게 변경
    file_pattern = 'train*.h5' if partition == 'train' else 'test*.h5'
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', file_pattern)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            # 기본적으로 xyz만 사용
            xyz = data[:, :, :3]  # Centroid (x, y, z)

            # datamodel에 따라 선택적인 속성 추가
            if datamodel == 'xyz':  
                data = xyz
                
            elif datamodel == 'Op':  
                opacity = data[:, :, 54:55]  # Opacity (1개)
                data = np.concatenate((xyz, opacity), axis=-1)

            elif datamodel == 'Sh':  
                sh_coeffs = data[:, :, 6:54]  # SH Coefficients (기본 3개 + 추가 45개)
                data = np.concatenate((xyz, sh_coeffs), axis=-1)

            elif datamodel == 'SR':  
                scale = data[:, :, 55:58]  # Scale (3개)
                rotation = data[:, :, 58:62]  # Rotation (4개)
                data = np.concatenate((xyz, scale, rotation), axis=-1)

            elif datamodel == 'ECSH':  
                opacity = data[:, :, 54:55]  # Opacity (1개)
                scale = data[:, :, 55:58]  # Scale (3개)
                rotation = data[:, :, 58:62]  # Rotation (4개)
                data = np.concatenate((xyz, opacity, scale, rotation), axis=-1)

            elif datamodel == 'full':  
                additional_features = data[:, :, 6:]  # xyz 제외한 전체 속성 (59개)
                data = np.concatenate((xyz, additional_features), axis=-1)

            else:
                raise ValueError(f"Unknown datamodel: {datamodel}")
            print(f"🔍 [{partition.upper()}] 데이터 shape (datamodel={datamodel}): {data.shape}")
            	
            all_data.append(data)
            all_label.append(label)
    if not all_data:
        raise FileNotFoundError(f"No files found for pattern {file_pattern} in directory {DATA_DIR}/modelnet40_ply_hdf5_2048")
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label    

''' original code 
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud
'''
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    # XYZ 좌표(처음 3개)만 변환
    pointcloud[:, :3] = np.add(np.multiply(pointcloud[:, :3], xyz1), xyz2).astype('float32')

    return pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', datamodel='xyz'):
        self.data, self.label = load_data(partition, datamodel)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024, datamodel='xyz')
    test = ModelNet40(1024, 'test', datamodel='xyz')
    for data, label in train:
        print(data.shape)
        print(label.shape)
