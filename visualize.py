"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import open3d as o3d
import numpy as np
import torch
import struct

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(coord) if color is None else color
    )
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")

def read_npy_by_torch(file_path):
    data = np.array(np.load(file_path), np.float32)
    #tensor_data = torch.from_numpy(data)
    return data

data_path = "/home/tower_crane_data/crcust/dataset/crcust_top_3d_seg/v1/dataset/sequences/00/velodyne/000011.bin"
points = []

# Open the binary file
with open(data_path, "rb") as file:
    while True:
        # Read 4 floats (16 bytes) at a time
        bytes = file.read(12)
        if not bytes:
            break
        # Unpack the bytes to floats
        x, y, z = struct.unpack('fff', bytes)
        points.append([x, y, z])

points = np.array(points)

NUM = "000032"

np.savetxt("/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top/txt/" + NUM + ".txt", points, delimiter=',')

pre_path = "/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top/result/00_" + NUM + "_pred.npy"
label_path = "/home/tower_crane_data/crcust/dataset/crcust_top_3d_seg/v1/dataset/sequences/00/labels/" + NUM + ".label"
#save_path = "/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base/pcd/11_000000_pred.ply"
prediction = read_npy_by_torch(pre_path)


# pre_points = points.copy()
# print(len(prediction), len(points))
# pre_points[np.where(prediction!=0)] = [0, 0, 0]
# np.savetxt("/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top/txt" + NUM + "_pred_0.txt", pre_points, delimiter=',')


labels = np.fromfile(label_path, dtype=np.int32)
print(len(prediction), len(points), len(labels))
gt_points = points.copy()
gt_points[np.where(labels!=2)] = [0, 0, 0]
np.savetxt("/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top/txt/" + NUM + "_label_2.txt", gt_points, delimiter=',')
#save_point_cloud(points, color=prediction, file_path=save_path, logger=None)