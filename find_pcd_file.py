import open3d as o3d
import numpy as np
import scipy

bin_array = np.fromfile("/home/tower_crane_data/crcust/dataset/crcust_top_3d_seg/v1/dataset/sequences/00/velodyne/000099.bin", dtype=np.float32).reshape(-1, 3)

mat_data = scipy.io.loadmat("/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/3d_seg_anno/VoxelLabelData/pcd_Label_100.mat")
data = mat_data['L']

# Convert the data to a NumPy array
data_np = np.array(data)

pcd_array_100 = data_np[:, :-1].astype(np.float32)

for i in [187]:
    file_name = "/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/pcd/avia_" + str(i) + ".pcd"
    pcd_data = o3d.io.read_point_cloud(file_name)
    pcd_array = np.array(pcd_data.points)
    new_pcd_array = pcd_array[np.any(pcd_array != 0, axis=1)]

    new_pcd_array_100 = pcd_array_100[np.any(pcd_array_100 != 0, axis=1)]
    
    if pcd_array_100.shape[0] == pcd_array.shape[0]:
        diff = abs(np.sum(pcd_array_100 - pcd_array))
        if diff < 1:
            print(file_name)