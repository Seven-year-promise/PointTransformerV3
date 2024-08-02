import scipy
import numpy as np


mat_data = scipy.io.loadmat("/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/3d_seg_anno/VoxelLabelData_1/pcd_Label_99.mat")

# Access the data from the loaded .mat file
# For example, if the variable in the .mat file is named 'data'
data = mat_data['L']

# Convert the data to a NumPy array
data_np = np.array(data)

bin_data = data_np[:, :-1].astype(np.float32)
label_data = data_np[:, -1].astype(np.int32)
np.savetxt("/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top/txt/mat_99.txt", bin_data, delimiter=',')

# pcd_array = np.fromfile("/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/bin/avia_100.bin", dtype=np.float32).reshape(-1, 3)