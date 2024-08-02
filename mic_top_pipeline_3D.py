import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from config import img_size, img_size, model_path, \
                    video_path, lidar_path, \
                      depth_path, save_path
                        
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from tower_utils import pixel2Camera, camera2Lidar2World
import cv2
import random
import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans, MeanShift
import sys
import open3d as o3d
import heapq
from pathlib import Path
from pipeline_utils import * #find_closet_point_2d, draw_mask, \
    # draw_safe_danger_area, find_edge_points_from_mask, find_danger_area2D_from_3D, \
    # generate_edge_point_mask, generate_rotated_bbox, pcd2depth, find_closest_points_between_array, \
    # Lidar2World, pixel_and_z_to_camera_xyz
from inference import Seg3D
# sys.path.append(str(Path(__file__).parent.parent/"data_process"))
# sys.path.append(str(Path(__file__).parent))

# from small_human_detection_im_crop import crop_im

class App:
    def __init__(self, video_path="", 
                 depth_path="", 
                 pcd_path="", 
                 mic_model_path="", 
                 person_model_path="", 
                 seg_model_path="",
                 save_path="", 
                 separate=False,
                 img_size=[]) -> None:
        self.video_path = video_path
        self.depth_path = depth_path
        self.pcd_path = pcd_path
        self.mic_model_path = mic_model_path
        self.person_model_path = person_model_path
        self.separate = separate
        self.seg_model_path = seg_model_path
        self.save_path = save_path
        self.save_path.mkdir(exist_ok=True, parents=True)

        self.image_size = img_size

        self.model_im_size = 640

        self.x_ratio, self.y_ratio = self.image_size[0] / self.model_im_size, self.image_size[1] / self.model_im_size

        self.safe_dis = 7 # m
        self.init_model()
        self.read_video()

    def init_model(self):
        self.mic_model = torch.hub.load('yolov5', 'custom', path=self.mic_model_path, source='local')
        self.mic_model.iou = 0.45
        self.mic_model.conf = 0.05
        self.human_thre = 0.65

        if self.separate:
            self.person_model = torch.hub.load('yolov5', 'custom', path=self.person_model_path, source='local')
            self.person_model.iou = 0.2
            self.person_model.conf = 0.2
        else:
            self.person_model = None

        self.kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
        self.mean_shift = MeanShift(bandwidth=5.0)

        output_mode = "binary_mask"

        #sam = sam_model_registry["vit_h"](checkpoint="sam_checkpoint/sam_vit_h_4b8939.pth")
        sam = sam_model_registry["vit_b"](checkpoint="/home/weights/sam_checkpoint/sam_vit_b_01ec64.pth")
        #sam = sam_model_registry["vit_l"](checkpoint="sam_checkpoint/sam_vit_l_0b3195.pth")

        #sam.mask_threshold =0.01
        self.seg_predictor = SamPredictor(sam)

        self.segmentor_3d = Seg3D(config_file = "/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3/config.py",
                      save_path = "exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3")

    def read_video(self):
        self.image_video = sorted(self.video_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
        self.depth_video = sorted(self.depth_path.rglob("*.png"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))
        self.pcd_video = sorted(self.pcd_path.rglob("*.pcd"), key=lambda a: int(str(a).split("_")[-1].split(".")[0]))

    def rotate_pt_with_imc(self, pt):
        x, y = pt
        x -= self.image_size[0] / 2
        y -= self.image_size[1] / 2

        return (self.image_size[0]/2 -x, self.image_size[1] / 2 - y)

    def depth_cluster(self, pts, method="KMeans"):
        # hist_name = str(random.randint(0, 22345))
        # plt.hist(pts, bins=100, color='skyblue', edgecolor='black')
 
        # # Adding labels and title
        # plt.xlabel('Values')
        # plt.ylabel('Frequency')
        # plt.title('Basic Histogram')
        
        # # Display the plot
        # plt.savefig("results/"+hist_name+"hist_z.png")
        if method == "KMeans":
            clustering_res = self.kmeans.fit(pts)
        elif method == "MeanShift":
            #print(pts.shape)
            clustering_res = self.mean_shift.fit(pts)

        #print("centers for z: ", hist_name, clustering_res.cluster_centers_)
        cluster_centers = clustering_res.cluster_centers_
        cluster_labels = clustering_res.labels_

        cluster_centers = cluster_centers[np.where(cluster_centers>0)]
        cluster_labels = cluster_labels[np.where(cluster_centers>0)]
        ind = np.argmin(cluster_centers)
        min_c = np.min(cluster_centers)
        # if clustering_res.cluster_centers_[0, 0] > 0:
        #     ind = 0
        # elif clustering_res.cluster_centers_[1, 0] > 0: 
        #     ind = 1
        # elif clustering_res.cluster_centers_[1, 0] > 0 and clustering_res.cluster_centers_[0, 0] > 0: 
        #     ind = np.argmin(np.sum(np.abs(clustering_res.cluster_centers_), axis=1))
        # else:
        #     raise NotImplementedError

        #print(kmeans_res.cluster_centers_)

        selected_data_ind = np.where(clustering_res.labels_==cluster_labels[ind])

        return min_c, selected_data_ind # np.median(pts[selected_data_ind])
        
    def seg(self, image, input_box):
        self.seg_predictor.set_image(image)
        input_label = np.array([1])

        masks, scores, logits = self.seg_predictor.predict(
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )

        masks = np.array(masks*255, np.uint8)

        return masks
    
    def danger_detect(self, mic_3d_box, dep, danger_dist=5):
        """
        mic_pt: x_min, x_max, y_min, y_max in pixel coordinate, z_min, z_max in camera coordinate 
        dep: depth image 16 channels with mic area as zero
        danger_dist: the distance between mic and the objects regarded as danger [m]
        return: list of the bboxes of dangers, [[x_min, y_min, x_max, y_max]]
        """
        x_min, x_max, y_min, y_max, z_min, z_max = mic_3d_box
        dep_danger = dep[np.where(dep>(z_min-danger_dist))]
        danger_coordinates = np.where(dep_danger<(z_max+danger_dist))
        num = danger_coordinates[0].shape
        selected_index = np.zeros((num, 1)) * False
        for i in range(num):
            x = danger_coordinates[1][i]
            y = danger_coordinates[0][i]
            if x > (x_min - danger_dist) and x < (x_max + danger_dist) and y < (y_max + danger_dist)and y > (y_min - danger_dist):
                selected_index[i] = True

        slected_coordinates = np.array([danger_coordinates[0][selected_index], danger_coordinates[1][selected_index]]).reshape(-1, 2)

        mean_shift_res = self.mean_shift.fit(slected_coordinates)

        num_cluster = mean_shift_res.cluster_centers_.shape[0]
        danger_bbox_list = []
        for j in range(num_cluster):
            danger_piece_coor = slected_coordinates[mean_shift_res.labels_==j]
            danger_piece_y_min, danger_piece_x_min = np.min(danger_piece_coor, axis = 0)
            danger_piece_y_max, danger_piece_x_max = np.max(danger_piece_coor, axis = 0)
            danger_bbox_list.append([danger_piece_x_min, danger_piece_y_min, danger_piece_x_max, danger_piece_y_max])

        return danger_bbox_list
    

    def find_z_from_mask_by_dep(self, center, mask, dep): 
        """
        This is used for the mic distance
        center this the object: x, y
        mask: mask the object, the area of the mic, (h, w), no channels
        dep: depth image
        return: 3d point, x,y,z in camera coordinate 
        """   
        depth_area = np.zeros_like(dep)
        mask_coor = np.where(mask>0)
        depth_area[mask_coor] = dep[mask_coor]

        depth_info = depth_area[np.where(depth_area>0)]
        #print(len(depth_info))
        for d_i in depth_info:
            print(d_i)

        Z, mic_coor = self.depth_cluster(np.array(depth_info, np.float32).reshape(-1, 1)) # h,w, i.e., y, x

        camera_pt = pixel2Camera(np.array([center[0], center[1], 1]), -1 * Z)

        return camera_pt, Z

    def find_z_from_mask_by_pcd_pixel(self, center, mask, pcd_pixel, pcd_world): 
        """
        This is used for the mic distance
        center this the object: x, y
        mask: mask the object, the area of the mic, (h, w), no channels
        dep: depth image
        return: 3d point, x,y,z in camera coordinate 
        """   
        # pcd_pixel_xy = pcd_pixel[:, :2]
        # pcd_world_xyz = pcd_world[:, :3]

        mask_height, mask_width = mask.shape
        pcd_pixel[:, 0] = np.clip(pcd_pixel[:, 0], 0, mask_width - 1)
        pcd_pixel[:, 1] = np.clip(pcd_pixel[:, 1], 0, mask_height - 1)

        
        
        #mask_resize = cv2.resize(mask, f_x=0.25, f_y=0.25, 0, 0, )
        # new_mask = np.zeros_like(mask)
        # new_mask[pcd_pixel_xy[:, 1].astype(int), pcd_pixel_xy[:, 0].astype(int)] = 255
        # new_masked_mask = cv2.bitwise_and(mask, mask, mask=new_mask)
        # #mask = mask[pcd_pixel_xy[:, 1].astype(int), pcd_pixel_xy[:, 0].astype(int)]
        # #cv2.imwrite("results/mic_mask.png", new_masked_mask)
        # new_masked_mask = np.array(new_masked_mask, np.float32)
        # mask_coor_yx = np.where(new_masked_mask>0)
        
        # mask_coor_xy = np.zeros((mask_coor_yx[0].shape[0], 2))
        # mask_coor_xy[:, 0] = mask_coor_yx[1][:] #.reshape(-1, 1)
        # mask_coor_xy[:, 1] = mask_coor_yx[0][:] #.reshape(-1, 1)
        
        # cloest_points_ind = find_closest_points_between_array(pcd_pixel_xy, mask_coor_xy)
        # print("number of close points: ", pcd_pixel.shape)
        # print(pcd_pixel[5841, :])
        # print(mask_coor_xy, cloest_points_ind)

        #print(pcd_pixel[cloest_points_ind, 2])

        Zc_array, Zc_ind = select_value_by_mask(pcd_pixel, mask)

        np.savetxt('results/test_selected.txt', pcd_world[Zc_ind], delimiter=',')
        np.savetxt('results/test_all.txt', pcd_world, delimiter=',')
        
        camera_Z, mic_coor = self.depth_cluster(Zc_array.reshape(-1, 1), method="KMeans")

        camera_xyz = pixel_and_z_to_camera_xyz(center[0], center[1], camera_Z)
        world_XYZ = camera2Lidar2World(camera_xyz.reshape(3, -1))
        
        #print("5 minimum values", pcd_pixel[min_indices_list, 2], world_XYZ)
        
        return world_XYZ.transpose().squeeze(axis=0), camera_Z

    def find_z_from_point_by_dep(self, center, dep): 
        """
        This is used for the human distance
        center this the object: x, y
        dep: depth image
        return: 3d point, x,y,z in camera coordinate 
        """   
        depth_coor = np.where(dep>0)

        cloest_pt = find_closet_point_2d(depth_coor, center)
        Z = dep[cloest_pt[1], cloest_pt[0]]
        camera_pt = pixel2Camera(np.array([center[0], center[1], 1]), -1 * Z)

        return camera_pt, Z

    def find_z_from_point_by_pcd_pixel(self, center, pcd_pixel, pcd_world): 
        """
        This is used for the human distance
        center this the object: x, y
        dep: depth image
        return: 3d point, x,y,z in world coordinate 
        """   
        pcd_pixel_xy = pcd_pixel[:, :2]
        pcd_world_xyz = pcd_world[:, :3]

        distances = np.linalg.norm(pcd_pixel_xy - center, axis=1)

        min_values = heapq.nsmallest(10, enumerate(distances), key=lambda x: x[1])

        # Extract the minimum values and indices
        min_indices_list = [index for index, value in min_values]
        #world_XYZ = pcd_world_xyz[cloest_point_ind[0]] 
        camera_Z = np.median(pcd_pixel[min_indices_list, 2])

        camera_xyz = pixel_and_z_to_camera_xyz(center[0], center[1], camera_Z)
        world_XYZ = camera2Lidar2World(camera_xyz.reshape(3, -1))
        #print("5 minimum values", pcd_pixel[min_indices_list, 2], world_XYZ)

        return world_XYZ.transpose().squeeze(axis=0)
    
    def detect_mic_people_3d(self, img_rotated, pcd):
        # img_input = cv2.resize(img, (self.model_im_size, self.model_im_size), cv2.INTER_CUBIC)
        img_input = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)
        detect_results = self.mic_model(img_input, size=self.model_im_size)

        pcd_array = np.array(pcd)
            
        pcd_array = pcd_array[np.any(pcd_array != 0, axis=1)]

        segments_3d = self.segmentor_3d.predict(pcd_array)
        

        pred = results.pred[0].cpu().numpy()

        if pred.shape[0] > 0:
            

            pcd_world = Lidar2World(pcd_array.transpose()) # 3 x n
            pcd_pixel = pcd2depth(pcd_world).transpose()
            pcd_world = pcd_world.transpose()
            #pcd_pixel = pcd_pixel[:, ~np.isnan(pcd_pixel).any(axis=0)]
            # print(pcd_pixel)
            
            #pcd_world = pcd_world[:, ~np.isnan(pcd_world).any(axis=0)]
            # pcd_world = pcd_world[~np.isnan(pcd_world)]
            # print(pcd_world)
            bboxes = {}
            bboxes["human"] = []
            bboxes["mic"] = []
            bboxes["mic_frame"] = []

            bboxes_rotated = {}
            bboxes_rotated["human"] = []
            bboxes_rotated["mic"] = []
            bboxes_rotated["mic_frame"] = []

            for p in pred: #x,y,x,y, prob, cls
                if p[-1]==0:
                    if p[-2] > self.human_thre:
                        bboxes_rotated["human"].append(p[:-2].tolist())
                        x_max, y_max = self.rotate_pt_with_imc((p[0], p[1])) # x_min, y_min --> x_max, y_max
                        x_min, y_min = self.rotate_pt_with_imc((p[2], p[3])) # x_max, y_max --> x_min, y_min
                        bboxes["human"].append([x_min, y_min, x_max, y_max])
                elif p[-1] == 1:
                    bboxes_rotated["mic"].append(p[:-2].tolist())
                    x_max, y_max = self.rotate_pt_with_imc((p[0], p[1])) # x_min, y_min --> x_max, y_max
                    x_min, y_min = self.rotate_pt_with_imc((p[2], p[3])) # x_max, y_max --> x_min, y_min
                    bboxes["mic"].append([x_min, y_min, x_max, y_max])
                elif p[-1] == 2:
                    bboxes_rotated["mic_frame"].append(p[:-2].tolist())
                    x_max, y_max = self.rotate_pt_with_imc((p[0], p[1])) # x_min, y_min --> x_max, y_max
                    x_min, y_min = self.rotate_pt_with_imc((p[2], p[3])) # x_max, y_max --> x_min, y_min
                    bboxes["mic_frame"].append([x_min, y_min, x_max, y_max])
                else:
                    raise NotImplementedError

            if len(bboxes["mic"]) > 0:
                mic_pixel_pt = np.array([(bboxes["mic"][0][0]+bboxes["mic"][0][2]) / 2, 
                                         (bboxes["mic"][0][1]+bboxes["mic"][0][3]) / 2])
                
                mic_mask_rotated = self.seg(img_rotated, np.array(bboxes_rotated["mic"][0]))
                mic_mask = mic_mask_rotated[0, :, :] 
                mic_mask = cv2.rotate(mic_mask, cv2.ROTATE_180)

                # mic_camera_pt, mic_Z_center = self.find_z_from_mask_by_dep(center=mic_pixel_pt, mask=mic_mask, dep=dep)
                # mic_camera_pt, mic_Z_center = self.find_z_from_point_by_dep(mic_pixel_pt, dep)
                mic_world_pt, mic_Z_center = self.find_z_from_mask_by_pcd_pixel(mic_pixel_pt, mic_mask, pcd_pixel, pcd_world)
                print("mic: ", mic_world_pt, mic_Z_center)
                #mic_edge_points = find_edge_points_from_mask(mic_mask)
                mic_edge_pixel_pts = generate_rotated_bbox(mic_mask)
                #print(mic_edge_pixel_pts)
                mic_edge_world_pts = []
                for m_e_p in mic_edge_pixel_pts:
                    mic_edge_c_pt = pixel_and_z_to_camera_xyz(float(m_e_p[0]), float(m_e_p[1]), mic_Z_center)
                    
                    mic_e_world_pt = camera2Lidar2World(mic_edge_c_pt.reshape(3, -1)).transpose().squeeze(axis=0)
                    # mic_e_world_pt = self.find_z_from_point_by_pcd_pixel(m_e_p, pcd_pixel, pcd_world)
                    mic_edge_world_pts.append(mic_e_world_pt)
                    #print(mic_e_world_pt)
            else:
                mic_world_pt = None
                mic_mask = None
                mic_edge_world_pts= None
            
            if len(bboxes["human"]) > 0:
                human_camera_pts = []
                danger_flags = []
                human_pixel_pts = []
                human_world_pt_zs = []
                human_world_pts = []
                for p_b in bboxes["human"]:
                    #p_b = yolo_box2xyxy(p_b, w, h)
                    
                    p_center = ((p_b[0]+p_b[2])/2.0, (p_b[1]+p_b[3])/2.0)
                    human_pixel_pts.append(p_center)

                    # p_camera_pt, Z = self.find_z_from_point_by_dep(p_center, dep)
                    # h_world_pt = camera2Lidar2World(p_camera_pt)
                    h_world_pt = self.find_z_from_point_by_pcd_pixel(p_center, pcd_pixel, pcd_world)
                    human_world_pts.append(h_world_pt)
                    human_world_pt_zs.append(h_world_pt[2])
                    #print(h_world_pt)
                    # human_camera_pts.append(p_camera_pt)

                    # if mic_world_pt is not None:
                    #     if np.sqrt((h_world_pt[0] - mic_world_pt[0])**2 + \
                    #                (h_world_pt[1] - mic_world_pt[1])**2 + \
                    #                (h_world_pt[2] - mic_world_pt[2])**2)  < self.safe_dis:
                    #         danger_flags.append("danger")
                    #     else:
                    #         danger_flags.append("safe")
                    # else:
                    #     danger_flags.append("safe")
            else:
                human_world_pts = None
            if len(bboxes["mic"]) > 0:
                if len(bboxes["human"]) > 0:
                    building_plane_z = np.median(human_world_pt_zs)
                else:
                    building_plane_z = -21
                mic_edge_pixel_pts, mic_edge_world_pts = find_danger_area2D_from_3D(mic_edge_world_pts, 
                                                                                    safe_dis=3.0, 
                                                                                    building_plane_z=building_plane_z) #building_plane_z)
                print(mic_edge_world_pts)
                if len(bboxes["human"]) > 0:
                    for h_p_p in human_pixel_pts:
                        is_danger = is_point_in_2d_box_pixel(np.array(h_p_p[:2]).astype(int), 
                                                             mic_edge_pixel_pts.transpose()[:, :2].astype(int),
                                                             img_input.shape[:2])
                        if is_danger:
                            danger_flags.append("danger")
                        else:
                            danger_flags.append("safe")

                        #print(h_w_p[:2].astype(int), mic_edge_world_pts.transpose()[:, :2].astype(int), is_danger)
            else:
                mic_edge_pixel_pts = None
                danger_flags= None
            
            
            # dep_mask = dep
            # dep_mask[mic_coor] = 0

            # danger_list = self.danger_detect(mic_3d_box=[right_bottom[0], right_bottom[1], left_top[0], left_top[1]], dep=dep_mask, danger_dist=10)

            return mic_world_pt, mic_mask, human_world_pts, danger_flags, bboxes, mic_edge_pixel_pts # (right_bottom[0], right_bottom[1], left_top[0], left_top[1]), danger_list
        else:
            return None, None, None, None, None, None

    def run(self, Separate=False):
        out = cv2.VideoWriter(str(self.save_path / "video_comp.avi"), cv2.VideoWriter_fourcc(*'MPEG'), 2, self.image_size, True)

        for n, (i_p, pcd_p) in enumerate(zip(self.image_video[0:], self.pcd_video[0:])):
            print(i_p)
            # print(n)
            img = cv2.imread(str(i_p))
            #dep = cv2.imread(str(d_p), cv2.IMREAD_UNCHANGED)
            # print(dep.shape)
            # dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)
            img_input = cv2.rotate(img, cv2.ROTATE_180)
            pcd_data = o3d.io.read_point_cloud(str(pcd_p))
            pcd_list = np.array(pcd_data.points).tolist()

            if self.separate:
                mic_camera_pt, mic_mask, human_camera_pt, danger_flags, bboxes = self.mic_2D_detect_dep(img_input, pcd_list)
                # people_bboxes = self.person_2D_detect_dep(img_input, dep)
                # TODO: detect mic and human separately
            else:
                mic_camera_pt, mic_mask, human_camera_pt, danger_flags, bboxes, mic_edge_pixel_pts = self.detect_mic_people_3d(img_input, pcd_list)

            img_vis = img.copy()
            #print(bboxes)
            # if len(bboxes["mic_frame"]) > 0:
            #     mic_frame_box = bboxes["mic_frame"][0]
            #     img_vis = cv2.rectangle(img_vis, 
            #                     pt1=(int(mic_frame_box[0]), int(mic_frame_box[1])), 
            #                     pt2=(int(mic_frame_box[2]), int(mic_frame_box[3])), 
            #                     color=(255, 0, 255), 
            #                     thickness=5)

            if mic_camera_pt is not None:
                mic_box = bboxes["mic"][0]
                danger_area_mask = np.zeros_like(img)
                danger_box = [int(mic_box[0]-200), int(mic_box[1]-200), int(mic_box[2]+200), int(mic_box[3]+200)]
                danger_area_mask[danger_box[1]:danger_box[3], danger_box[0]:danger_box[2], :] = 255
                if danger_flags is not None and "danger" in danger_flags:
                    text = 'Warning! Someone is in the dangerous area!'
                    if_safe=False
                    # img_vis = draw_safe_danger_area(img_vis, 
                    #                                     mic_box, 
                    #                                     text='Warning! Someone is in the dangerous area!',
                    #                                     if_safe=False,
                    #                                     if_arrow=True)
                    mic_color = (0,0,255)
                else:
                    text='                 Safe'
                    if_safe=True
                    # img_vis = draw_safe_danger_area(img_vis, 
                    #                                     mic_box, 
                    #                                     text='                 Safe',
                    #                                     if_safe=True,
                    #                                     if_arrow=True)
                    mic_color = (0,255,0)
                    
                img_vis = draw_mask(img_vis, mic_mask, r=0.6, c=mic_color)
                # img_vis = draw_mask(img_vis, danger_area_mask, r=0.2, c=mic_color)
                # img_vis = cv2.rectangle(img_vis, 
                #                     pt1=(int(danger_box[0]), int(danger_box[1])), 
                #                     pt2=(int(danger_box[2]), int(danger_box[3])), 
                #                     color=mic_color, 
                #                     thickness=2)
                img_vis = cv2.rectangle(img_vis, 
                                    pt1=(int(mic_edge_pixel_pts.transpose()[0, 0]), int(mic_edge_pixel_pts.transpose()[0, 1])), 
                                    pt2=(int(mic_edge_pixel_pts.transpose()[0, 0]+10), int(mic_edge_pixel_pts.transpose()[0, 1]+10)), 
                                    color=mic_color, 
                                    thickness=2)
                img_vis = cv2.rectangle(img_vis, 
                                    pt1=(int(mic_edge_pixel_pts.transpose()[1, 0]), int(mic_edge_pixel_pts.transpose()[1, 1])), 
                                    pt2=(int(mic_edge_pixel_pts.transpose()[1, 0]+10), int(mic_edge_pixel_pts.transpose()[1, 1]+10)), 
                                    color=mic_color, 
                                    thickness=2)
                img_vis = cv2.rectangle(img_vis, 
                                    pt1=(int(mic_edge_pixel_pts.transpose()[2, 0]), int(mic_edge_pixel_pts.transpose()[2, 1])), 
                                    pt2=(int(mic_edge_pixel_pts.transpose()[2, 0]+10), int(mic_edge_pixel_pts.transpose()[2, 1]+10)), 
                                    color=mic_color, 
                                    thickness=2)

                img_vis = cv2.rectangle(img_vis, 
                                    pt1=(int(mic_edge_pixel_pts.transpose()[3, 0]), int(mic_edge_pixel_pts.transpose()[3, 1])), 
                                    pt2=(int(mic_edge_pixel_pts.transpose()[3, 0]+10), int(mic_edge_pixel_pts.transpose()[3, 1]+10)), 
                                    color=mic_color, 
                                    thickness=2)
                cv2.imwrite(str(self.save_path/("test.png")), img_vis)
                print(mic_edge_pixel_pts)
                edge_point_mask = generate_edge_point_mask(img_vis, mic_edge_pixel_pts[:2, :])
                img_vis = draw_mask(img_vis, edge_point_mask, r=0.2, c=mic_color)
            else:
                continue
            if human_camera_pt is not None:
                for _, (h_c_p, d_f, h_b) in enumerate(zip(human_camera_pt, danger_flags, bboxes["human"])):
                    if d_f == "danger":
                        h_color = (0, 0, 255)
                    else:
                        h_color = (0, 255, 0)
                
                    img_vis = cv2.rectangle(img_vis, 
                                pt1=(int(h_b[0]), int(h_b[1])), 
                                pt2=(int(h_b[2]), int(h_b[3])), 
                                color=h_color, 
                                thickness=5)
                
                # img_detected = cv2.putText(img_detected, str(mic_pt), (int(mic_box_2d[0]), int(mic_box_2d[1])), font,  
                #     fontScale, color, thickness, cv2.LINE_AA) 
                # if len(danger_list) >0:
                #     for d_bb in danger_list:
                #         img_detected = cv2.rectangle(img_detected, 
                #                     pt1=(int(d_bb[0]), int(d_bb[1])), 
                #                     pt2=(int(d_bb[2]), int(d_bb[3])), 
                #                     color=(0, 0, 255), 
                #                     thickness=5)
            # if people_bboxes is not None:
            #     for p_b in people_bboxes:
            #         img_detected = cv2.rectangle(img, 
            #                             pt1=(int(p_b[0]), int(p_b[1])), 
            #                             pt2=(int(p_b[2]), int(p_b[3])), 
            #                             color=(0, 255, 0), 
            #                             thickness=5)
            img_vis = cv2.rotate(img_vis, cv2.ROTATE_180)
            img_vis = draw_safe_danger_area(img_vis, 
                                            mic_box, 
                                            text=text,
                                            if_safe=if_safe,
                                            if_arrow=True)
            cv2.imwrite(str(self.save_path/(str(n) + ".png")), img_vis)
            out.write(img_vis)
            # break

        out.release()
        cv2.destroyAllWindows()
            

if __name__ == "__main__":
    app = App(video_path=Path('/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/pic/'), 
              depth_path=Path('/home/tower_crane_data/crcust/crcust_top/mvs_avia/2024-06-28-11-26-26_ruian_m/dep/'), 
              pcd_path=Path('/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/pcd/'), 
              mic_model_path="/home/HKCRC_perception/tower_crane_2Ddetection/2D/runs/train/crcust_top_v4/weights/last.pt",
              seg_model_path="/home/weights/sam_checkpoint/sam_vit_b_01ec64.pth", 
              person_model_path="", 
              save_path=save_path,
              separate=False,
              img_size=img_size)
    app.run()
