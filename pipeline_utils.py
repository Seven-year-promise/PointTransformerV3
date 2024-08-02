import cv2
import numpy as np
from config import Intrinsic, c2L, camera_dist, R_LIVOX2BODY
from typing import List
import math

def pixel2Camera(pixel_pt:np.array=[], distance = 1.0):
    """
    pixel_pt: u,v,1
    distance: [m]
    return: x,y,z in camera coordinate
    """
    n_pt = cv2.undistortPoints(np.array([[pixel_pt[:-1]]]), Intrinsic, camera_dist, P=Intrinsic)
    # print(n_pt)
    new_pt = np.array([n_pt[0, 0, 0], n_pt[0, 0, 1], 1]) # new u, v, 1
    n_pt = np.dot(np.linalg.inv(Intrinsic), new_pt)

    ratio = distance / np.sqrt(n_pt[0]**2 + n_pt[1]**2 + n_pt[2]**2)

    return n_pt*ratio

def pixel_and_z_to_camera_xyz(x, y, z_c) -> np.array:
    X = (x - Intrinsic[0, -1]) * z_c / Intrinsic[0, 0] # (x-c_x) * z / f_x
    Y = (y - Intrinsic[1, -1]) * z_c / Intrinsic[1, 1]
    Z = z_c
    return np.array([X, Y, Z])

def pixel2Camera(pixel_pt:np.array=[], distance = 1.0):
    """
    pixel_pt: u,v,1
    distance: [m]
    return: x,y,z in camera coordinate
    """
    n_pt = cv2.undistortPoints(np.array([[pixel_pt[:-1]]]), Intrinsic, camera_dist, P=Intrinsic)
    # print(n_pt)
    new_pt = np.array([n_pt[0, 0, 0], n_pt[0, 0, 1], 1]) # new u, v, 1
    n_pt = np.dot(np.linalg.inv(Intrinsic), new_pt)

    ratio = distance / np.sqrt(n_pt[0]**2 + n_pt[1]**2 + n_pt[2]**2)

    return n_pt*ratio

def camera2Pixel(camera_pt:np.array=[]):
    """
    camera_pt: [[x_c, x_c, x_c, x_c],
                [y_c, y_c, y_c, y_c],
                [z_c, z_c, z_c, z_c],
                [1, 1, 1, 1]]
    """
    c_pt = camera_pt[:3, :] / (camera_pt[2, :]+1e-9)
    
    return np.dot(Intrinsic, c_pt)

def lidar2Camera(lidar_pt:np.array=[]):
    return np.dot(np.linalg.inv(c2L), lidar_pt)

def camera2Lidar(camera_pt:np.array=[]):
    n = camera_pt.shape[-1]
    c_pt = np.ones((4, n), np.float32)
    c_pt[:3] = camera_pt
    return np.dot(c2L, c_pt)

def Lidar2World(lidar_pt:np.array=[]):
    return np.dot(R_LIVOX2BODY, lidar_pt)

def camera2Lidar2World(camera_pt:np.array=[]):
    n = camera_pt.shape[-1]
    c_pt = np.ones((4, n), np.float32)
    c_pt[:3] = camera_pt
    return Lidar2World(np.dot(c2L, c_pt)[:3, :])

def world2Lidar2Camera2Pixel(world_pt:np.array=[]):
    """
    """
    # n = len(world_pt)
    # world_pt = np.array(world_pt).reshape(-1, n)
    lidar_pt = np.dot(np.linalg.inv(R_LIVOX2BODY), world_pt[:3, :])

    l_pt = np.ones((4, lidar_pt.shape[-1]), np.float32)
    l_pt[:3, :] = lidar_pt

    c_pt = lidar2Camera(l_pt)

    p_pt = camera2Pixel(c_pt)

    p_pt[2, :] = c_pt[2, :]

    return p_pt

def pcd2depth(world_pt:np.array):
    """
    world_pt: [[x_w,y_w,z_w],
               [x_w,y_w,z_w],] 
    return: [[x_p,y_p,z_c],
             [x_p,y_p,z_c],]T, list
    """
    
    return world2Lidar2Camera2Pixel(world_pt)

def find_closest_points_between_array(array1, array2):
    """
    Find the closest point from array1 for all points in array2.
    
    Parameters:
    array1 (np.ndarray): Array of points to search from.
    array2 (np.ndarray): Array of points to find the closest points for.
    
    Returns:
    np.ndarray: Array of closest points from array1 for each point in array2.
    """
    closest_points = []
    for point in array2:
        distances = np.linalg.norm(array1 - point, axis=1)
        closest_point_ind = np.argmin(distances)
        closest_points.append(closest_point_ind)
    return np.array(closest_points)

def select_value_by_mask(array1, mask_im):
    """
    Find the closest point from array1 for all points in array2.
    
    Parameters:
    array1 (np.ndarray): Array of points to search from.
    mask_im cv2.im: Mask used to find the value of array1.
    
    Returns:
    np.ndarray: Array of closest points from array1 for each point in array2.
    """
    points_pixels_array = array1
    points_maskValue_array = mask_im[array1[:, 1].astype(int), array1[:, 0].astype(int)]
    ind = np.where(points_maskValue_array==255)[0]
    return array1[ind, 2], ind

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between_vectors(v1=[], v2=[]):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def find_closest_cluster_angle(vecs=[[]], vec1=[], n=2):
    angles = []
    for i in range(len(vecs)):
        angle = angle_between_vectors(vecs[i], vec1)
        angles.append(angle)
    print(angles)
    return np.argmin(angles) #, np.argpartition(angles,n-1)[:n]

def find_closest_cluster_eucli(vecs=[[]], vec1=[]):
    """
    vecs = [[1, a],
            [2, b]
            [3, c],
            [.., ..]]
    vec1 = [[1],
            [2]
            [3],
            [..]]
    """
    assert vecs.shape[0] == vec1.shape[0]
    diff_sqrt = (vecs - vec1) * (vecs - vec1) 
    # print("test math", vecs, vec1, (vecs - vec1))
    # print("test math", diff_sqrt)
    
    dis = np.sum(diff_sqrt, axis=0)
    # print("test math", dis)
    return np.argmin(dis) #, np.argpartition(angles,n-1)[:n]

def get_3d_box_from_points(pts=[]):
    x_min = np.min(pts[:, 0])
    x_max = np.max(pts[:, 0])

    y_min = np.min(pts[:, 1])
    y_max = np.max(pts[:, 1])

    z_min = np.min(pts[:, 2])
    z_max = np.max(pts[:, 2])


    return (x_min, x_max, y_min, y_max, z_min, z_max)

def find_closet_point_2d(np_coor, target_pt):
    coors = np.zeros((len(np_coor[0]), 2))
    coors[:, 0] = np_coor[1][:]
    coors[:, 1] = np_coor[0][:]

    distances = np.linalg.norm(target_pt - coors, axis=1)

    return coors[np.argmin(distances), :].astype(int) # x, y

def draw_mask(image, mask, r, c):
    """
    mask: h,w or h,w,c
    """
    masked_image = image.copy()

    if len(mask.shape) == 2:
        mask_3_channel = np.zeros_like(image)
        mask_3_channel[:, :, 0] = mask
        mask_3_channel[:, :, 1] = mask
        mask_3_channel[:, :, 2] = mask
    else:
        mask_3_channel = mask

    masked_image = np.where(mask_3_channel,
                          np.array(c, dtype='uint8'),
                          masked_image)

    masked_image = masked_image.astype(np.uint8)

    image_vis = cv2.addWeighted(image, 1-r, masked_image, r, 0)

    return image_vis

def calculate_angle(point1, point2, center):
    # Calculate the vectors of two sides of the rectangle
    vector1 = (point1[0] - center[0], point1[1] - center[1])
    vector2 = (point2[0] - center[0], point2[1] - center[1])
    
    # Calculate the magnitudes of the vectors
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    base_vec = (0, 3.5)

    if magnitude1 > magnitude2:
        # Calculate the dot product of the two vectors
        dot_product = vector1[0] * base_vec[0] + vector1[1] * base_vec[1]

        # Calculate the angle between the two sides
        angle = math.acos(dot_product / (magnitude1 * 3.5))
    else:
        # Calculate the dot product of the two vectors
        dot_product = vector2[0] * base_vec[0] + vector2[1] * base_vec[1]

        # Calculate the angle between the two sides
        angle = math.acos(dot_product / (magnitude2 * 3.5))
    angle_degree = math.degrees(angle)
    return angle_degree

def rotate_points(points, angle):
    # Convert the angle from degrees to radians
    angle_rad = np.radians(angle)

    # Create a 2D NumPy array from the points
    points_array = np.array(points)


    # Perform the rotation transformation
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_points = np.dot(points_array, rotation_matrix)


    return rotated_points

def aplly_fixed_box(box, fixed_box=np.array([[1.25, 3.5], 
                                            [1.25, -3.5],
                                            [-1.25, -3.5],
                                            [-1.25, 3.5]])):
    center = np.mean(box, axis=0)
    orientation = calculate_angle(np.mean(box[:2, :], axis=0), 
                                  np.mean(box[1:3, :], axis=0), 
                                  center)
    # Calculate the center point of the box
    
    fixed_box = rotate_points(fixed_box, angle=orientation-90)
    new_box = center + fixed_box

    return new_box

def enlarge_box(box, distance):
    # Calculate the center point of the box
    center = np.mean(box, axis=0)

    # Calculate the distance between each corner point and the center
    distances = np.linalg.norm(box - center, axis=1)

    # Add the fixed distance to the distances
    enlarged_distances = distances + distance * 1.414

    # Recalculate the new corner points based on the enlarged distances and the center
    unit_vectors = (box - center) / distances[:, np.newaxis]
    new_box = center + unit_vectors * enlarged_distances[:, np.newaxis]

    return new_box

# def find_danger_area2D_from_3D(camera_pts, safe_dis=7.0, building_plane_z = 20):
#     """
#     camera_pts: the four edge points of the object in camera coordinates
#     safe_dis: [m]
#     return: the four edge points of the dangerous area in pixel coordinates
#     """

#     # transfer all pts to world body
#     world_danger_pts = []
#     for c_p in camera_pts:
#         w_p = camera2Lidar2World(c_p)
        
#         danger_p = [w_p[0], w_p[1], building_plane_z]

#         world_danger_pts.append(danger_p)
#     new_world_safe_box = enlarge_box(world_danger_pts, safe_dis)
#     pixel_danger_pts = []

#     for n_d_p in new_world_safe_box:
#         pixel_danger_p = world2Lidar2Camera2Pixel(n_d_p)
#         pixel_danger_pts.append([pixel_danger_p[0], pixel_danger_p[1]])
        
#     return pixel_danger_pts

def find_danger_area2D_from_3D(world_pts, safe_dis=7.0, building_plane_z = 20):
    """
    camera_pts: the four edge points of the object in camera coordinates
    safe_dis: [m]
    return: the four edge points of the dangerous area in pixel coordinates
    """

    # transfer all pts to world body
    world_pts = np.array(world_pts) #[:, :, 0]
    world_pts[:, 2] = building_plane_z
    world_danger_box = world_pts
    world_danger_box[:, :2] = aplly_fixed_box(world_danger_box[:, :2])
    #world_sanger_box = world_pts.transpose() #enlarge_box(world_pts, safe_dis)
    world_danger_box[:, :2] = enlarge_box(world_danger_box[:, :2], safe_dis)
    world_danger_box = world_danger_box.transpose()
    #print(world_sanger_box)
    
    #world_sanger_box = np.array(world_sanger_box).reshape(3, -1)
    ones_row = np.ones((1, world_danger_box.shape[1]), dtype=world_danger_box.dtype)
    world_danger_box = np.vstack((world_danger_box, ones_row))
        
    return world2Lidar2Camera2Pixel(world_danger_box), world_danger_box

def is_point_in_3d_box(point, box):
    """
    Check whether a point (x, y, z) is inside a 3D box defined by 8 points (xyz).

    Parameters:
    point (tuple): The point to check (x, y, z).
    box (np.ndarray): The box defined by 8 points (8x3 array).

    Returns:
    bool: True if the point is inside the box, False otherwise.
    """
    x, y, z = point
    box = np.array(box)

    # Calculate the min and max coordinates for the box
    min_x, min_y, min_z = np.min(box, axis=0)
    max_x, max_y, max_z = np.max(box, axis=0)

    # Check if the point is within the bounds of the box
    return (min_x <= x <= max_x) and (min_y <= y <= max_y) and (min_z <= z <= max_z)

def is_point_in_2d_box_pixel(point, box, image_size):
    """
    Check whether a point (x, y) is inside a 2D box defined by 4 points using cv2.fillPoly.

    Parameters:
    point (tuple): The point to check (x, y).
    box (np.ndarray): The box defined by 4 points (4x2 array).

    Returns:
    bool: True if the point is inside the box, False otherwise.
    """
    x, y = point
    box = np.array(box, dtype=np.int32)

    # Create a mask with the same size as the bounding box
    mask = np.zeros((image_size), dtype=np.uint8)

    # Fill the polygon defined by the box points
    cv2.fillPoly(mask, [box], 1)

    # Check if the point is inside the filled polygon
    return mask[y, x] == 1




def is_point_in_2d_box(point, box):
    """
    Check whether a point (x, y, z) is inside a 3D box defined by 8 points (xyz).

    Parameters:
    point (tuple): The point to check (x, y, z).
    box (np.ndarray): The box defined by 8 points (8x3 array).

    Returns:
    bool: True if the point is inside the box, False otherwise.
    """
    x, y = point
    box = np.array(box)

    # Calculate the min and max coordinates for the box
    min_x, min_y = np.min(box, axis=0)
    max_x, max_y = np.max(box, axis=0)

    # Check if the point is within the bounds of the box
    return (min_x <= x <= max_x) and (min_y <= y <= max_y)

def generate_edge_point_mask(image: np.ndarray, points: List[List[int]]) -> np.ndarray:
    """
    Generates a mask for the edge points of an image.

    Parameters:
    image (np.ndarray): The input image.
    points (List[List[int]]): The edge points to create the mask.

    Returns:
    np.ndarray: The generated mask.
    """
    mask = np.zeros_like(image)
    roi_corners = np.array(points, dtype=np.int32).transpose()
    cv2.fillPoly(mask, [roi_corners], (255, 255, 255))
    return mask

def generate_rotated_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return box

def find_edge_points_from_mask(mask):
    """
    mask: mask area of object, h,w
    return: left_pt, top_pt, right_pt, bottom_pt
    """
    mask_coor = np.where(mask>0)
    left_ind = np.argmin(mask_coor[1])
    left_pt = [mask_coor[1][left_ind], mask_coor[0][left_ind]]
    right_ind = np.argmax(mask_coor[1])
    right_pt = [mask_coor[1][right_ind], mask_coor[0][right_ind]]

    top_ind = np.argmin(mask_coor[0])
    top_pt = [mask_coor[1][top_ind], mask_coor[0][top_ind]]
    bottom_ind = np.argmax(mask_coor[0])
    bottom_pt = [mask_coor[1][bottom_ind], mask_coor[0][bottom_ind]]

    return [left_pt, top_pt, right_pt, bottom_pt]
    
def draw_safe_danger_area(image, box, text="Safe", if_safe=True, if_arrow=True):
    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # org 
    org = (50, 50) 

    # fontScale 
    fontScale = 3

    # Blue color in BGR 
    color = (0, 0, 255) 

    # Line thickness of 2 px 
    thickness = 15
    if if_safe:
        color=(0, 255, 0)
    else:
        color=(0, 0, 255)

    # center = (int((box[0] + box[2])/2), int((box[1] + box[3])/2))
    # box = [int(x) for x in box]
    # image_vis = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 5)
    # mask_3_channel = np.zeros_like(image)
    # mask_3_channel[box[1]:box[3], box[0]:box[2], 0] = 255
    # mask_3_channel[box[1]:box[3], box[0]:box[2], 1] = 255
    # mask_3_channel[box[1]:box[3], box[0]:box[2], 2] = 255
    # image_vis = draw_mask(image_vis, mask_3_channel, r=0.2, c=color)
    # image_vis = cv2.arrowedLine(image_vis, (box[0], center[1]), (box[0], center[1]), color, 3)
    # image_vis = cv2.arrowedLine(image_vis, (box[2], center[1]), (box[2], center[1]), color, 3)
    # image_vis = cv2.putText(image_vis, '7 m', (box[0]+50, center[1]+50), font,  
    #                     2, color, 5, cv2.LINE_AA) 
    # image_vis = cv2.putText(image_vis, '7 m', (box[2]+50, center[1]+50), font,  
    #                     2, color, 5, cv2.LINE_AA) 
    # image_vis = cv2.rectangle(image_vis, (2500, 50), (5300, 300), (0, 0, 255), 5)
    title_3_channel = np.zeros_like(image)
    title_3_channel[50:300, 2500:5300, 0] = 255
    title_3_channel[50:300, 2500:5300, 1] = 255
    title_3_channel[50:300, 2500:5300, 2] = 255
    image_vis = draw_mask(image, title_3_channel, r=0.2, c=color)
    
    image_vis = cv2.putText(image_vis, text, (3000, 200), font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
    
    return image_vis

def draw_3d_box(img, threed_box, line_thickness = 10, line_color_c = (0,0,255)):
    (x_min, x_max, y_min, y_max, z_min, z_max) = threed_box
    point0 = np.array((x_min, y_min, z_min, 1), np.float32).reshape(4, 1)
    point0_camera = lidar2Camera(point0)
    point0_pixel = camera2Pixel(point0_camera).astype(int)
    point1 = np.array((x_min, y_min, z_max, 1), np.float32).reshape(4, 1)
    point1_camera = lidar2Camera(point1)
    point1_pixel = camera2Pixel(point1_camera).astype(int)

    point2 = np.array((x_min, y_max, z_min, 1), np.float32).reshape(4, 1)
    point2_camera = lidar2Camera(point2)
    point2_pixel = camera2Pixel(point2_camera).astype(int)
    point3 = np.array((x_min, y_max, z_max, 1), np.float32).reshape(4, 1)
    point3_camera = lidar2Camera(point3)
    point3_pixel = camera2Pixel(point3_camera).astype(int)

    point4 = np.array((x_max, y_min, z_min, 1), np.float32).reshape(4, 1)
    point4_camera = lidar2Camera(point4)
    point4_pixel = camera2Pixel(point4_camera).astype(int)
    point5 = np.array((x_max, y_min, z_max, 1), np.float32).reshape(4, 1)
    point5_camera = lidar2Camera(point5)
    point5_pixel = camera2Pixel(point5_camera).astype(int)

    point6 = np.array((x_max, y_max, z_min, 1), np.float32).reshape(4, 1)
    point6_camera = lidar2Camera(point6)
    point6_pixel = camera2Pixel(point6_camera).astype(int)
    point7 = np.array((x_max, y_max, z_max, 1), np.float32).reshape(4, 1)
    point7_camera = lidar2Camera(point7)
    point7_pixel = camera2Pixel(point7_camera).astype(int)

    line_thickness = 10
    line_color_c = (0,255,0)
    print(point0_pixel, point1_pixel, point2_pixel, point3_pixel, point4_pixel, point5_pixel, point6_pixel, point7_pixel)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # fontScale 
    fontScale = 5
    
    # Blue color in BGR 
    color = (0, 0, 255) 
    # Line thickness of 2 px 
    thickness = 3
    cv2.putText(img, "0", (point0_pixel[0][0], point0_pixel[1][0]), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "1", (point1_pixel[0][0], point1_pixel[1][0]), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "2", (point2_pixel[0][0], point2_pixel[1][0]), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "3", (point3_pixel[0][0], point3_pixel[1][0]), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "4", (point4_pixel[0][0], point4_pixel[1][0]), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "5", (point5_pixel[0][0], point5_pixel[1][0]), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "6", (point6_pixel[0][0], point6_pixel[1][0]), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, "7", (point7_pixel[0][0], point7_pixel[1][0]), font, fontScale, color, thickness, cv2.LINE_AA)


    img = cv2.line(img, (point0_pixel[0][0], point0_pixel[1][0]), (point4_pixel[0][0], point4_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point4_pixel[0][0], point4_pixel[1][0]), (point5_pixel[0][0], point5_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point5_pixel[0][0], point5_pixel[1][0]), (point1_pixel[0][0], point1_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point1_pixel[0][0], point1_pixel[1][0]), (point0_pixel[0][0], point0_pixel[1][0]), line_color_c, line_thickness)
    
    img = cv2.line(img, (point0_pixel[0][0], point0_pixel[1][0]), (point2_pixel[0][0], point2_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point2_pixel[0][0], point2_pixel[1][0]), (point6_pixel[0][0], point6_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point6_pixel[0][0], point6_pixel[1][0]), (point4_pixel[0][0], point4_pixel[1][0]), line_color_c, line_thickness)

    img = cv2.line(img, (point2_pixel[0][0], point2_pixel[1][0]), (point3_pixel[0][0], point3_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point3_pixel[0][0], point3_pixel[1][0]), (point7_pixel[0][0], point7_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point7_pixel[0][0], point7_pixel[1][0]), (point6_pixel[0][0], point6_pixel[1][0]), line_color_c, line_thickness)
    
    img = cv2.line(img, (point5_pixel[0][0], point5_pixel[1][0]), (point7_pixel[0][0], point7_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point7_pixel[0][0], point7_pixel[1][0]), (point3_pixel[0][0], point3_pixel[1][0]), line_color_c, line_thickness)
    img = cv2.line(img, (point3_pixel[0][0], point3_pixel[1][0]), (point1_pixel[0][0], point1_pixel[1][0]), line_color_c, line_thickness)

    return img

def downsample_point_cloud(pcd, voxel_size=0.05):
    down_pcd = pcd.voxel_down_sample(voxel_size)
    return down_pcd

if __name__ == "__main__":
    # print("test point:" , camera2Lidar(np.array([0,0,1]).reshape(3, -1)))
    # print("test point:" , Lidar2World(np.array([1,0,]).reshape(3, -1)))
    print(pcd2depth([[1, 0, 0]]))