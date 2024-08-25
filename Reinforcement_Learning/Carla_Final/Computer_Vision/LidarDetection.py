"""
Class Lidar uses a PyTorch FPN_RESNET_18 model trained by us to detect Cars and return the distances

__author__ = "Bavo Lesy"

"""
import math
from socket import *

import cv2
import torch
from PIL import Image
from carla import Location
from shapely import Polygon, MultiPoint
from shapely.geometry import Point

import Computer_Vision.utils.kitti_config as cnf
from Carla_Final.CONFIG import CONFIG
from Computer_Vision.utils import fpn_resnet
from Computer_Vision.utils.evaluation_utils import *
from Computer_Vision.utils.kitti_bev_utils import makeBEVMap, get_filtered_lidar, get_corners
from Computer_Vision.utils.torch_utils import _sigmoid


class Lidar():

    def __init__(self):
        # Load in model
        self.num_classes = 2
        self.num_center_offset = 2
        self.num_direction = 2
        self.num_z = 1
        self.num_dim = 3
        # FPN resnet 18 so 18 layers
        heads = {
            'hm_cen': self.num_classes,
            'cen_offset': self.num_center_offset,
            'direction': self.num_direction,
            'z_coor': self.num_z,
            'dim': self.num_dim
        }
        self.lidar_model = fpn_resnet.get_pose_net(num_layers=18, heads=heads, head_conv=64,
                                                   imagenet_pretrained=False)
        self.lidar_model.load_state_dict(
            torch.load(CONFIG['lidar_model_folder'],
                       map_location='cpu'))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.lidar_model = self.lidar_model.to(self.device)
        self.lidar_model.eval()

        ## Sockets
        # self.init_receiver()
        # self.listen("127.0.0.1", 65432)
        # self.process_data()

    def get_distance(self, lidar_data, visualize, ego, bb_ego):
        with torch.no_grad():
            # get lidar data
            # convert lidar data to np array of shape (N, 4)
            # N is number of points
            # 4 is x, y, z, intensity
            # reshape to N, 4 but make sure N is divisible by 4 with dtype uint8
            lidar = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
            lidar = lidar.reshape(-1, 4).astype(dtype=np.float32)
            # flip y
            lidar[:, 1] *= -1
            # filter lidar
            lidar = get_filtered_lidar(lidar, cnf.boundary)
            bev_map = makeBEVMap(lidar, cnf.boundary)
            torch.from_numpy(bev_map)
            # create tensor
            bev_map = torch.from_numpy(bev_map).float().to(self.device)
            bev_map = bev_map.to(self.device, non_blocking=True).float()
            # add batch dimension
            bev_map = bev_map.unsqueeze(0)
            outputs = self.lidar_model(bev_map)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=50)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, self.num_classes, down_ratio=4, peak_thresh=0.1)
            detections = detections[0]

            # Draw prediction in the image
            bev_map = (bev_map.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections.copy(), self.num_classes)
            # Rotate the bev_map
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            bev_map = cv2.cvtColor(bev_map, cv2.COLOR_BGR2RGB)
            # convert to np array
            bev_map = np.array(bev_map)
            bev_map = Image.fromarray(bev_map)
            bev_map = bev_map.resize((480, 480))

            if visualize:
                cv2.imshow('Lidar Detection', bev_map)

            kitti_dets = convert_det_to_real_values(detections, num_classes=self.num_classes)
            distances = []
            if len(kitti_dets) > 0:

                boxes = kitti_dets[:, 0:]
                for box in boxes:
                    classification, x, y, z, w, l, h, yaw = box
                    # determine corners
                    if x > 0:
                        angle = np.arctan(y / x)
                        corners = get_corners(x, y, z, l, yaw)
                        if self.find_intersecting_box(ego, box, bb_ego, corners):# or abs(angle) < 0.15:
                            # determine angle of the car with tan(x/y)
                            closest_corner = np.argmin(np.linalg.norm(corners, axis=1))
                            # print(closest_corner)
                            angle = np.arctan(y / x)
                            # print('x: ', x)
                            # print('y: ', y)
                            # only keep distance with smallest angle
                            distances.append((np.linalg.norm(corners[closest_corner]), angle, classification))

                # only return the distance with the smallest angle
            if len(distances) > 0:
                return min(distances, key=lambda x: abs(x[0]))[0], bev_map, boxes
            else:
                return -1, bev_map, None

    def find_intersecting_box(self, ego, box, bb_ego: Polygon, corners):
        """
        Find whether the detected vehicle with Lidar lies on our GPS route
        :param ego: The ego vehicle
        :param box: The bounding box of the detected vehicle
        :param bb_ego: The bounding box 'train' of the ego vehicle along its route
        :param corners: The corners of the lidar detected bounding box
        :return: True if on route else false
        """
        classification, x, y, z, w, l, h, yaw = box
        if x > 0:
            loc_ego_t = ego.get_transform()
            x_ego = loc_ego_t.location.x
            y_ego = loc_ego_t.location.y
            z_ego = loc_ego_t.location.z
            fwd_vect = loc_ego_t.get_forward_vector()
            right_vect = loc_ego_t.get_right_vector()
            bb_loc = x * fwd_vect - y * right_vect
            box_location = Location(x_ego + bb_loc.x, bb_loc.y + y_ego, z + z_ego + 1.5)

            pointlist = []
            point1 = (self.transform(ego, corners[0, 0],corners[0, 1], z+l/2))
            point2 = (self.transform(ego, corners[1, 0], corners[1, 1], z))
            point3 = (self.transform(ego, corners[2, 0], corners[2, 1], z))
            point4 = (self.transform(ego, corners[3, 0], corners[3, 1], z+l/2))

            pointlist.append(point1)
            pointlist.append(point2)
            pointlist.append(point3)
            pointlist.append(point4)

            bb_polygon = Polygon([[p.x, p.y, p.z] for p in pointlist])
            pointlist.append(Point(box_location.x, box_location.y, box_location.z))

            if bb_ego is not None:
                for point in pointlist:
                    if bb_ego.intersects(point) or bb_ego.contains(point) or bb_ego.contains(bb_polygon) or bb_ego.intersects(bb_polygon):
                        return True
        return False

    def transform(self, ego,x ,y, z):
        """
        Transform coordinate points relative to ego car, to world coordinates
        :param ego: the ego car
        :param x: x coordinate to transform
        :param y: y coordinate to transform
        :param z: z coordinate to transform
        :return: Point object representing the point in world coordinates
        """
        loc_ego_t = ego.get_transform()
        x_ego = loc_ego_t.location.x
        y_ego = loc_ego_t.location.y
        z_ego = loc_ego_t.location.z
        fwd_vect = loc_ego_t.get_forward_vector()
        right_vect = loc_ego_t.get_right_vector()
        bb_loc = x * fwd_vect - y * right_vect
        box_location = Location(x_ego + bb_loc.x, bb_loc.y + y_ego, z + z_ego + 1.8)
        return Point(box_location.x, box_location.y, box_location.z)