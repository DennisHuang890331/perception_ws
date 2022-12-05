from mmdet3d.apis import init_model
import torch
import time
import pickle
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
import pypcd
import numpy as np
import argparse

def timer(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def Scores_Filter(result, threshold):
    """Filter scores_3d lower than theshold
    Args:
        reslut: list(
                    dict{'boxes_3d': LiDARInstance3DBoxes(),
                         'scores_3d': tensor()
                         'labels_3d': tensor()})
        threshold: float

    Returns:
        reslut: 'boxes_3d': tensor(),
                'scores_3d': tensor()
                'labels_3d': tensor()}
    """
    boxes_3d = result[0]['boxes_3d']
    scores_3d = result[0]['scores_3d']
    labels_3d = result[0]['labels_3d']
    length = len(scores_3d)
    index = set()

    for i in range(len(boxes_3d)):
        if scores_3d[i] < threshold:
            index.add(i)

    index = list(set(range(length)) - index)
    boxes_3d = boxes_3d[index]
    scores_3d = scores_3d[index]
    labels_3d = labels_3d[index]
    
    return boxes_3d.tensor.cpu().numpy(), scores_3d, labels_3d

@timer
def Inferencemodel(pointcloud, model, prototypeFile):
    """Customized model inference data for realtime pointcloud.
        Args:
            pointcloud: torch.tensor or numpy.array()
            model: mmdet3d.apis.initmodel()
        Returns:
            reslut: list(
                dict{'boxes_3d': LiDARInstance3DBoxes(),
                    'scores_3d': tensor()
                    'labels_3d': tensor()})
    """
    # Must use this fuction before Inference your pointcloud data
    with open(prototypeFile, 'rb') as fp:
        prototype = pickle.load(fp)

    # check whether pointcloud is torch.tensor() or numpy.arrary()
    if torch.is_tensor(pointcloud):
        data = {'img_metas': prototype, 'points': [[pointcloud]]}
    else:
        pointcloud = torch.tensor(pointcloud, device=torch.device("cuda:0")).float()
        data = {'img_metas': prototype, 'points': [[pointcloud]]}
    
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result
           
def model_generater(checkpointName):
    dic = {
        'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class': {
            'config': '/home/dennis/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py',
            'checkpoint': '/home/dennis/ros2_ws/src/pointpillars/pointpillars/checkpoint/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth',
            'prototypeFile': '/home/dennis/perception_wc/src/pointpillars/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class-prototype.p',
        },
        'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car': {
            'config': '/home/dennis/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py',
            'checkpoint': '/home/dennis/ros2_ws/src/pointpillars/pointpillars/checkpoint/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth',
            'prototypeFile': '/home/dennis/perception_wc/src/pointpillars/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car-prototype.p',
        }
    }
    checkpointName = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class'
    config = dic[checkpointName]['config']
    checkpoint = dic[checkpointName]['checkpoint']
    
    model = init_model(config, checkpoint, device='cuda:0')
    return model

class subscriber(Node):

    def __init__(self, args):
        super().__init__('subscriber')
        self.proto = "/home/dennis/ros2_ws/src/pointpillars/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class-prototype.p"
        checkpointName = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class'
        self.threshhold = args.t
        self.model = model_generater(checkpointName)
        self.declare_parameter("threshold",0.5)
        self.subscription = self.create_subscription(PointCloud2,'carla/ego_vehicle/lidar',self.callback,1)
        self.subscription  # prevent unused variable warning

        self.publisher = self.create_publisher(MarkerArray, 'rviz2_object', 1)
        self.publisher  # prevent unused variable warning

    def callback(self, pointcloud2):
        arr = MarkerArray()
        table = ["pedestrian", "cyclist", "car" ]
        pc = pypcd.PointCloud.from_msg(pointcloud2)
        x = pc.pc_data['x']
        length = x.shape[0]
        y = pc.pc_data['y']
        z = pc.pc_data['z']
        intensity = np.zeros(length, dtype=np.float32)
        pcd_arr = list(zip(x, y, z, intensity))
        pcd_arr = np.array(pcd_arr)
        res = Inferencemodel(pcd_arr, self.model, self.proto)
        box_3d, scores_3d, labels_3d = Scores_Filter(res, self.threshhold)
        print("scores: ", scores_3d)
        for i in range(len(box_3d)):
            marker = Marker()
            marker.header = pointcloud2.header
            marker.ns = table[int(labels_3d[i])]
            marker.id = i + 1
            marker.type = 1
            marker.action = Marker.MODIFY
            marker.pose.position.x = np.float64(box_3d[i][0]) 
            marker.pose.position.y = np.float64(box_3d[i][1])
            marker.pose.position.z = np.float64(box_3d[i][2])
            """
            TODO fix pose.orientation
            """
            theta = np.sin((np.pi/2-box_3d[i][6])/2)
            marker.pose.orientation.x = np.float64(0)
            marker.pose.orientation.y = np.float64(0)
            marker.pose.orientation.z = np.float64(theta)
            marker.pose.orientation.w = np.sqrt(1-np.square(theta))
            
            marker.scale.x = np.float64(box_3d[i][4])
            marker.scale.y = np.float64(box_3d[i][3])
            marker.scale.z = np.float64(box_3d[i][5])
            marker.color.r = np.float64(255)
            marker.color.g = np.float64(0)
            marker.color.b = np.float64(0)
            marker.color.a = np.float64(1)
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200000000
            arr.markers.append(marker)
        self.publisher.publish(arr)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",help="threshold of pointpillars",type=float)
    parser = parser.parse_args()
    rclpy.init(args=args)
    node = subscriber(parser)
    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()