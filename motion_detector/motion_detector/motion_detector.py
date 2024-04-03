#!/usr/bin/env python3


import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from rtabmap_msgs.msg import MotionDetectorData

from .submodules.yolo import Yolo
from .submodules.yolo_classes import classes_list


class MotionDetector(Node):
    
    def __init__(self):
        super().__init__('motion_detector')
        self.subscription_ = self.create_subscription(MotionDetectorData, '/rtabmap/to_motion_detector', self.callback, 10)
        self.subscription_  # prevent unused variable warning
        self.camera_info_pub_ = self.create_publisher(CameraInfo, '/rtabmap/motion_detector/camera_info', 10)
        self.rgb_pub_ = self.create_publisher(Image, '/rtabmap/motion_detector/rgb', 10)
        self.depth_pub_ = self.create_publisher(Image, '/rtabmap/motion_detector/depth', 10)
        self.mask_pub_ = self.create_publisher(Image, '/rtabmap/motion_detector/mask', 10)
        self.odom_pub_ = self.create_publisher(Odometry, '/rtabmap/motion_detector/odom', 10)
        self.bridge_ = CvBridge()
        self.model_ = Yolo(classes=classes_list)
        self.model_.run(np.zeros((480, 640, 3), dtype='uint8'))
        self.get_logger().info('model initialized')

    def callback(self, msg):
        cv_frame = self.bridge_.imgmsg_to_cv2(msg.rgb, desired_encoding='passthrough')
        output = self.model_.run(cv_frame)
        success, masks = self.model_.merge_masks(output.masks)
        if not success:
            masks = np.zeros(cv_frame.shape, dtype='uint8')
            self.get_logger().info('no_masks')
        else:
            h, w = cv_frame.shape[:2]
            masks = masks[(w-h)//2:h+(w-h)//2, :]
        msg.mask = self.bridge_.cv2_to_imgmsg(masks, encoding='passthrough')
        msg.mask.header = msg.rgb.header
        self.camera_info_pub_.publish(msg.camera_info)
        self.rgb_pub_.publish(msg.rgb)
        self.depth_pub_.publish(msg.depth)
        self.mask_pub_.publish(msg.mask)
        self.odom_pub_.publish(msg.odom)
        cv.imshow('main_mask', masks)
        cv.imshow('frame', cv_frame)
        cv.waitKey(20)


def main(args=None):
    rclpy.init(args=args)
    
    motion_detector = MotionDetector()

    rclpy.spin(motion_detector)

    motion_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
