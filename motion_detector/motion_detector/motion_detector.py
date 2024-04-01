#!/usr/bin/env python3


import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from rtabmap_msgs.msg import CameraCallback

from .submodules.yolo import Yolo
from .submodules.yolo_classes import classes_list


class MotionDetector(Node):
    
    def __init__(self):
        super().__init__('motion_detector')
        self.subscription_ = self.create_subscription(CameraCallback, '/rtabmap/to_motion_detector', self.callback, 10)
        self.subscription_  # prevent unused variable warning
        self.publisher_ = self.create_publisher(CameraCallback, '/rtabmap/from_motion_detector', 10)
        self.bridge_ = CvBridge()
        self.model_ = Yolo(classes=classes_list)
        self.model_.run(np.zeros((480, 640, 3), dtype='uint8'))
        self.get_logger().info('model initialized')

    def callback(self, msg):
        cv_frame = self.bridge_.imgmsg_to_cv2(msg.image, desired_encoding='passthrough')
        output = self.model_.run(cv_frame)
        success, masks = self.model_.merge_masks(output.masks)
        if not success:
            masks = np.zeros(cv_frame.shape, dtype='uint8')
            self.get_logger().info('no_masks')
        h, w = cv_frame.shape[:2]
        masks = masks[(w-h)//2:h+(w-h)//2, :]
        msg.mask = self.bridge_.cv2_to_imgmsg(masks, encoding='passthrough')
        msg.mask.header = msg.image.header
        self.publisher_.publish(msg)
        # cv.imshow('main_mask', masks)
        # cv.imshow('frame', cv_frame)
        # cv.waitKey(20)
    

def main(args=None):
    rclpy.init(args=args)
    
    motion_detector = MotionDetector()

    rclpy.spin(motion_detector)

    motion_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
