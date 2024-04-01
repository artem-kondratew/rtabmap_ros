#!/usr/bin/env python3


import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from rtabmap_msgs.msg import CameraCallback


class MotionDetector(Node):
    
    def __init__(self):
        super().__init__('fake_detector')
        self.subscription_ = self.create_subscription(CameraCallback, '/rtabmap/to_motion_detector', self.callback, 10)
        self.subscription_  # prevent unused variable warning
        self.publisher_ = self.create_publisher(CameraCallback, '/rtabmap/from_motion_detector', 10)
        self.bridge_ = CvBridge()

    def callback(self, msg):
        cv_frame = self.bridge_.imgmsg_to_cv2(msg.image, desired_encoding='passthrough')
        masks = np.zeros(cv_frame.shape, dtype='uint8')
        msg.mask = self.bridge_.cv2_to_imgmsg(masks, encoding='passthrough')
        msg.mask.header = msg.image.header
        self.publisher_.publish(msg)
    

def main(args=None):
    rclpy.init(args=args)
    
    motion_detector = MotionDetector()

    rclpy.spin(motion_detector)

    motion_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
