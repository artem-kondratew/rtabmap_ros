#!/usr/bin/env python3


import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from rtabmap_msgs.msg import VisualizerData, KeyPoint


class Visualizer(Node):
    
    def __init__(self):
        super().__init__('rtabmap_visualizer')
        self.subscription_ = self.create_subscription(VisualizerData, '/rtabmap/to_visualizer', self.callback, 10)
        self.subscription_  # prevent unused variable warning
        self.publisher_ = self.create_publisher(Image, '/rtabmap/visualizer', 10)
        self.bridge_ = CvBridge()

    def callback(self, msg):
        image = self.bridge_.imgmsg_to_cv2(msg.image, desired_encoding='passthrough')
        mask = self.bridge_.imgmsg_to_cv2(msg.mask, desired_encoding='passthrough')
        static_keypoints = msg.static_keypoints
        dynamic_keypoints = msg.dynamic_keypoints

        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        for kp in static_keypoints:
            cv.circle(image, (int(kp.pt.x), int(kp.pt.y)), 2, (0, 255, 0), 1)
        for kp in dynamic_keypoints:
            cv.circle(image, (int(kp.pt.x), int(kp.pt.y)), 2, (0, 0, 255), 1)
        
        msg.image = self.bridge_.cv2_to_imgmsg(image, encoding='passthrough')
        self.publisher_.publish(msg.image)

        cv.imshow('result', image)
        cv.waitKey(20)


def main(args=None):
    rclpy.init(args=args)
    
    visualizer = Visualizer()

    rclpy.spin(visualizer)

    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
