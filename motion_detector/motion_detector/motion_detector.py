#!/usr/bin/env python3


import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from data_synchronizer_msgs.msg import DetectorInput, DetectorOutput

from .submodules.yolo import Yolo
from .submodules.yolo_classes import classes_list


class MotionDetector(Node):
    
    def __init__(self):
        super().__init__('motion_detector')
        self.declare_parameter('input_topic', '')
        self.declare_parameter('output_topic', '')
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.get_logger().info(f'input_topic: {input_topic}')
        self.get_logger().info(f'output_topic: {output_topic}')
        self.subscription_ = self.create_subscription(DetectorInput, input_topic, self.callback, 10)
        self.subscription_  # prevent unused variable warning
        self.publisher_ = self.create_publisher(DetectorOutput, output_topic, 10)
        self.bridge_ = CvBridge()
        self.model_ = Yolo(classes=classes_list)
        self.get_logger().info('model initialized')

    def callback(self, msg):
        cv_frame = self.bridge_.imgmsg_to_cv2(msg.rgb, desired_encoding='passthrough')
        output = self.model_.run(cv_frame)
        success, masks = self.model_.merge_masks(output.masks)
        if not success:
            masks = np.zeros(cv_frame.shape, dtype='uint8')
            self.get_logger().info('no_masks')
        pub_msg = DetectorOutput()
        pub_msg.camera_info = msg.camera_info
        pub_msg.rgb = msg.rgb
        pub_msg.depth = msg.depth
        pub_msg.mask = self.bridge_.cv2_to_imgmsg(masks, encoding='passthrough')
        pub_msg.odom = msg.odom
        self.publisher_.publish(pub_msg)
        cv.imshow('main_mask', masks)
        cv.imshow('frame', cv_frame)
        cv.waitKey(20)

    def fake_callback(self, msg):
        cv_frame = self.bridge_.imgmsg_to_cv2(msg.rgb, desired_encoding='passthrough')   
        h, w = cv_frame.shape[:2]
        copy = np.copy(cv_frame)
        cv.circle(copy, (w // 2, h // 2), 20, (255, 0, 255), -1)
        # mask_msg = self.bridge.cv2_to_imgmsg(copy, encoding='passthrough')
        # self.publisher.publish(mask_msg)
    

def main(args=None):
    rclpy.init(args=args)
    
    motion_detector = MotionDetector()

    rclpy.spin(motion_detector)

    motion_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
