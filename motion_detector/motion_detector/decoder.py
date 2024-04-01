#!/usr/bin/env python3


import rclpy
from rclpy.node import Node

from rtabmap_msgs.msg import CameraCallback
from sensor_msgs.msg import Image


class Decoder(Node):
    
    def __init__(self):
        super().__init__('motion_detector_msg_decoder')
        self.subscription_ = self.create_subscription(CameraCallback, '/rtabmap/from_motion_detector', self.callback, 10)
        self.subscription_  # prevent unused variable warning
        self.image_pub_ = self.create_publisher(Image, '/decoder/image', 10)
        self.depth_pub_ = self.create_publisher(Image, '/decoder/depth', 10)
        self.mask_pub_ = self.create_publisher(Image, '/decoder/mask', 10)

    def callback(self, msg):
        self.image_pub_.publish(msg.image)
        self.depth_pub_.publish(msg.depth)
        self.mask_pub_.publish(msg.mask)
        print(
            msg.image.header.stamp.sec,
            msg.image.header.stamp.nanosec,
            msg.depth.header.stamp.sec,
            msg.depth.header.stamp.nanosec,
            msg.mask.header.stamp.sec,
            msg.mask.header.stamp.nanosec,
            msg.image.header.frame_id,
            msg.image.header.frame_id,
            msg.image.header.frame_id
        )

def main(args=None):
    rclpy.init(args=args)
    
    decoder = Decoder()

    rclpy.spin(decoder)

    decoder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
