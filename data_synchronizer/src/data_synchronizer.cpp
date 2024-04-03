#include <iostream>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "rtabmap_msgs/msg/motion_detector_data.hpp"


using std::placeholders::_1;


class DataSyncronizer : public rclcpp::Node {
private:
    using approximate_policy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::CameraInfo,
    sensor_msgs::msg::Image, sensor_msgs::msg::Image, nav_msgs::msg::Odometry>;
    
    std::unique_ptr<message_filters::Synchronizer<approximate_policy>> sync_;

    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;

    rclcpp::Publisher<rtabmap_msgs::msg::MotionDetectorData>::SharedPtr output_pub_;

public:
    DataSyncronizer();

private:
    void callback(
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info,
        const sensor_msgs::msg::Image::ConstSharedPtr rgb,
        const sensor_msgs::msg::Image::ConstSharedPtr depth,
        const nav_msgs::msg::Odometry::ConstSharedPtr odom
    );
};


DataSyncronizer::DataSyncronizer() : Node("data_synchronizer") {
    this->declare_parameter("camera_info_input", "");
    this->declare_parameter("rgb_input", "");
    this->declare_parameter("depth_input", "");
    this->declare_parameter("odom_input", "");
    this->declare_parameter("output", "");

    std::string camera_info_topic = this->get_parameter("camera_info_input").as_string();
    std::string rgb_topic = this->get_parameter("rgb_input").as_string();
    std::string depth_topic = this->get_parameter("depth_input").as_string();
    std::string odom_topic = this->get_parameter("odom_input").as_string();
    std::string output_topic = this->get_parameter("output").as_string();

    RCLCPP_INFO(this->get_logger(), "camera_info_topic: '%s'", camera_info_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "rgb_topic: '%s'", rgb_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "depth_topic: '%s'", depth_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "odom_topic: '%s'", odom_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "output_topic: '%s'", output_topic.c_str());

    camera_info_sub_.subscribe(this, camera_info_topic);
    rgb_sub_.subscribe(this, rgb_topic);
    depth_sub_.subscribe(this, depth_topic);
    odom_sub_.subscribe(this, odom_topic);

    sync_.reset(new message_filters::Synchronizer<approximate_policy>(approximate_policy(10), camera_info_sub_, rgb_sub_, depth_sub_, odom_sub_));
    sync_->registerCallback(std::bind(&DataSyncronizer::callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

    output_pub_ = this->create_publisher<rtabmap_msgs::msg::MotionDetectorData>(output_topic, 10);
}


void DataSyncronizer::callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info, const sensor_msgs::msg::Image::ConstSharedPtr rgb,
                               const sensor_msgs::msg::Image::ConstSharedPtr depth, const nav_msgs::msg::Odometry::ConstSharedPtr odom) {
    auto msg = rtabmap_msgs::msg::MotionDetectorData();
    msg.camera_info = *camera_info;
    msg.rgb = *rgb;
    msg.depth = *depth;
    msg.odom = *odom;

    output_pub_->publish(msg);

    RCLCPP_INFO(this->get_logger(), "sync created: %u:%u %u:%u %u:%u", msg.rgb.header.stamp.sec, msg.rgb.header.stamp.nanosec,
    msg.depth.header.stamp.sec, msg.depth.header.stamp.nanosec, msg.odom.header.stamp.sec, msg.odom.header.stamp.nanosec);
}


int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DataSyncronizer>()); 
    rclcpp::shutdown();
    return 0;
}
