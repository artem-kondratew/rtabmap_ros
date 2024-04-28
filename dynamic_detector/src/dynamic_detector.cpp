#include <iostream>
#include <memory>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "rtabmap_msgs/msg/motion_detector_data.hpp"

#include "../include/dynamic_detector/depth_conversions.hpp"


class DynamicDetector : public rclcpp::Node {
private:
    rclcpp::Subscription<rtabmap_msgs::msg::MotionDetectorData>::SharedPtr input_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc2_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr box_pub_;

    image_geometry::PinholeCameraModel camera_model_;
    float range_max_;

public:
    DynamicDetector();

private:
    std::vector<cv::Point> findCenters(std::vector<uint64_t> boxes);
    std::vector<cv::Point3f> depth2XYZ(const sensor_msgs::msg::Image& depth, sensor_msgs::msg::PointCloud2::SharedPtr point_cloud);
    std::vector<std::vector<cv::Point>> setImagePoints(std::vector<uint64_t> boxes);
    std::vector<std::vector<cv::Point3f>> getRealPoints(size_t w, std::vector<std::vector<cv::Point>> image_points,
                                                                                    std::vector<cv::Point3f> xyz, std::vector<cv::Point> centers);
    std::vector<std::vector<cv::Point3f>> setRealBox(std::vector<std::vector<cv::Point3f>> real_points);
    void createLine(visualization_msgs::msg::Marker& line_list, geometry_msgs::msg::Point pt0, geometry_msgs::msg::Point pt1);
    visualization_msgs::msg::Marker createBoxMsg(std::vector<std::vector<cv::Point3f>> real_points);

    void drawBoxes(cv::Mat image, std::vector<uint64_t> boxes, std::vector<cv::Point> centers, std::vector<std::vector<cv::Point>> image_points);

    void callback(const rtabmap_msgs::msg::MotionDetectorData::ConstSharedPtr msg);
};


DynamicDetector::DynamicDetector() : Node("dynamic_detector") {
    using std::placeholders::_1;

    this->declare_parameter("input_topic", "");
    this->declare_parameter("range_max", 0.0);

    std::string input_topic = this->get_parameter("input_topic").as_string();
    range_max_ = this->get_parameter("range_max").as_double();

    RCLCPP_INFO(this->get_logger(), "input_topic: '%s'", input_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "range_max: '%f'", range_max_);

    input_sub_ = this->create_subscription<rtabmap_msgs::msg::MotionDetectorData>(input_topic, 10, std::bind(&DynamicDetector::callback, this, _1));
    pc2_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/rtabmap/dynamic_detector/point_cloud2", 10);
    box_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/rtabmap/dynamic_detector/box", 10);
}


std::vector<cv::Point> DynamicDetector::findCenters(std::vector<uint64_t> boxes) {
    std::vector<cv::Point> centers;
    for (size_t i = 0; i < boxes.size(); i += 4) {
        int cx = static_cast<int>((boxes[i+0] + boxes[i+2]) / 2);
        int cy = static_cast<int>((boxes[i+1] + boxes[i+3]) / 2);
        centers.push_back(cv::Point{cx, cy});
    }
    return centers;
}


std::vector<cv::Point3f> DynamicDetector::depth2XYZ(const sensor_msgs::msg::Image& depth, sensor_msgs::msg::PointCloud2::SharedPtr point_cloud) {
    point_cloud->header.frame_id = "camera_link_optical";
    point_cloud->header.stamp = this->get_clock()->now();
    point_cloud->height = depth.height;
    point_cloud->width = depth.width;
    point_cloud->is_dense = false;
    point_cloud->is_bigendian = false;
    point_cloud->fields.clear();
    point_cloud->fields.reserve(2);

    sensor_msgs::PointCloud2Modifier pcd_modifier(*point_cloud);
    pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

    std::vector<cv::Point3f> xyz = depthimage_to_pointcloud2::convert<float>(depth, point_cloud, camera_model_, range_max_, false);

    return xyz;
}


std::vector<std::vector<cv::Point>> DynamicDetector::setImagePoints(std::vector<uint64_t> boxes) {
    std::vector<std::vector<cv::Point>> points;
    for (size_t i = 0; i < boxes.size(); i += 4) {
        std::vector<cv::Point> pts;
        cv::Point ulf {int(boxes[i+0]), int(boxes[i+1])};
        cv::Point urf {int(boxes[i+2]), int(boxes[i+1])};
        cv::Point llf {int(boxes[i+0]), int(boxes[i+3])};
        cv::Point lrf {int(boxes[i+2]), int(boxes[i+3])};
        pts.push_back(ulf);
        pts.push_back(urf);
        pts.push_back(llf);
        pts.push_back(lrf);
        points.push_back(pts);
    }
    return points;
}


std::vector<std::vector<cv::Point3f>> DynamicDetector::getRealPoints(
    size_t w,
    std::vector<std::vector<cv::Point>> image_points,
    std::vector<cv::Point3f> xyz,
    std::vector<cv::Point> centers) {
    std::vector<std::vector<cv::Point3f>> real_points;
    for (size_t i = 0; i < image_points.size(); i++) {
        std::vector<cv::Point> image_pts = image_points[i];
        std::vector<cv::Point3f> real_pts;
        cv::Point3f real_center = xyz[centers[i].y*w+centers[i].x];
        for (size_t j = 0; j < image_pts.size(); j++) {
            cv::Point3f real_pt = xyz[image_pts[j].y*w+image_pts[j].x];
            real_pt.z = real_center.z;
            real_pts.push_back(real_pt);
        }
        real_points.push_back(real_pts);
    }
    return real_points;
}


std::vector<std::vector<cv::Point3f>> DynamicDetector::setRealBox(std::vector<std::vector<cv::Point3f>> real_points) {
    std::vector<std::vector<cv::Point3f>> all_points;
    for (size_t i = 0; i < real_points.size(); i++) {
        std::vector<cv::Point3f> real_pts = real_points[i];
        float l = real_pts[3].x - real_pts[2].x;
        std::vector<cv::Point3f> all_pts = real_pts;
        for (size_t j = 0; j < real_pts.size(); j++) {
            cv::Point3f pt = real_pts[j];
            pt.z += l;
            all_pts.push_back(pt);
        }
        all_points.push_back(all_pts);
        std::cout << all_pts.size() << std::endl;
    }
    std::cout << all_points.size() << std::endl;
    return all_points;
}


void DynamicDetector::createLine(visualization_msgs::msg::Marker& line_list, geometry_msgs::msg::Point pt0, geometry_msgs::msg::Point pt1) {
    line_list.points.push_back(pt0);
    line_list.points.push_back(pt1);
}


visualization_msgs::msg::Marker DynamicDetector::createBoxMsg(std::vector<std::vector<cv::Point3f>> real_points) {
    visualization_msgs::msg::Marker line_list;
    line_list.header.frame_id = "camera_link_optical";
    line_list.header.stamp = this->get_clock()->now();
    line_list.id = 2;
    line_list.type = visualization_msgs::msg::Marker::LINE_LIST;
    line_list.scale.x = 0.01;
    line_list.color.b = 1.0; // blue
    line_list.color.a = 1.0; // alpha

    for (size_t i = 0; i < real_points.size(); i++) {
        std::vector<cv::Point3f> real_pts = real_points[i];
        std::vector<geometry_msgs::msg::Point> geometry_pts;
        for (size_t j = 0; j < real_pts.size(); j++) {
            auto pt = geometry_msgs::msg::Point();
            pt.x = real_pts[j].x;
            pt.y = real_pts[j].y;
            pt.z = real_pts[j].z;
            geometry_pts.push_back(pt);
        }

        geometry_msgs::msg::Point ulf = geometry_pts[0];
        geometry_msgs::msg::Point urf = geometry_pts[1];
        geometry_msgs::msg::Point llf = geometry_pts[2];
        geometry_msgs::msg::Point lrf = geometry_pts[3];
        geometry_msgs::msg::Point ulb = geometry_pts[4];
        geometry_msgs::msg::Point urb = geometry_pts[5];
        geometry_msgs::msg::Point llb = geometry_pts[6];
        geometry_msgs::msg::Point lrb = geometry_pts[7];

        createLine(line_list, ulf, urf);
        createLine(line_list, ulf, llf);
        createLine(line_list, llf, lrf);
        createLine(line_list, urf, lrf);
        
        createLine(line_list, ulb, urb);
        createLine(line_list, ulb, llb);
        createLine(line_list, llb, lrb);
        createLine(line_list, urb, lrb);

        createLine(line_list, ulf, ulb);
        createLine(line_list, urf, urb);
        createLine(line_list, llf, llb);
        createLine(line_list, lrf, lrb);
    }
    return line_list;
}


void DynamicDetector::drawBoxes(cv::Mat image, std::vector<uint64_t> boxes, std::vector<cv::Point> centers, std::vector<std::vector<cv::Point>> image_points) {
    for (size_t i = 0; i < boxes.size(); i += 4) {
        cv::rectangle(image, cv::Rect(cv::Point{int(boxes[i]), int(boxes[i+1])}, cv::Point{int(boxes[i+2]), int(boxes[i+3])}), {255, 0, 255}, 1);
        cv::circle(image, centers[i/4], 4, {255, 0, 255}, 1);
    }

    for (size_t i = 0; i < image_points.size(); i++) {
        auto pts = image_points[i];
        cv::circle(image, pts[0], 4, {255, 0, 255}, 1);
        cv::circle(image, pts[1], 4, {255, 0, 255}, 1);
        cv::circle(image, pts[2], 4, {255, 0, 255}, 1);
        cv::circle(image, pts[3], 4, {255, 0, 255}, 1);
    }
}


void DynamicDetector::callback(const rtabmap_msgs::msg::MotionDetectorData::ConstSharedPtr msg) {
    size_t obstacles_num = msg->boxes.size() / 4;
    RCLCPP_INFO(this->get_logger(), "%ld obstacle(s) detected", obstacles_num);

    size_t w = msg->mask.width;
    size_t h = msg->mask.height;
    size_t size = w * h;
    cv::Mat rgb(h, w, CV_8UC3);
    cv::Mat depth(h, w, CV_32FC1);
    cv::Mat mask(h, w, CV_8UC1);
    std::memcpy(rgb.data, msg->rgb.data.data(), sizeof(uint8_t) * size * 3);
    std::memcpy(depth.data, msg->depth.data.data(), sizeof(float) * size * 1);
    std::memcpy(mask.data, msg->mask.data.data(), sizeof(uint8_t) * size * 1); 

    std::vector<uint64_t> boxes;
    boxes.resize(msg->boxes.size());
    std::memcpy(boxes.data(), msg->boxes.data(), msg->boxes.size() * sizeof(uint64_t));

    camera_model_.fromCameraInfo(msg->camera_info);

    sensor_msgs::msg::PointCloud2::SharedPtr point_cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    std::vector<cv::Point3f> xyz = depth2XYZ(msg->depth, point_cloud);

    std::vector<std::vector<cv::Point>> image_points = setImagePoints(boxes);

    std::vector<cv::Point> centers = findCenters(boxes);
    std::vector<std::vector<cv::Point3f>> front_real_points = getRealPoints(w, image_points, xyz, centers);
    std::vector<std::vector<cv::Point3f>> real_points = setRealBox(front_real_points);

    visualization_msgs::msg::Marker line_list = createBoxMsg(real_points);

    pc2_pub_->publish(*point_cloud);
    box_pub_->publish(line_list);

    drawBoxes(rgb, boxes, centers, image_points);
    cv::normalize(depth, depth, 1, 0, cv::NORM_MINMAX);
    cv::imshow("depth", depth);
    cv::imshow("mask", mask);
    cv::imshow("image", rgb);
    cv::waitKey(20);
}


int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamicDetector>()); 
    rclcpp::shutdown();
    return 0;
}
