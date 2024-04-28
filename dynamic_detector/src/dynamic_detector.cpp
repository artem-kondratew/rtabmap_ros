#include <iostream>
#include <memory>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
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
    double unit_scaling_;

public:
    DynamicDetector();

private:
    std::vector<cv::Point> findCenters(std::vector<uint64_t> boxes);
    std::vector<geometry_msgs::msg::Point> depth2XYZ(const sensor_msgs::msg::Image& depth, sensor_msgs::msg::PointCloud2::SharedPtr point_cloud, double* unit_scaling);
    std::vector<std::vector<cv::Point>> setImagePoints(std::vector<uint64_t> boxes);
    std::vector<std::vector<geometry_msgs::msg::Point>> getRealPoints(size_t w, std::vector<std::vector<cv::Point>> image_points,
                                                                                    std::vector<geometry_msgs::msg::Point> xyz, std::vector<cv::Point> centers);
    std::vector<std::vector<geometry_msgs::msg::Point>> setRealBox(std::vector<std::vector<geometry_msgs::msg::Point>> real_points);
    double calcUpperY(geometry_msgs::msg::Point lower_point, cv::Point image_lower_point);
    void createLine(visualization_msgs::msg::Marker& line_list, geometry_msgs::msg::Point pt0, geometry_msgs::msg::Point pt1);
    visualization_msgs::msg::Marker createBoxMsg(std::vector<std::vector<geometry_msgs::msg::Point>> real_points, std::vector<std::vector<cv::Point>> image_points);

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


std::vector<geometry_msgs::msg::Point> DynamicDetector::depth2XYZ(const sensor_msgs::msg::Image& depth, sensor_msgs::msg::PointCloud2::SharedPtr point_cloud, double* unit_scaling) {
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

    std::vector<geometry_msgs::msg::Point> xyz = depthimage_to_pointcloud2::convert<float>(depth, point_cloud, camera_model_, range_max_, unit_scaling);

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


std::vector<std::vector<geometry_msgs::msg::Point>> DynamicDetector::getRealPoints(size_t w, std::vector<std::vector<cv::Point>> image_points,
                                                                                        std::vector<geometry_msgs::msg::Point> xyz, std::vector<cv::Point> centers) {
    std::vector<std::vector<geometry_msgs::msg::Point>> real_points;
    for (size_t i = 0; i < image_points.size(); i++) {
        std::vector<cv::Point> image_pts = image_points[i];
        std::vector<geometry_msgs::msg::Point> real_pts;
        geometry_msgs::msg::Point real_center = xyz[centers[i].y*w+centers[i].x];
        for (size_t j = 0; j < image_pts.size(); j++) {
            geometry_msgs::msg::Point real_pt = xyz[image_pts[j].y*w+image_pts[j].x];
            real_pt.z = real_center.z;
            real_pts.push_back(real_pt);
        }
        real_points.push_back(real_pts);
    }
    return real_points;
}


std::vector<std::vector<geometry_msgs::msg::Point>> DynamicDetector::setRealBox(std::vector<std::vector<geometry_msgs::msg::Point>> real_points) {
    std::vector<std::vector<geometry_msgs::msg::Point>> all_points;
    for (size_t i = 0; i < real_points.size(); i++) {
        std::vector<geometry_msgs::msg::Point> real_pts = real_points[i];
        float l = real_pts[3].x - real_pts[2].x;
        std::vector<geometry_msgs::msg::Point> all_pts = real_pts;
        for (size_t j = 0; j < real_pts.size(); j++) {
            all_pts[j].z -= 0.5 * l;
            geometry_msgs::msg::Point pt = real_pts[j];
            pt.z += l;
            all_pts.push_back(pt);
        }
        all_points.push_back(all_pts);
    }
    return all_points;
}


double DynamicDetector::calcUpperY(geometry_msgs::msg::Point lower_point, cv::Point image_upper_point) {
   return (image_upper_point.y - camera_model_.cy()) * lower_point.z * unit_scaling_ / camera_model_.fy();
}


void DynamicDetector::createLine(visualization_msgs::msg::Marker& line_list, geometry_msgs::msg::Point pt0, geometry_msgs::msg::Point pt1) {
    line_list.points.push_back(pt0);
    line_list.points.push_back(pt1);
}


visualization_msgs::msg::Marker DynamicDetector::createBoxMsg(std::vector<std::vector<geometry_msgs::msg::Point>> real_points,
                                                                                                            std::vector<std::vector<cv::Point>> image_points) {
    visualization_msgs::msg::Marker line_list;
    line_list.header.frame_id = "camera_link_optical";
    line_list.header.stamp = this->get_clock()->now();
    line_list.id = 2;
    line_list.type = visualization_msgs::msg::Marker::LINE_LIST;
    line_list.scale.x = 0.01;
    line_list.color.b = 1.0; // blue
    line_list.color.a = 1.0; // alpha

    for (size_t i = 0; i < real_points.size(); i++) {
        std::vector<geometry_msgs::msg::Point> real_pts = real_points[i];
        std::vector<cv::Point> image_pts = image_points[i];

        geometry_msgs::msg::Point ulf = real_pts[0];
        geometry_msgs::msg::Point urf = real_pts[1];
        geometry_msgs::msg::Point llf = real_pts[2];
        geometry_msgs::msg::Point lrf = real_pts[3];
        geometry_msgs::msg::Point ulb = real_pts[4];
        geometry_msgs::msg::Point urb = real_pts[5];
        geometry_msgs::msg::Point llb = real_pts[6];
        geometry_msgs::msg::Point lrb = real_pts[7];

        ulb.x = llb.x; ulb.z = llb.z; ulb.y = calcUpperY(llb, image_pts[0]);
        urb.x = lrb.x; urb.z = lrb.z; urb.y = calcUpperY(lrb, image_pts[1]);
        ulf.x = llf.x; ulf.z = llf.z; ulf.y = ulb.y;
        urf.x = lrf.x; urf.z = lrf.z; urf.y = urb.y;

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
    std::vector<geometry_msgs::msg::Point> xyz = depth2XYZ(msg->depth, point_cloud, &unit_scaling_);

    std::vector<std::vector<cv::Point>> image_points = setImagePoints(boxes);

    std::vector<cv::Point> centers = findCenters(boxes);
    std::vector<std::vector<geometry_msgs::msg::Point>> front_real_points = getRealPoints(w, image_points, xyz, centers);
    std::vector<std::vector<geometry_msgs::msg::Point>> real_points = setRealBox(front_real_points);

    visualization_msgs::msg::Marker line_list = createBoxMsg(real_points, image_points);

    pc2_pub_->publish(*point_cloud);
    box_pub_->publish(line_list);

    drawBoxes(rgb, boxes, centers, image_points);
    cv::normalize(depth, depth, 1, 0, cv::NORM_MINMAX);
    // cv::imshow("depth", depth);
    // cv::imshow("mask", mask);
    cv::imshow("image", rgb);
    cv::waitKey(20);
}


int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamicDetector>()); 
    rclcpp::shutdown();
    return 0;
}
