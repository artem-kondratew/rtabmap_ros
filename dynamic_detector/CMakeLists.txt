cmake_minimum_required(VERSION 3.8)
project(dynamic_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rtabmap_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(dynamic_detector
    include/dynamic_detector/depth_conversions.hpp
    src/dynamic_detector.cpp
)

target_link_libraries(dynamic_detector ${OpenCV_LIBS})

ament_target_dependencies(dynamic_detector
    cv_bridge
    geometry_msgs
    image_geometry
    nav_msgs
    rclcpp
    rtabmap_msgs
    sensor_msgs
    visualization_msgs
)

install(DIRECTORY
    src config include
    DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
    dynamic_detector
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
