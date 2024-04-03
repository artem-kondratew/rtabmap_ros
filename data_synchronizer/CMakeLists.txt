cmake_minimum_required(VERSION 3.8)
project(data_synchronizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(message_filters REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rtabmap_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(data_synchronizer
    src/data_synchronizer.cpp
)

ament_target_dependencies(data_synchronizer message_filters nav_msgs rclcpp rtabmap_msgs sensor_msgs)

install(DIRECTORY
    src config
    DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
    data_synchronizer
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()
