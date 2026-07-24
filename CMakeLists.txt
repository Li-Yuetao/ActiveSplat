cmake_minimum_required(VERSION 3.8)
project(activesplat)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# ===========================
# Messages
# ===========================

set(msg_files
  "msg/Frame.msg"
)

# ===========================
# Services
# ===========================

set(srv_files
  "srv/GetDatasetConfig.srv"
  "srv/ResetEnv.srv"
  "srv/GetTopdownConfig.srv"
  "srv/GetTopdown.srv"
  "srv/SetPlannerState.srv"
  "srv/SetMapper.srv"
  "srv/GetOpacity.srv"
  "srv/GetVoronoiGraph.srv"
  "srv/GetNavPath.srv"
)

# ===========================
# Generate interfaces
# ===========================

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
)

# ===========================
# Install Python scripts
# ===========================

ament_get_python_install_dir(PYTHON_INSTALL_DIR)
install(
  DIRECTORY
    src/dataloader
    src/mapper
    src/planner
    src/utils
    src/visualizer
  DESTINATION ${PYTHON_INSTALL_DIR}
)

install(PROGRAMS
  scripts/nodes/mapper_node.py
  scripts/nodes/planner_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ===========================
# Install launch
# ===========================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/launch/habitat.launch.py)
  install(
    FILES launch/habitat.launch.py
    DESTINATION share/${PROJECT_NAME}/launch
  )
endif()

# ===========================
# Install config
# ===========================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
  install(
    DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
  )
endif()

# ===========================
# Install rviz
# ===========================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config/rviz)
  install(
    DIRECTORY config/rviz/
    DESTINATION share/${PROJECT_NAME}/config/rviz
  )
endif()

ament_export_dependencies(rosidl_default_runtime)

ament_package()