# Franka FR3 Robotics Integration
This repository branch contains the ROS 2 Humble development environment and MoveIt 2 configurations for the Franka Emika FR3 robot. The setup is optimized for WSL2 (Ubuntu 22.04) environments and is designed to interface with the project's computer vision modules.
* **Simulation:** Provides a stable MoveIt 2 environment using `fake_hardware` for trajectory validation.
* **Robot Logic:** Contains the `franka_pick_place` package for implementing pick-and-place task execution.
* **Physical Constraints:** Will enforce safety limits based on the FR3 datasheet, including the 855mm reach, 83mm maximum gripper width, and $A4$ joint limits of $-176^\circ$ to $-4^\circ$.

## Setup
Teammates joining this branch are recommended to follow these procedures within a WSL terminal to initialize the workspace.

### 1. Workspace Initialization
```bash
mkdir -p ~/franka_ros2_ws/src
cd ~/franka_ros2_ws/src
# Clone this specific integration branch
git clone -b robotics_integration [https://github.com/suchirss/franka-pick-and-place.git](https://github.com/suchirss/franka-pick-and-place.git)
```

### 2. Dependency Management
Ensure the ROS 2 Humble environment is sourced before resolving dependencies.

```bash
cd ~/franka_ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Build Procedure
To prevent memory allocation failures or hanging in WSL, libfranka and the workspace must be built using a single parallel worker.

```bash
colcon build --symlink-install --packages-select libfranka --parallel-workers 1
colcon build --symlink-install --parallel-workers 1
source install/setup.bash
```

### Running the Simulation
Execute the following command to initialize the FR3 model in RViz with the MoveIt 2 planning scene and fake hardware controllers active.
```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py
  robot_ip:=dont-care use_fake_hardware:=true
```
