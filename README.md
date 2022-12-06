# PointPillars Ros2 package
IVlab 水源實驗室 IPC版本

## Prerequirement:  
- [ROS2 Galatic](https://docs.ros.org/en/galactic/Installation/Ubuntu-Install-Debians.html)  
- [MMdetection3d](https://github.com/open-mmlab/mmdetection3d)  
- [pypcd](https://github.com/dimatura/pypcd)  

## ***New:***
New branch for 3060ti IPC version  
Fix argparse bug  
Add warning for thershold
  
## ***Note:***  
pypcd github repository 有一些 bug  
TODO 未來更新修改內容  
## To use:
Build ROS package  
```
colcon build
source install/setup.bash
```
Run pointpillars model
```
ros2 run pointpillars pointpillars -t (desire threshold)
```
