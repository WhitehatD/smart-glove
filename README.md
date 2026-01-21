# ORB_SLAM3 docker

This docker is based on <b>Ros Noetic Ubuntu 20</b>

There are two versions available:

- CPU based (Xorg Nouveau display)
- Nvidia Cuda based.

To check if you are running the nvidia driver, simply run `nvidia-smi` and see if get anything.

Based on which graphic driver you are running, you should choose the proper docker. For cuda version, you need to have [nvidia-docker setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your machine.

---

## Compilation and Running

Steps to compile the Orbslam3 on the sample dataset:

- `./download_dataset_sample.sh`
- `build_container_cpu.sh` or `build_container_cuda.sh` depending on your machine.

Now you should see ORB_SLAM3 is compiling.

- Run download_dataset_sample.sh only once. This will download MH01_easy dataset from EuRoC MAV Dataset.
-

To run a test example:

- `docker exec -it orbslam3 bash`
- `cd /ORB_SLAM3/Examples && bash ./euroc_examples.sh`
It will take few minutes to initialize. Please Be patient.

Progress logs: Each example prints a line every ~50 frames like `[mono_euroc] progress: 150/3620`. Final completion line shows total frames processed.

---

You can use vscode remote development (recommended) or sublime to change codes.

- `docker exec -it orbslam3 bash`
- `subl /ORB_SLAM3`

---

## Repository Structure

### Calibration

Contains a comprehensive suite of tools for camera-IMU calibration and data formatting compatible with ORB-SLAM3 and Kalibr.

- **Recording**: `stereo_recorder_euroc.py`, `record_with_imu.py` for capturing data with precise timestamps.
- **Preprocessing**: `generate_timestamps.py`, `fix_time.py` for EuroC format compliance.
- **Conversion**: `kalibr_to_orbslam3.py`, `opencv_to_kalibr.py` to bridge different calibration file formats.

### Visualisation

Tools to inspect the resulting point clouds and trajectories.

- **Python Viewer**: `point_cloud_advanced.py` is a robust Open3D-based viewer with features like outlier removal, auto-scaling, and frustum visualization.
- **Web Viewer**: Located in `pointPloter - Christian's Visualizer`, provides a browser-based interactive 3D view.

### Experiments

This folder contains the datasets for 5 distinct experiments used for validation.
**Note**: The raw data is too large for version control and is excluded from this repository.

- **[Download Experiments Dataset](https://drive.google.com/uc?export=download&id=10qP89dVOmioyUQNZ_IvoiLDkTQXBEzjVH)**
