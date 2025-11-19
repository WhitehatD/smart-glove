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
