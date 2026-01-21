import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load the Point Cloud Data ---
# Change variable name to 'points' for clarity
points = np.loadtxt("MapPointCloud.txt")
# The trajectory file is loaded separately later if you want to plot it.

# --- 2. Extract Coordinates (0-indexed columns) ---
# MapPointCloud.txt contains X, Y, Z coordinates only.
X_points, Y_points, Z_points = points[:,0], points[:,1], points[:,2]

# --- 3. Optional: Load Trajectory Data for Context ---
traj = np.loadtxt("KeyFrameTrajectory.txt")
# The TUM trajectory format is: timestamp X Y Z QX QY QZ QW
X_traj, Y_traj, Z_traj = traj[:,1], traj[:,2], traj[:,3]


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the Point Cloud using scatter
ax.scatter(X_points, Y_points, Z_points, s=1, c='blue', marker='.', label='Map Points')

# Plot the Camera Trajectory (KeyFrame Poses) using plot
ax.plot(X_traj, Y_traj, Z_traj, color='red', linestyle='-', linewidth=2, label='KeyFrame Trajectory')

# --- Adjustments for Better Visualization ---
ax.set_xlabel('X (meters)', color='r')
ax.set_ylabel('Y (meters)', color='g')
ax.set_zlabel('Z (meters)', color='b')
ax.set_title('ORB-SLAM3 Point Cloud and KeyFrame Trajectory')
ax.legend()

# Set equal aspect ratio for realistic 3D viewing
max_range = np.array([X_points.max()-X_points.min(), Y_points.max()-Y_points.min(), Z_points.max()-Z_points.min()]).max() / 2.0
mid_x = (X_points.max()+X_points.min()) * 0.5
mid_y = (Y_points.max()+Y_points.min()) * 0.5
mid_z = (Z_points.max()+Z_points.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()
