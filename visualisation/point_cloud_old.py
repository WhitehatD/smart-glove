import pandas as pd
import plotly.graph_objects as go
import numpy as np

def plot_full_slam(traj_file, points_file, 
                   flip_x=False, flip_y=False, flip_z=False, swap_y_z=False,
                   traj_scale=1.0, 
                   offset_x=0.0, offset_y=0.0, offset_z=0.0):
    
    # Load Trajectory
    df_t = pd.read_csv(traj_file, sep=' ', header=None, 
                        names=['ts','x','y','z','qx','qy','qz','qw'])
    
    # Load Map Points
    df_p = pd.read_csv(points_file, sep=' ', header=None, names=['x','y','z'])

    # 1. --- SWAP AXES ---
    if swap_y_z:
        for df in [df_t, df_p]:
            tmp = df['y'].copy()
            df['y'] = df['z']
            df['z'] = tmp

    # 2. --- TRAJECTORY SCALE & OFFSET ---
    df_t['x'] = (df_t['x'] * traj_scale) + offset_x
    df_t['y'] = (df_t['y'] * traj_scale) + offset_y
    df_t['z'] = (df_t['z'] * traj_scale) + offset_z

    # 3. --- MIRRORING ---
    if flip_x:
        df_t['x'] *= -1
        df_p['x'] *= -1
    if flip_y:
        df_t['y'] *= -1
        df_p['y'] *= -1
    if flip_z:
        df_t['z'] *= -1
        df_p['z'] *= -1

    fig = go.Figure()

    # 4. THE CAMERA PATH (With Time Flow)
    # We use mode='lines+markers' so you can see individual keyframes
    fig.add_trace(go.Scatter3d(
        x=df_t['x'], y=df_t['y'], z=df_t['z'], 
        mode='lines+markers',
        marker=dict(
            size=4,
            color=df_t['ts'],       # Color based on timestamp
            colorscale='Viridis',   # Purple = Start, Yellow = End
            showscale=True,
            colorbar=dict(title="Time (Start -> End)", thickness=15, x=0)
        ),
        line=dict(color='rgba(255, 255, 255, 0.5)', width=3), # Faded white line to connect dots
        name='Camera Path'
    ))

    # 5. THE START POINT (Special Marker)
    fig.add_trace(go.Scatter3d(
        x=[df_t['x'].iloc[0]], y=[df_t['y'].iloc[0]], z=[df_t['z'].iloc[0]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='START HERE'
    ))

    # 6. THE POINT CLOUD
    fig.add_trace(go.Scatter3d(
        x=df_p['x'], y=df_p['y'], z=df_p['z'], 
        mode='markers', 
        marker=dict(size=1.5, color='rgb(150, 150, 150)', opacity=0.5), 
        name='Environment'
    ))

    fig.update_layout(
        template="plotly_dark",
        scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title="ORB-SLAM3: Directional Flow & Environment"
    )
    
    fig.show(renderer="browser")

# --- EXECUTE ---
plot_full_slam('/Users/batigozen/Desktop/ORB_SLAM3_Dataset/testkey.txt', '/Users/batigozen/Desktop/ORB_SLAM3_Dataset/testcloud.txt', 
               flip_y=True, 
               traj_scale=1.0) # Set this to 0.5 if the path is too "long" for the room