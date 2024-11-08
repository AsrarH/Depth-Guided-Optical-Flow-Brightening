import cv2
import numpy as np
import glob

# Define file paths
video_path = 'input_video.mp4'  # Path to input video
depth_map_folder = 'source_depth/'  # Path to folder containing depth maps

# Initialize video capture and output settings
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the video writer to save the output video
output_video = cv2.VideoWriter(
    'output_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# Load all depth maps, ensuring they are sorted in the correct frame order
depth_maps = sorted(glob.glob(depth_map_folder + "*.png"))

# Verify that video and depth maps are loaded correctly
ret, first_frame = cap.read()
if not ret:
    print("Failed to read the video.")
    cap.release()
    output_video.release()
    exit()

# Set initial variables for optical flow processing
frame_index = 0
prev_gray = None  # Placeholder for the previous frame in grayscale

# Processing loop for each video frame
while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    if not ret or frame_index >= len(depth_maps):
        break

    # Convert the frame to grayscale for optical flow calculation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Set up the previous frame if not already initialized
    if prev_gray is None:
        prev_gray = gray
        frame_index += 1
        continue

    # Calculate the optical flow between the current and previous frame
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_magnitude = np.linalg.norm(flow, axis=2)

    # Estimate and subtract global motion to approximate camera movement
    median_flow_x = np.median(flow[..., 0])
    median_flow_y = np.median(flow[..., 1])
    flow[..., 0] -= median_flow_x
    flow[..., 1] -= median_flow_y
    flow_magnitude = np.linalg.norm(flow, axis=2)

    # Load and process the corresponding depth map for the current frame
    depth_map = cv2.imread(depth_maps[frame_index], cv2.IMREAD_GRAYSCALE)
    depth_map_resized = cv2.resize(depth_map, (frame_width, frame_height))
    depth_normalized = cv2.normalize(depth_map_resized, None, 0, 1, cv2.NORM_MINMAX)

    # Calculate scene flow as a depth-weighted flow magnitude
    scene_flow = flow_magnitude * (1 - depth_normalized) ** 1.5

    # Apply a dynamic threshold based on the 90th percentile of scene flow values
    if np.any(scene_flow > 0):
        dynamic_threshold = np.percentile(scene_flow[scene_flow > 0], 90)
    else:
        dynamic_threshold = 0  # Set to 0 if no significant scene flow is detected

    # Create a brightness mask where scene flow exceeds the dynamic threshold
    brightness_mask = np.where(scene_flow > dynamic_threshold, scene_flow, 0).astype(np.uint8)

    # Normalize the brightness mask to enhance visual contrast
    brightness_mask = cv2.normalize(brightness_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a color map to the brightness mask for visual effect
    brightness_overlay = cv2.applyColorMap(brightness_mask, cv2.COLORMAP_HOT)

    # Blend the brightness overlay with the original frame
    brightened_frame = cv2.addWeighted(frame, 0.6, brightness_overlay, 0.4, 0)

    # Write the processed frame to the output video
    output_video.write(brightened_frame)

    # Update the previous frame for the next optical flow calculation
    prev_gray = gray
    frame_index += 1

# Release all resources
cap.release()
output_video.release()
print("Processing completed. Check output_video.mp4")
