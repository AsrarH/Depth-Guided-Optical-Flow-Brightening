import sys
sys.path.append('core')

import sys
import os
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from raft import RAFT
from utils.utils import InputPadder
from utils import flow_viz

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_frame(frame):
    """Convert a video frame to a tensor for RAFT model."""
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # Convert to tensor
    return frame[None].to(DEVICE)

def compute_optical_flow(model, frame1, frame2):
    """Compute optical flow between two frames using RAFT model."""
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    
    # Run the model and detach the result before converting to NumPy
    flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
    return flow_up[0].permute(1, 2, 0).detach().cpu().numpy()  # Detach before numpy conversion


def apply_brightening_effect(flow_magnitude, depth_map):
    """Apply depth-weighted brightening effect."""
    depth_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    scene_flow = flow_magnitude * (1 - depth_normalized) ** 1.5
    
    # Apply dynamic threshold to enhance high-motion areas
    dynamic_threshold = np.percentile(scene_flow[scene_flow > 0], 90) if np.any(scene_flow > 0) else 0
    brightness_mask = np.where(scene_flow > dynamic_threshold, scene_flow, 0).astype(np.uint8)
    brightness_mask = cv2.normalize(brightness_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply color map and return as overlay
    return cv2.applyColorMap(brightness_mask, cv2.COLORMAP_HOT)

def run(video_path, depth_map_folder, output_path, model_weights='models/raft-things.pth'):
    # Initialize RAFT model
    args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_weights))
    model = model.module
    model.to(DEVICE)
    model.eval()

    # Load depth maps
    depth_maps = sorted(Path(depth_map_folder).glob("*.png"))
    if not depth_maps:
        print("No depth maps found in the specified directory.")
        return

    # Set up video capture and writer
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process video frames
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        cap.release()
        output_video.release()
        return

    prev_tensor = load_frame(prev_frame)
    for i in range(len(depth_maps)):
        ret, frame = cap.read()
        if not ret:
            break

        curr_tensor = load_frame(frame)

        # Load and check depth map
        depth_map_path = str(depth_maps[i])
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
        if depth_map is None:
            print(f"Warning: Could not load depth map at {depth_map_path}. Skipping frame.")
            continue  # Skip processing if depth map is missing or cannot be loaded

        # Resize depth map
        depth_map_resized = cv2.resize(depth_map, (frame_width, frame_height))

        # Calculate optical flow using RAFT
        flow = compute_optical_flow(model, prev_tensor, curr_tensor)
        flow_magnitude = np.linalg.norm(flow, axis=2)

        # Apply depth-weighted brightening effect
        brightness_overlay = apply_brightening_effect(flow_magnitude, depth_map_resized)
        
        # Blend overlay with original frame
        brightened_frame = cv2.addWeighted(frame, 0.6, brightness_overlay, 0.4, 0)

        # Write processed frame to output
        output_video.write(brightened_frame)

        # Update previous frame tensor
        prev_tensor = curr_tensor

    # Release resources
    cap.release()
    output_video.release()
    print(f"Processing completed. Check {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Depth-Guided Optical Flow Brightening with RAFT")
    parser.add_argument('--video_path', type=str, required=True, help="Path to input video file")
    parser.add_argument('--depth_map_folder', type=str, required=True, help="Path to folder containing depth maps")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output video")
    parser.add_argument('--model_weights', type=str, default='models/raft-things.pth', help="Path to RAFT model weights")
    
    args = parser.parse_args()

    # Pass parsed arguments to run function
    run(args.video_path, args.depth_map_folder, args.output_path, args.model_weights)

