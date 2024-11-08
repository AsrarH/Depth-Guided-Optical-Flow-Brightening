# Depth-Guided-Optical-Flow-Brightening


This project implements two versions of a depth-weighted optical flow visualization pipeline, which highlights high-motion areas in a video based on their proximity to the camera. The closer and faster the movement, the brighter it appears in the output video.

## Objective

The main goal is to:
1. Estimate optical flow between consecutive frames in a video.
2. Use depth information to adjust the brightness of high-flow regions, making closer motion appear brighter.
3. Export the modified video with depth-guided brightening effects.

## Project Structure

This project includes:
1. **Non-ML (Traditional) Optical Flow** using OpenCV's Farneback method.
2. **ML-based Optical Flow** using RAFT (Recurrent All-Pairs Field Transforms), a deep learning-based method for high-quality flow estimation.

## Installation

### Dependencies

Install the required libraries:
```bash
pip install opencv-python-headless numpy torch torchvision
```

For the ML-based approach, clone the [RAFT repository](https://github.com/princeton-vl/RAFT) and place it in the project directory. Ensure you have the RAFT model weights (`raft-things.pth`), which can be downloaded from the official RAFT GitHub page.

## Setup

### Project Files
1. **input_video.mp4**: The video file to process.
2. **source_depth/**: A directory containing per-frame depth maps aligned with the video frames.
3. **models/raft-things.pth**: The pretrained RAFT model weights (only needed for the ML-based approach).

### Usage

Both methods share the same structure for input/output paths. Use the following commands to run each version.

### Non-ML Approach (Farneback Optical Flow)

```bash
python traditional_optical_flow.py --video_path input_video.mp4 --depth_map_folder source_depth/ --output_path output_video_traditional.mp4
```

### ML-Based Approach (RAFT Optical Flow)

```bash
python ml_optical_flow.py --video_path input_video.mp4 --depth_map_folder source_depth/ --output_path output_video_raft.mp4 --model_weights models/raft-things.pth
```

## Processing Pipeline

1. **Optical Flow Calculation**:
   - **Non-ML Approach**: Uses the Farneback method from OpenCV to estimate optical flow.
   - **ML-Based Approach**: Uses RAFT, a deep learning-based model, for dense optical flow estimation, providing higher accuracy, especially for complex motions.
   
2. **Depth Map Integration**:
   - Depth maps are resized to match the video frame dimensions and normalized. This data is used to weight the flow magnitude, so closer regions appear brighter.

3. **Dynamic Thresholding**:
   - A dynamic threshold is applied to the flow magnitude, using the 90th percentile to focus on high-motion areas.

4. **Color Mapping and Blending**:
   - The processed motion data is color-mapped (`COLORMAP_HOT`) and blended with the original frame to create a brightening effect for high-motion regions.

## Output

Each script saves the processed video as:
- `output_video_traditional.mp4` (Non-ML)
- `output_video_raft.mp4` (ML-based)

## Performance and Comparison

| Method        | Pros                             | Cons                                      |
|---------------|----------------------------------|-------------------------------------------|
| **Non-ML**    | Fast and easy to implement       | Less accurate for complex scenes          |
| **ML-Based**  | Higher accuracy, better detail   | Slower, requires a GPU for efficient use  |

The ML-based approach with RAFT provides higher-quality flow estimations, especially in scenes with complex motion or occlusions. However, it is computationally intensive and may be slower on large videos without GPU support.

## Assumptions and Limitations

- The input video frames and depth maps must be aligned, with each depth map corresponding to a specific video frame.
- Depth maps are assumed to accurately represent proximity (closer areas have lower values).
- Flickering can occur due to frame-to-frame variations in optical flow, especially in the non-ML approach.


## Example Output

Example output files:
- Non-ML: `output_video_traditional.mp4`
- ML-Based: `output_video_raft.mp4`
