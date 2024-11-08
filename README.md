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
python traditional_optical_flow.py 
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
   - A dynamic threshold is applied to the depth-weighted flow magnitude, based on the 90th percentile of scene flow values. This step helps to focus the brightening effect on only the most significant motion, isolating high-flow regions in each frame while filtering out background motion or minor movements. Lower threshold values introduced noise, while higher thresholds reduced the visibility of the effect.

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

## Trade-offs, Assumptions, and Decisions
Choice of Optical Flow Method:

The Farneback method was chosen for its simplicity and speed in lower-complexity scenes, while RAFT was selected for its accuracy in complex motions. This allows the user to balance between speed and accuracy based on the scene’s needs.


Dynamic Brightness Thresholding:

The 90th percentile threshold was selected for its effectiveness in isolating high-motion regions. Lower thresholds introduced more noise, while higher thresholds reduced the effect's visibility.


Depth Map Assumptions:

The depth maps are assumed to align with the video frames and accurately represent object proximity (lower values for closer areas). This assumption is critical for the depth-weighted brightening effect to work effectively.


Camera Movement Compensation:

Implemented for both approaches to reduce the effect of global motion due to camera movement. This was crucial in achieving a consistent brightening effect focused on object motion.


Challenges and Limitations:

Flickering: 

Small frame-to-frame variations can introduce flickering, especially in the non-ML approach. This may require further smoothing or stabilization techniques.


GPU Dependency for ML-Based Approach: 

The RAFT model is resource-intensive and requires a GPU for practical processing speeds on high-resolution or long videos.

## Optional Extensions
The following optional extensions were explored and documented in the code:


Non-ML vs. ML-Based Comparison:

Implemented both non-ML (Farneback) and ML-based (RAFT) optical flow approaches to compare accuracy and performance. RAFT provides superior detail for complex motion but is slower, while Farneback is faster and effective for simpler scenes.
Beautification and Color Mapping:

Experimented with various color maps and blending settings (COLORMAP_HOT, etc.) to improve the look of the effect. COLORMAP_HOT was chosen for its vibrant effect on high-motion areas.
Future Extensions:

Scene Flow Estimation: Estimating 3D scene flow for better depth-weighted visualization.
Forward Edge Effect: Highlighting only the forward edges of motion for a more realistic effect.

## Example Output

Example output files:
-  Non-ML (Traditional): [output_video_traditional.mp4 on Google Drive](https://drive.google.com/file/d/1wBzC2qknJu-oIoUQP2D8KvhxsgXZtw7_/view?usp=sharing)
- ML-Based: `output_video_raft.mp4`

  
