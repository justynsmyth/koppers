# Video Processing with YOLO

This script processes a video file using a YOLO (You Only Look Once) model for object detection and can optionally generate a new video with the detected objects highlighted. It also saves frame times in a JSON file and allows cropping of images.

## Features

- Process video files using a YOLO model.
- Specify start time for processing.
- Optionally generate an output video.
- Save cropped images to a specified directory.
- Output frame times in JSON format.

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- YOLO model weights

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   pip install -r requirements.txt

   

## Usage

To use the script, run the following command in your terminal:

```
python your_script_name.py --model_path <path_to_model_weights> --vid_in <path_to_video_file> --start_min <start_time_minutes> --start_sec <start_time_seconds> --crop_dir <directory_for_cropped_images> --generate_video <True/False> --vid_out <output_video_file_name> --json_out <output_json_file_name>

For Example:
```
python .\primary_stack_tool.py --model_path ..\runs\detect\stack_identifier_v2\weights\stack_best.pt --vid_in .\IMG_4791.mp4 --start_min 1 --start_sec 30 --vid_out test.mp4 --crop_dir cropped_new


## Arguments
--model_path: Path to the YOLO model weights (required).
--vid_in: Path to the input video file (required).
--start_min: Start time in minutes (default: 0).
--start_sec: Start time in seconds (default: 0).
--crop_dir: Directory to save cropped images (default: 'cropped_images').
--generate_video: Flag to generate output video (default: True).
--vid_out: Output video file name (default: 'output_video.mp4').
--json_out: Output JSON file to save frame times (default: 'frame_times.json').