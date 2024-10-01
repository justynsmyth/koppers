import cv2
import os
import torch
import json
import argparse
import numpy as np
from ultralytics import YOLO
from skimage.metrics import structural_similarity


PADDING_H = 10
PADDING_W = 50

class FrameData:
    def __init__(self, frame, original_frame):
        self.frame = frame
        self.original_frame = original_frame
        self.height, self.width, _ = frame.shape

class DetectionResult:
    def __init__(self, red_boxes, largest_box, max_area):
        self.red_boxes = red_boxes
        self.largest_box = largest_box
        self.max_area = max_area

class CropData:
    def __init__(self, crop_dir, prev_img=None, prev_img_red=None, crop_idx=0, frame_dict=None):
        self.crop_dir = crop_dir
        self.prev_img = prev_img
        self.prev_img_red = prev_img_red
        self.crop_idx = crop_idx
        self.frame_dict = frame_dict if frame_dict is not None else {}

def compare_images(im1, im2):
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    height = min(gray1.shape[0], gray2.shape[0])
    width = min(gray1.shape[1], gray2.shape[1])
    gray1 = cv2.resize(gray1, (width, height))
    gray2 = cv2.resize(gray2, (width, height))

    (score, __) = structural_similarity(gray1, gray2, full=True)

    # Save debug images
    if score < 0.25:
        cv2.imwrite('debug_prev_img.png', gray1)
        cv2.imwrite('debug_curr_img.png', gray2)

        # # Display debug images
        # cv2.imshow('Debug Previous Image', gray1)
        # cv2.imshow('Debug Current Image', gray2)

        # # Wait for a key press to continue
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return score

def save_cropped_image(cropped_img, prev_img, crop_idx, crop_dir, frame_time, frame_dict):
    """Saves the cropped image if it meets the comparison criteria."""
    if prev_img is None or compare_images(prev_img, cropped_img) < 0.25:
        image_name = f"stack_{crop_idx:04d}.png"
        cv2.imwrite(f"{crop_dir}/{image_name}", cropped_img)


        frame_dict[image_name] = frame_time
        crop_idx += 1
        return cropped_img, crop_idx
    return prev_img, crop_idx  # Return unchanged values if no saving occurs

def check_red_boxes(frame_data: FrameData, detection_result: DetectionResult, crop_data: CropData, frame_time):
    # Calculate the center of the frame
    center_x_min = frame_data.width * 0.125  # 1/8 from the left
    center_x_max = frame_data.width * 0.875  # 1/8 from the right
    # Define size similarity threshold (e.g., 80% of largest box area)
    size_threshold = 0.8 * detection_result.max_area

    # Get coordinates of the largest green box
    x1_g, y1_g, x2_g, y2_g = detection_result.largest_box[:4]

    for red_box in detection_result.red_boxes:
        x1_r, y1_r, x2_r, y2_r = red_box[:4]
        area_r = (x2_r - x1_r) * (y2_r - y1_r)
        
        # Check if the red box area is similar to the largest box area
        if area_r >= size_threshold:
            # Calculate the center of the red box
            red_box_center_x = (x1_r + x2_r) / 2
            # Check if the red box is within the central width of the video frame
            if center_x_min <= red_box_center_x <= center_x_max:
                # Discard this red box if it intersects with the green box
                if not (x1_r >= x2_g or x2_r <= x1_g or y1_r >= y2_g or y2_r <= y1_g):
                    continue
                # Crop the red box image
                x1_pad_r = max(int(x1_r) - PADDING_W, 0)
                y1_pad_r = max(int(y1_r) - PADDING_H, 0)
                x2_pad_r = min(int(x2_r) + PADDING_W, frame_data.width)
                y2_pad_r = min(int(y2_r) + PADDING_H, frame_data.height)

                cropped_red_img = frame_data.original_frame[y1_pad_r:y2_pad_r, x1_pad_r:x2_pad_r]
                cropped_red_img, crop_data.crop_idx = save_cropped_image(
                    cropped_red_img, 
                    crop_data.prev_img_red or crop_data.prev_img, # if prev_img_red is None, use prev_img
                    crop_data.crop_idx, 
                    crop_data.crop_dir, 
                    frame_time, 
                    crop_data.frame_dict
                )
                cv2.rectangle(frame_data.frame, (int(x1_r), int(y1_r)), (int(x2_r), int(y2_r)), (255, 0, 0), 2)

    return crop_data.prev_img, crop_data.crop_idx, crop_data.prev_img_red



def process_video_frame(frame, model, prev_img, prev_img_red, crop_idx, crop_dir, frame_dict, video):
    original_frame = frame.copy()
    frame_height, frame_width, _ = frame.shape

    results = model(frame)
    stack_count = len(results[0].boxes)
    if stack_count > 0:
        boxes = results[0].boxes.xyxy
        max_area = 0
        largest_box = None
        red_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            area = (x2 - x1) * (y2 - y1)

            if x1 <= 5 or x2 >= frame_width - 5:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                continue

            if area > max_area:
                max_area = area
                largest_box = box
            else:
                red_boxes.append(box)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        frame_time_ms = video.get(cv2.CAP_PROP_POS_MSEC) # time in milliseconds
        frame_time = frame_time_ms / 1000  # Convert milliseconds to seconds

        if largest_box is not None:
            x1, y1, x2, y2 = largest_box[:4]
            if x1 <= 0 or x2 >= frame_width - 1:
                cv2.rectangle(frame, (int(largest_box[0]), int(largest_box[1])), 
                                (int(largest_box[2]), int(largest_box[3])), 
                                (0, 255, 255), 3)
            else:
                x1_pad = max(int(x1) - PADDING_W, 0)
                y1_pad = max(int(y1) - PADDING_H, 0)
                x2_pad = min(int(x2) + PADDING_W, frame_width)
                y2_pad = min(int(y2) + PADDING_H, frame_height)

                cropped_img = original_frame[y1_pad:y2_pad, x1_pad:x2_pad]
                prev_img, crop_idx = save_cropped_image(cropped_img, prev_img, crop_idx, crop_dir, frame_time, frame_dict)
                cv2.rectangle(frame, (int(largest_box[0]), int(largest_box[1])), 
                                (int(largest_box[2]), int(largest_box[3])), 
                                (0, 255, 0), 3)
                
                frame_data = FrameData(frame, original_frame)
                detection_result = DetectionResult(red_boxes, largest_box, max_area)
                crop_data = CropData(crop_dir, prev_img=prev_img, prev_img_red=prev_img_red, crop_idx=crop_idx, frame_dict=frame_dict)
                prev_img, crop_idx, prev_img_red = check_red_boxes(frame_data, detection_result, crop_data, frame_time)

    return frame, prev_img, prev_img_red, crop_idx

def parse_args():
    parser = argparse.ArgumentParser(description="Video processing with YOLO model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the YOLO model weights")
    parser.add_argument('--vid_in', type=str, required=True, help="Path to the video file")
    parser.add_argument('--start_min', type=int, default=0, help="Start time in minutes")
    parser.add_argument('--start_sec', type=int, default=0, help="Start time in seconds")
    parser.add_argument('--crop_dir', type=str, default='cropped_images', help="Directory to save cropped images")
    parser.add_argument('--generate_video', type=bool, default=True, help="Flag to generate output video")
    parser.add_argument('--vid_out', type=str, default='output_video.mp4', help="Output video file name")
    parser.add_argument('--json_out', type=str, default='frame_times.json', help="Output JSON file to save frame times")
    return parser.parse_args()

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model_path).to(device)

    start_time_ms = (args.start_min * 60 + args.start_sec) * 1000  # Convert start time to milliseconds

    video = cv2.VideoCapture(args.vid_in)
    video.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

    width, height, fps = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                          int(video.get(cv2.CAP_PROP_FPS)))

    out = None
    if args.generate_video:
        out = cv2.VideoWriter(args.vid_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    os.makedirs(args.crop_dir, exist_ok=True)
    frame_dict = {}

    prev_img, prev_img_red, crop_idx = None, None, 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame, prev_img, prev_img_red, crop_idx = process_video_frame(
            frame, model, prev_img, prev_img_red, crop_idx, args.crop_dir, frame_dict, video
        )

        if args.generate_video:
            out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    with open(args.json_out, "w") as json_file:
        json.dump(frame_dict, json_file, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)