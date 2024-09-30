import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from skimage.metrics import structural_similarity


PADDING_H = 10
PADDING_W = 50

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

def save_cropped_image(cropped_img, prev_img, crop_idx, crop_dir):
    """Saves the cropped image if it meets the comparison criteria."""
    if prev_img is None or compare_images(prev_img, cropped_img) < 0.25:
        cv2.imwrite(f"{crop_dir}/stack_{crop_idx:04d}.png", cropped_img)
        crop_idx += 1
        return cropped_img, crop_idx  # Return unchanged values if no saving occurs
    return prev_img, crop_idx  # Return unchanged values if no saving occurs


def check_red_boxes(red_boxes, frame, original_frame, largest_box, max_area, prev_img, prev_img_red, crop_idx, crop_dir):
    frame_height, frame_width, _ = frame.shape
    # Calculate the center of the frame
    center_x_min = frame_width * 0.125  # 1/8 from the left
    center_x_max = frame_width * 0.875  # 1/8 from the right

    # Define size similarity threshold (e.g., 80% of largest box area)
    size_threshold = 0.8 * max_area
        
    # Get coordinates of the largest green box
    x1_g, y1_g, x2_g, y2_g = largest_box[:4]

    for red_box in red_boxes:
        x1_r, y1_r, x2_r, y2_r = red_box[:4]
        area_r = (x2_r - x1_r) * (y2_r - y1_r)

        # Check if the red box area is similar to the largest box area
        if area_r >= size_threshold:
            # Calculate the center of the red box
            red_box_center_x = (x1_r + x2_r) / 2

            # Check if the red box is within the central width of the video frame
            if center_x_min <= red_box_center_x <= center_x_max:

                if not (x1_r >= x2_g or x2_r <= x1_g or y1_r >= y2_g or y2_r <= y1_g):
                    # Discard this red box if it intersects with the green box
                    continue

                # Crop the red box image
                x1_pad_r = max(int(x1_r) - PADDING_W, 0)
                y1_pad_r = max(int(y1_r) - PADDING_H, 0)
                x2_pad_r = min(int(x2_r) + PADDING_W, frame_width)
                y2_pad_r = min(int(y2_r) + PADDING_H, frame_height)

                cropped_red_img = original_frame[y1_pad_r:y2_pad_r, x1_pad_r:x2_pad_r]

                # Save or process the cropped red box image as needed
                if prev_img_red is None:
                    prev_img_red, crop_idx = save_cropped_image(cropped_red_img, prev_img, crop_idx, crop_dir)
                else:
                    prev_img_red, crop_idx = save_cropped_image(cropped_red_img, prev_img_red, crop_idx, crop_dir)

                cv2.rectangle(frame, (int(x1_r), int(y1_r)), (int(x2_r), int(y2_r)), (255, 0, 0), 2)  # Blue color
    return prev_img, crop_idx, prev_img_red



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = r'C:\Users\Justin\MDP\runs\detect\stack_identifier_v2\weights\best.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLO(model_path)
    model = model.to(device)

    start_time_min = 0
    start_time_sec =50
    start_time_ms = (start_time_min * 60 + start_time_sec) * 1000  # Convert to milliseconds

    video_path = r'C:\Users\Justin\MDP\stacks\IMG_4791.mp4'
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    video.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)


    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_name = 'version2_example.mp4'
    out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

    crop_dir = "cropped_images_new"
    os.makedirs(crop_dir, exist_ok=True)
    crop_idx = 0   

    prev_img = None
    prev_img_red = None
    while True:
        ret, frame = video.read()

        if not ret:
            break

        original_frame = frame.copy()
        frame_height, frame_width, _ = frame.shape

        results = model(frame)
        stack_count = len(results[0].boxes)

        if stack_count > 0:
            boxes = results[0].boxes.xyxy
            max_area = 0
            largest_box = None
            red_boxes = []
            drawn_box_cnt = 0
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                area = (x2 - x1) * (y2 - y1)

                # Check if the box touches the left or right border of the frame
                if x1 <= 5 or x2 >= frame_width - 5:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2) # yellow
                    drawn_box_cnt += 1
                    continue  # Ignore boxes that touch the left or right borders

                if area > max_area:
                    max_area = area
                    largest_box_coords = (x1, y1, x2, y2)
                    largest_box = box
                else:
                    # If the box is not the largest and is in-frame, add it to the red boxes list
                    red_boxes.append(box)

                    # Draw a rectangle around the red boxes
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    drawn_box_cnt += 1

            if largest_box is not None:
                x1, y1, x2, y2 = largest_box_coords
                green_color  = (0, 255, 0)
                yellow_color = (0, 255, 255)
                # if largest box clips out of frame horizontally, mark it as yellow
                if x1 <= 0 or x2 >= frame_width - 1:
                    cv2.rectangle(frame, (int(largest_box[0]), int(largest_box[1])), 
                                    (int(largest_box[2]), int(largest_box[3])), 
                                    yellow_color, 3)
                    drawn_box_cnt += 1

                else:
                    # Otherwise, mark the largest box as green and crop it
                    x1_pad = max(int(x1) - PADDING_W, 0)
                    y1_pad = max(int(y1) - PADDING_H, 0)
                    x2_pad = min(int(x2) + PADDING_W, frame_width)
                    y2_pad = min(int(y2) + PADDING_H, frame_height)

                    cropped_img = original_frame[y1_pad:y2_pad, x1_pad:x2_pad]

                    prev_img, crop_idx = save_cropped_image(cropped_img, prev_img, crop_idx, crop_dir)

                    cv2.rectangle(frame, (int(largest_box[0]), int(largest_box[1])), 
                                        (int(largest_box[2]), int(largest_box[3])), 
                                        green_color, 3)
                    drawn_box_cnt += 1
                    
                    prev_img, crop_idx, prev_img_red = check_red_boxes(red_boxes, frame, original_frame, largest_box, max_area, prev_img, prev_img_red, crop_idx, crop_dir)

                
        cv2.putText(frame, f"Count: {drawn_box_cnt}", (10, 150), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 0), 5)


        out.write(frame)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video.release()
    out.release()
    cv2.destroyAllWindows()

