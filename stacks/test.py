import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
from skimage.metrics import structural_similarity


def compare_images(im1, im2):
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    height = min(gray1.shape[0], gray2.shape[0])
    width = min(gray1.shape[1], gray2.shape[1])
    gray1 = cv2.resize(gray1, (width, height))
    gray2 = cv2.resize(gray2, (width, height))

    (score, __) = structural_similarity(gray1, gray2, full=True)
    cv2.imwrite('debug_prev_img.png', gray1)
    cv2.imwrite('debug_curr_img.png', gray2)
    return score
    
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(r'best2.pt')
    model = model.to(device)

    start_time_min = 3
    start_time_sec = 50
    start_time_ms = (start_time_min * 60 + start_time_sec) * 1000  # Convert to milliseconds

    video = cv2.VideoCapture('DJI_0027.MOV')
    video.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)


    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('new_out.mp4', fourcc, fps, (width, height))

    crop_dir = "cropped_images"
    os.makedirs(crop_dir, exist_ok=True)
    crop_idx = 0   
    PADDING_H = 10
    PADDING_W = 50

    prev_img = None
    while True:
        ret, frame = video.read()

        if not ret:
            break

        results = model(frame)
        stack_count = len(results[0].boxes)

        if stack_count > 0:
            boxes = results[0].boxes.xyxy
            max_area = 0
            largest_box = None
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_box_coords = (x1, y1, x2, y2)
                    largest_box = box
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            if largest_box is not None:
                x1, y1, x2, y2 = largest_box_coords
                green_color  = (0, 255, 0)
                yellow_color = (0, 255, 255)
                frame_height, frame_width, _ = frame.shape
                # if largest box clips out of frame, mark it as yellow
                if x1 <= 0 or y1 <= 0 or x2 >= frame_width-1 or y2 >= frame_height-1:
                    cv2.rectangle(frame, (int(largest_box[0]), int(largest_box[1])), 
                                        (int(largest_box[2]), int(largest_box[3])), 
                                        yellow_color, 3)
                # otherwise, mark it as green
                else:
                    x1_pad = max(int(x1) - PADDING_W, 0)
                    y1_pad = max(int(y1) - PADDING_H, 0)
                    x2_pad = min(int(x2) + PADDING_W, frame_width)
                    y2_pad = min(int(y2) + PADDING_H, frame_height)

                    cropped_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]


                    if prev_img is None or compare_images(prev_img, cropped_img) < 0.25:
                        print('hi')
                        cv2.imwrite(f"{crop_dir}/stack_{crop_idx:04d}.png", cropped_img)
                        crop_idx += 1
                        prev_img = cropped_img

                    cv2.rectangle(frame, (int(largest_box[0]), int(largest_box[1])), 
                                        (int(largest_box[2]), int(largest_box[3])), 
                                        green_color, 3)
                
        cv2.putText(frame, f"Count: {stack_count}", (10, 150), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 0), 5)


        out.write(frame)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video.release()
    out.release()
    cv2.destroyAllWindows()

