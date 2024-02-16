import ffmpeg
import os
from ultralytics import YOLO
import ultralytics
import torch


# def compress_video(input_file, output_file, crf=23):
#     input_file = os.path.join(os.path.dirname(__file__), input_file)
#     ffmpeg.input(input_file).output(output_file, crf=crf).run()

if __name__ == '__main__':
    input_file = r'DJI_0027.MOV'
    output_file = 'out.mp4'

    # compress_video(input_file, output_file)
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)

    # Load a model
    model = YOLO(r'.\best.pt').to(device)

    # Use the model
    model.train(data=".\KoppersStacks.v1i.yolov8\data.yaml", epochs=50)  # train the model
    