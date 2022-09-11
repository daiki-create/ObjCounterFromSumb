import os
import glob
from pathlib import Path

import config
from utils import util

from PIL import Image
import cv2
import pandas as pd
import csv
import numpy as np

import torch
import torchvision

import pandas as pd

VIDEO_DIR = "./video"
CSV_DIR = "./OUTPUT/csv"
IMAGE_DIR = "./OUTPUT/image"

VIDEO_LIST = glob.glob(f"{VIDEO_DIR}/*.mp4")
LABELS = config.LABELS

for idx, video in enumerate(VIDEO_LIST):
    print(video)

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    if frame.shape[2] == 3:  # カラー画像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif frame.shape[2] == 4:  # 透過カラー画像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)

    pillow_img = Image.fromarray(frame)
    image_tensor = torchvision.transforms.functional.to_tensor(frame)

    model = torch.load(Path("WEIGHT", "model.pt"))
    model = model.eval()
    output = model([image_tensor])[0]

    obj_cnt = len(output["labels"])

    f = open(f'{CSV_DIR}/{os.path.dirname(video)}.csv', 'a')
    data = [os.path.split(video)[-1], obj_cnt]
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()

    result_image = np.array(pillow_img.copy())
    for box, label, score in zip(
        output["boxes"], output["labels"], output["scores"]
    ):
        if LABELS[label] == "car" or LABELS[label] =="truck":
            color = util.make_color(LABELS)
            line = util.make_line(result_image)
            c1 = (int(box[0]), int(box[1]))
            c2 = (int(box[2]), int(box[3]))
            cv2 = util.draw_lines(c1, c2, result_image, line, color)
            cv2 = util.draw_texts(result_image, line, c1, cv2, color, LABELS, label)

    cv2.imwrite(
        # f"IMAGE_DIR/{os.path.split(video)[-2]}", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        f"OUTPUT/image/{os.path.split(video)[-1]}.png", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    )
