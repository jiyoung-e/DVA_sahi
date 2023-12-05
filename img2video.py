import cv2
import re
import os
import numpy as np

path = '/home/dva3/workspace/DVA_sahi/ByteTrack/YOLOX_outputs/dva_1/track_vis/2023_12_03_14_58_25'

# 파일 가져오기
def get_files(path):
    for root, subdirs, files in os.walk(path):
       
        list_files = []

        if len(files) > 0:
            for f in files:
                fullpath = root + '/' + f
                list_files.append(fullpath)

    return list_files


image_files = get_files(path)
print(image_files)
image_files = sorted(image_files)  # Sort the list of image files

pathOut = "/home/dva3/workspace/viz_video/demo_1203_dolphin.mp4"
fps = 30

frame_array = []
print(image_files)
for idx in range(len(os.listdir(path))) : 
    img = cv2.imread(image_files[idx])
    height, width, layers = img.shape
    size = (width,height)
    frame_array.append(img)
    
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()