# -*- coding: utf-8 -*-  

'''
editor: Yang WenJun 
email:  wenjuny@outlook.com
time:   2020/8/3

purpose:
            1. load our serialized model from disk
            2. grap a refernce to be webcam or video files
            3. inference 
            4. decode the outputs of model
            5. post-process for decode results
            6. draw hps & bbox in the frame
            7. show the results in the windows   

'''

# USAGE
# python multi_keypoint.py --xml /path/to/your/model/*.xml --bin /path/to/your/model/*.bin --input_video /path/to/your/video \
# --output_result_video /path/to/your/video/

# import the necessary packages
from detector import pre_process, multi_pose_decode, post_process, show_results
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import time
import cv2
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xml", 
    help="path to {model}.xml")
ap.add_argument("-b", "--bin", 
    help="path to {model}.bin") 
ap.add_argument("-i", "--input_video", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output result video file")
args = vars(ap.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNet(args["xml"], args["bin"])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

if not args.get("input_video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(usePiCamera=False).start()
    time.sleep(2.0)
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input_video"])

K = 100
scale = 1
fps = FPS().start()     
outNames = net.getUnconnectedOutLayersNames()

while True:
    frame = vs.read()
    frame = frame[1] if args.get("input_video", False) else frame
    frames, meta = pre_process(frame, scale, None)

    blob = cv2.dnn.blobFromImage(frames, size=(416, 416), mean=(104.04,113.985,119.85), scalefactor=1/255., swapRB=True)
    net.setInput(blob)
    output = net.forward(outNames)
    
    output_hm = torch.from_numpy(output[0]) ## hm:(1, 1, 128, 128)
    output_hm = output_hm.sigmoid_()
    output_wh = torch.from_numpy(output[1])  ## wh:(1, 2, 128, 128)
    output_hps = torch.from_numpy(output[2])  ## hps:(1, 34, 128, 128)
    output_reg = torch.from_numpy(output[3])  ## reg:(1, 2, 128, 128)
    output_hm_hp = torch.from_numpy(output[4]) ## hm_hp:(1, 17, 128, 128)
    output_hm_hp = output_hm_hp.sigmoid_()
    output_hp_offset = torch.from_numpy(output[5])  ## hp_offset:(1, 2, 128, 128)
    
    # decode the output of model
    dets = multi_pose_decode(output_hm, output_wh, output_hps, reg=output_reg, hm_hp=output_hm_hp, hp_offset=output_hp_offset, K=3)
    # post-process decode 
    dets = post_process(dets, meta, scale)
    # 转换output
    results = dets.tolist()
    # show results 
    framess = show_results(frame, results)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter    
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
    writer.release()
if not args.get("input_video", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()
