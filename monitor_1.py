# python monitor.py --input videos/pedestrians.mp4 --output output/pedestrians.avi
# python monitor.py -i test_footage/moving_horizontal.mp4 -o output/moving_horizontal.avi
# python monitor.py --input test_footage/moving_vertical.mp4 --output output/moving_vertical.avi

# NOTES:
# camera height = 204cm 
# camera distance to subjects = 310cm 

# 12 MP, f/1.7, 27mm (wide), 1/2.55", 1.4µm, Dual Pixel PDAF, OIS
# 12 MP, f/2.4, 52mm (telephoto), 1/3.6", 1.0µm, PDAF, OIS, 2x optical zoom
# 12 MP, f/2.2, 12mm (ultrawide) (this one here)


from setups import config
from setups.setup import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.USE_GPU:
    print("[INFO] setting prefable backend and target to CUDA")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

print("[INFO] accessing video stream")
vs =cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

while True:
    red = (0,0,255)
    green = (0,255,0)
    yellow = (0,255,255)
    white = (200,200,200)
    
    (grabbed, frame) = vs.read()
    
    if not grabbed:
        break
    frame = imutils.resize(frame, width = 700)
    results = detect_people(frame, net, ln, personIDx = LABELS.index("person"))
    
    red_v = set()
    green_v = set()
    yellow_v = set()
    
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        Distance_measured = dist.cdist(centroids, centroids, metric = "euclidean")
        
        #catergorises individuals based on distance measured
        for i in range(0, Distance_measured.shape[1]):
            for j in range(i+1, Distance_measured.shape[1]):
                if Distance_measured[i,j] < 100: # less than a metre
                    red_v.add(i)
                    red_v.add(j)
                    
                elif 100 < Distance_measured[i,j] < 180: #about 1m to 6 feet
                    yellow_v.add(i)
                    yellow_v.add(j)

                else: #2m metres and above
                    green_v.add(i)
                    green_v.add(j)

            
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = green
        
        
        #issues when there is more than 2 people having varying distances from each other
        #colour unable to change accordingly
        #updates colour according to detection above
        if i in red_v:
            color = red
        elif i in yellow_v:
            color = yellow
        elif i in green_v:
            color = green    
            
        cv2.rectangle(frame, (startX, startY), (endX,endY), color, 2)
        cv2.circle(frame, (cX,cY), 5, color, 1)
    
    #results btm display
    pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    text2 = "Number of people detected: {}".format(len(results))
    high_text = "-- HIGH RISK : {}".format(len(red_v))
    low_text = "-- LOW RISK : {}".format(len(yellow_v))
    safe_text = "-- SAFE : {}".format(len(green_v))
    
    
    cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
    cv2.putText(pad, text2, (50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
    cv2.putText(pad,  high_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 1)
    cv2.putText(pad, low_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 1)
    cv2.putText(pad, safe_text, (50,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 1)
    frame = np.vstack((frame,pad))
    
    print("Distance",Distance_measured)
    # print("Red Violation: ",red_v)
    # print("Green Violation:",green_v)
    
    if args["display"] > 0:
        cv2.imshow("Result", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
    
    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 60, (frame.shape[1], frame.shape[0]), True)
    
    if writer is not None:
        writer.write(frame)
    
        
        