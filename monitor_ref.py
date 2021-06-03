# python monitor_ref.py --input videos/pedestrians.mp4 --output output/pedestrians.avi
# python monitor_ref.py -i test_footage/moving_horizontal.mp4 -o output/moving_horizontal.avi
# python monitor_ref.py --input test_footage/moving_vertical.mp4 --output output/moving_vertical.avi

# NOTES:
# camera height = 204cm 
# camera distance to subjects = 310cm 

from setups import config
#from setups.detect import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import math #added
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

#enables GPU for faster video processing
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
    
    
    #captures video for video processing
    (grabbed, frame) = vs.read()
    
    if not grabbed:
        break
    #resize window width
    frame = imutils.resize(frame, width = 700)
    #results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    
    #starting from here
    #process video frames for individual detection
    (H,W) = frame.shape[:2]
    results = []
    personIDx = LABELS.index("person")
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    classID = []
    centroids = []
    confidences = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if classID == personIDx and confidence > config.MIN_CONF:
            
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                
    IDxs = cv2.dnn.NMSBoxes(boxes, confidences, config.MIN_CONF, config.NMS_THRESH)
    
    if len(IDxs) > 0:
        #declare variables for violations
        red_v = set()
        green_v = set()
        yellow_v = set()

        for i in IDxs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, w, h), centroids[i])
            results.append(r)

            #coordinates of referenced individual/s
            i_centreX = boxes[i][0] + (boxes[i][2] // 2)
            i_centreY = boxes[i][1] + (boxes[i][3] // 2)
            
            IDxs_copy = list(IDxs.flatten())
            IDxs_copy.remove(i)
            
            for j in np.array(IDxs_copy):
                #coordinates of other individual/s
                j_centreX = boxes[j][0] + (boxes[j][2] // 2)
                j_centreY = boxes[j][1] + (boxes[j][3] // 2)
                
                #calculating distance using Euclidean method
                distance_measured = math.sqrt((j_centreX  - i_centreX)**2 + (j_centreY - i_centreY)**2)
                
                startPoint = (boxes[i][0] + (boxes[i][2] // 2), boxes[i][1]  + (boxes[i][3] // 2))
                endPoint = (boxes[j][0] + (boxes[j][2] // 2), boxes[j][1] + (boxes[j][3] // 2))
                
                               
                if distance_measured <= 180:
                    #find away to put distance on screen
                    text = "test"#'{}'.format(int(startPoint - endPoint)
                    cv2.putText(frame,text, (int((j_centreX+i_centreX)/2), int((j_centreY+i_centreY)/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,  white, 1)
                    
                    if distance_measured < 100: # less than a metre
                        cv2.line(frame, startPoint, endPoint, color, 1)
                        red_v.add(i)
                        red_v.add(j)
                        
                    elif 100 <= distance_measured <= 180: #about 1m to 6 feet
                        cv2.line(frame, startPoint, endPoint, color, 1)
                        yellow_v.add(i)
                        yellow_v.add(j)

                    elif 180 < distance_measured: #2m metres and above
                        cv2.line(frame, startPoint, endPoint, color, 1)
                        green_v.add(i)
                        green_v.add(j) 
        
             #loops through appended results to generate bounding box and centroid
            for (i, (prob, bbox, centroid)) in enumerate(results):
                cX, cY = centroid
                X, Y, W, H = bbox
                color = green
                
                #updates colour according to detection above
                if i in red_v:
                    color = red
                elif i in yellow_v:
                    color = yellow                    
                elif i in green_v:
                    color = green
                    
                
                #draw bounding box and centroid for individuals detected    
                cv2.rectangle(frame, (X, Y), (X+W, Y+H), color, 2)
                cv2.circle(frame, (cX,cY), 5, color, 1)        
                        
            
    
    
    #results placed in bottom display
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
    
    #prints data in command prompt window
    # print("Distance",distance_measured)


    
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
    
        
        