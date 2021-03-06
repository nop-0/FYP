from .config import NMS_THRESH
from .config import MIN_CONF
import numpy as np
import cv2

def detect_people(frame, net, ln, personIDx = 0):
    #obtain frame from video, net from YOLO model, YOLO CNN layer, index for YOLO detection
    (H,W) = frame.shape[:2]
    results = []
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    centroids = []
    confidences = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if classID == personIDx and confidence > MIN_CONF: #30% confidence
            
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                #obtain x-y coordinates of individuals
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                
    #apply Non-Max Suppression for overlapping bounding box
    IDxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    if len(IDxs) > 0:
        for i in IDxs.flatten():
            #combines x,y, width and height of the individuals 
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])
            
            #combines data
            r = (confidences[i], (x, y, x+w, y+h), centroids[i])
            results.append(r)
            
    return results
    
    
    
    
    
    

def detect_people2(frame, net, ln, personIDx = 0):
    #obtain frame from video, net from YOLO model, YOLO CNN layer, index for YOLO detection
    (H,W) = frame.shape[:2]
    results = []
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []

    centroids = []
    confidences = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if classID == personIDx and confidence > MIN_CONF:
            
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                
    IDxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    
    
    if len(IDxs) > 0:
        for i in IDxs.flatten():
            (x,y) = (boxes[i][0], boxes[i][1])
            (w,h) = (boxes[i][2], boxes[i][3])

            #r = (confidences[i], (x, y, x+w, y+h), centroids[i]) #orig
            r = (confidences[i], (x, y, w, h), centroids[i])
            results.append(r)
            
    return results