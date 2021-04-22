# python monitor.py --input videos/pedestrians.mp4
# python monitor_transform.py --input videos/pedestrians.mp4 --output output/test_transform.avi

from setups import config
from setups import calc
from setups import view
from setups.setup import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import argparse
import cv2
import os

mouse_pts = []
writer = None

def get_mouse_pts(event, x, y, flags, param):
    global mouse_pts
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts)<4:
            cv2.circle(image, (x,y), 5, (0,0,255), 10)
        else:
            cv2.circle(image, (x,y), 5, (255,0,0), 10)
    
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x,y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70,70,70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x,y), (mouse_pts[0][0], mouse_pts[0][1]), (70,70,70), 2)
                
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x,y))    


def calc_social_dist(vid_path, net, output_vid, ln):
    count = 0
    print("[INFO] accessing video stream")
    vs =cv2.VideoCapture(vid_path if args["input"] else 0)
    
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    
    scale_w, scale_h = calc.get_scale(width, height)  

    
    
    points = []
    global image
    
    while True:
        
        (grabbed, frame) = vs.read()
        frame = imutils.resize(frame, width = 700)
        if not grabbed:
            break
            
        (H,W) = frame.shape[:2]
        
        if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                key = cv2.waitKey(1) 
                if len(mouse_pts) == 8:
                #if key == ord("q"):
                    cv2.destroyWindow("image")
                    break
            points = mouse_pts

        src = np.float32(np.array(points[:4]))
        #dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        perspective_transform = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, perspective_transform, (width,height))
        warped = imutils.resize(warped, width = 700)

        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, perspective_transform)[0]
        
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

        #returns people detected
        results = detect_people(frame, net, ln, personIDx = LABELS.index("person"))
        
        #obtain btm points from prespective
        person_pts = calc.getTransformedPoints(results, perspective_transform)
        
        distances_mat, boxes_mat = calc.get_distances(results, person_pts, distance_w, distance_h)
        risk_count = calc.get_count(distances_mat)
        
        frame_copy = np.copy(frame)
        
        bird_view = view.bird_eye(frame_copy, distances_mat, person_pts, scale_w, scale_h, risk_count)
        img = view.social_distance(frame, boxes_mat, results, risk_count)
        
        if count !=0:
            cv2.imshow("bird view", bird_view)
            cv2.imshow("Warped", warped)
            cv2.imshow("social distance", img)
            
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_vid, fourcc, 25, (width, height), True)
    
    vs.release()
    cv2.destroyAllWindows() 
            
if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input",type=str, default="",
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

    #set mouse callback
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_pts)
    np.random.seed(42)
    
    calc_social_dist(args["input"],net,args["output"],ln)
    


    


    
        
        