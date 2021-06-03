
import cv2
import numpy as np

def getTransformedPoints(boxes, perspective_transform):
    
    bottom_points = []
    
    for box in boxes:
        points = np.array([[[int(box[0]+ (box[2]*0.5)), int(box[1]+box[3])]]], dtype="float32")
        boundary_points = cv2.perspectiveTransform(points, perspective_transform)[0][0]
        
        points = [int(boundary_points[0]), int(boundary_points[1])]
        bottom_points.append(points)
        
    return bottom_points

    
def cal_dis(p1, p2, distance_w, distance_h):
    
    h = abs(p2[1]-p1[1])
    w = abs(p2[0]-p1[0])
    
    #rescale height and width
    dis_w = float((w/distance_w)*180)
    dis_h = float((h/distance_h)*180)
    
    measured_dist = int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))
    
    return measured_dist
    
def get_distances(boxes1, bottom_points, distance_w, distance_h):
    
    distance_matrix = []
    bxs = []
    
    for i in range(len(bottom_points)):
        for j in range(len(bottom_points)):
            if i != j:
                dist = cal_dis(bottom_points[i], bottom_points[j], distance_w, distance_h)
                
                if dist <= 150: #might change to 1m
                    closeness = 0
                    distance_matrix.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                elif dist > 150 and dist <=180: #might change to 1m
                    closeness = 1
                    distance_matrix.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])       
                else:
                    closeness = 2
                    distance_matrix.append([bottom_points[i], bottom_points[j], closeness])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                
    return distance_matrix, bxs
    
def get_scale(W, H):
    
    dis_w = 400
    dis_h = 600
    
    return float(dis_w/W),float(dis_h/H)

def get_count(distance_matrix):

    r = []
    g = []
    y = []
    
    for i in range(len(distance_matrix)):

        if distance_matrix[i][2] == 0:
            if (distance_matrix[i][0] not in r) and (distance_matrix[i][0] not in g) and (distance_matrix[i][0] not in y):
                r.append(distance_matrix[i][0])
            if (distance_matrix[i][1] not in r) and (distance_matrix[i][1] not in g) and (distance_matrix[i][1] not in y):
                r.append(distance_matrix[i][1])
                
    for i in range(len(distance_matrix)):

        if distance_matrix[i][2] == 1:
            if (distance_matrix[i][0] not in r) and (distance_matrix[i][0] not in g) and (distance_matrix[i][0] not in y):
                y.append(distance_matrix[i][0])
            if (distance_matrix[i][1] not in r) and (distance_matrix[i][1] not in g) and (distance_matrix[i][1] not in y):
                y.append(distance_matrix[i][1])
        
    for i in range(len(distance_matrix)):
    
        if distance_matrix[i][2] == 2:
            if (distance_matrix[i][0] not in r) and (distance_matrix[i][0] not in g) and (distance_matrix[i][0] not in y):
                g.append(distance_matrix[i][0])
            if (distance_matrix[i][1] not in r) and (distance_matrix[i][1] not in g) and (distance_matrix[i][1] not in y):
                g.append(distance_matrix[i][1])
   
    return (len(r),len(y),len(g))    