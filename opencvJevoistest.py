import numpy as np
import cv2
import copy


cap = cv2.VideoCapture(0);

lowerBound = np.array([0, 250, 0])
upperBound = np.array([255, 255, 255])

lower_orange = np.array([10, 70, 70])
upper_orange = np.array([50, 100, 100])

#cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH, imSize[0])
#cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, imSize[1])

while(True):
    ret, frame = cap.read()

    mask = cv2.inRange(frame, lowerBound, upperBound)
    maskRGB = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
   
    frame &= maskRGB

    edges = cv2.Canny(frame, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)

        if ((len(approx) < 6) & (area > 20) ):
            contour_list.append(contour)
            
    cv2.drawContours(frame, contour_list,  -1, (255,0,0), 2)
    
    cv2.imshow("test", frame)
    cv2.imshow("edges", edges)
            
 #   print('count %d' % count)
    
 #   print('[%d , %d , %d] , \n[%d,  %d,  %d]\n\n' % (int(lowerBound[0]), int(lowerBound[1]), int(lowerBound[2]),
  #                                         int(upperBound[0]), int(upperBound[1]), int(upperBound[2])))
    if(cv2.waitKey(1) == 27):
        break

cap.release()
cv2.destroyAllWindows()

#burn comment
'''
 cFrame = copy.deepcopy(frame)

    bilateral_filtered_image = cv2.bilateralFilter(cFrame, 5, 175, 175)
    edges = cv2.Canny(cFrame, 100, 200)



    gray = cv2.cvtColor(cFrame, cv2.COLOR_BGR2GRAY)
    ''''''
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100);

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for(x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow("output", np.hstack([cFrame, circles])
    
    #contours method
    ''''''
    contours, x = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)

        if ((len(approx) > 8) & (area > 80) ):
            contour_list.append(contour)
            
    cv2.drawContours(cFrame, contour_list,  -1, (255,0,0), 2)
    cv2.imshow('Objects Detected',cFrame)
    '''
