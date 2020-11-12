import cv2
import numpy as np


def getMirrorImage(orgin_img):
    gray_img = cv2.cvtColor(orgin_img, cv2.COLOR_BGR2GRAY)
    threshold = 5
    ret, thresh = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours == []:
        return np.array([])
    c1 = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(c1)
    if w*h < 100:
        return np.array([])
    # cv2.rectangle(orgin_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    crop_img = orgin_img[y:(y+h), x:(x + w)]


    #Visualization
    # cv2.drawContours(orgin_img, [c1], -1, (0, 0, 255), 3)
    # cv2.namedWindow("Image", flags=0)
    # cv2.resizeWindow("Image", 720, 480)
    # cv2.imshow('Image', crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return crop_img