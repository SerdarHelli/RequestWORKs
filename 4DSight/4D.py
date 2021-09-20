# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 20:15:25 2021
@author: serdarhelli
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob


vr3d=np.load("./vr3d.npy")
vr2d=np.load("./vr2d.npy")

img=plt.imread('./img1.png')
img=np.uint8(img*255)
vr3d=np.reshape(vr3d,(len(vr3d),3))
objpoints=[vr3d]
imgpoints=[vr2d]

cmx=np.asarray([[100,0,960],[0,100,540],[0,0,1]])
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints,imgpoints, img[:,:,0].shape, cmx, None,flags=cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_FIX_PRINCIPAL_POINT)



##References
#https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html
#http://cs.brown.edu/courses/cs143/proj5/
#https://docs.opencv.org/master/d7/d53/tutorial_py_pose.html

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw_3D(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

axis_2d= np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis_3d = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])



    
for fname in glob.glob('./img*.png'):
    img = plt.imread(fname)
    img=np.uint8(img*255)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    corners2 = cv.cornerSubPix(gray,vr2d,(11,11),(-1,-1),criteria)
    # Find the rotation and translation vectors.
    ret,rvecs, tvecs = cv.solvePnP(vr3d, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis_2d, rvecs, tvecs, mtx, None)
    img = draw(img,corners2,imgpts)
    show_image = cv.resize(img, (img[:,:,0].shape[1]//2, img[:,:,0].shape[0]//2))
    cv.imshow('img',show_image)
    k = cv.waitKey(0) & 0xFF
    if k == ord('s'):
        cv.imwrite(fname[:6]+'.png', img)
cv.destroyAllWindows()








