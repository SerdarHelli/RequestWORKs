# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:19:00 2021

@author: serdarhelli
"""


import cv2 
import numpy as np




#recog(PATH IMAGEs)


def check_blue(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(img,lower_blue,upper_blue)
    mask=mask/255
    return np.sum(mask)

def check_red(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,60,50]) 
    upper_red = np.array([10,255,255])
    mask = cv2.inRange(img,lower_red,upper_red)
    mask=mask/255
    return np.sum(mask)

    
    
       
    
def recog(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(256,256))
    if check_blue(img)>0:
        if check_blue(img[35:120,145:200,:])>((85*55)-check_blue(img[35:120,145:200,:])): #corner
            print("Forward")
        else:
            print("Right")
    if check_red(img)>0:
        if check_red(img)>((256*256)-check_red(img)):
            print("No Entry")
        else:
            print("Speed Limit 10")
    if check_red(img)==0 and check_blue(img)==0:
         print("Special Sign")

