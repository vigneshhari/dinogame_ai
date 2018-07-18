
import numpy as np

from PIL import Image
import keyboard
import cv2

def mouse_callback(event, x, y, flags, params):
    global thresh_img
    print x , y , thresh_img[y,x]


img = cv2.imread('screens/screenshot-22.png',0)

img = img[ 168 : 400 , 105: ]

print img[119,315]


ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#cv2.imwrite('bw_image.png', thresh_img)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1920, 500)


cv2.setMouseCallback('image', mouse_callback)

    



cv2.imshow("image" , thresh_img)


cv2.waitKey(0)
