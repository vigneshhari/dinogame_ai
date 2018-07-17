from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import numpy as np

from PIL import Image

import cv2

driver = webdriver.Chrome()
driver.get("chrome://dino")
elem = driver.find_element_by_id("t")
    
variable = 0
while True:
    driver.save_screenshot("screens/screenshot-{}.png".format(variable))
    img = cv2.imread("screens/screenshot-{}.png".format(variable))
    
    img = img[ 169 : 400 , 90: ]

    ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)



    print thresh_img[119,291] 

    cv2.imwrite("screens/screenshot-{}-converted.png".format(variable), thresh_img)

    if sum(thresh_img[119,291]) == 0 :
        elem.send_keys(Keys.SPACE)
        print "Sent Key"
    variable+=1
driver.close()