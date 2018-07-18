from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import numpy as np

import keyboard
import cv2

import pandas

import os

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome(chrome_options=chrome_options)

driver.get("chrome://dino")
elem = driver.find_element_by_id("t")
space = 0
variable = 0

batch = 1

image = []
action = []

try:
    # Create target Directory
    os.mkdir(str(batch))
except :
    print "Folder Exists"



while True:
    if( elem.get_attribute("class") == "offline" ):continue
    if(driver.execute_script("return Runner.instance_.playing;") == False ):continue
    
    driver.save_screenshot("{}/screenshot-{}.png".format(batch,variable))
    img = cv2.imread("{}/screenshot-{}.png".format(batch , variable))

    img = img[ 168 : 400 , 105: ]

    ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    if(keyboard.is_pressed("space")):space = 1
    if(keyboard.is_pressed("esc")):break

    print variable

    cv2.imwrite("{}/train-{}.png".format(batch,variable), thresh_img)
    
    image.append("{}/train-{}.png".format(batch,variable))
    action.append(space)
    space = 0

    variable+=1

df = pandas.DataFrame(data={"image": image, "action": action})
df.to_csv("1/out.csv", sep=',',index=False)

driver.close()