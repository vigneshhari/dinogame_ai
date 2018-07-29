import csv
from PIL import Image
import numpy as np
import PIL.ImageOps    
import cv2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def mouse_callback(event, x, y, flags, params):
    global i
    print y , x , i[y,x]



files = []
action = []

cactusBase = 158
MIN_LENGTH = 80

with open("1/out.csv" , "r") as f :
    data = csv.reader(f, delimiter=',')
    for i in data:
        if(i[0] == 1):
            action.append(0)
        else:
            action.append(1)
        files.append(i[1])
        

def parse_function(filename,basewidth = 0):
    img = img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    #img = PIL.ImageOps.invert(img)
    #wpercent = (basewidth/float(img.size[0]))
    #hsize = int((float(img.size[1])*float(wpercent)))
    #img = img.resize((basewidth,hsize), Image.ANTIALIAS)    
    return img

def length_to_cactus(imgline):
    startpos = 1000
    i = 0
    black = False
    inter = True
    while i < len(imgline):
        if(imgline[i] == 0 and black == True):
            if(i - startpos < MIN_LENGTH):black = False            
            else:return str(startpos) + "   " +str( i - startpos)
        if(imgline[i] == 0 and black == False):
            if(inter):
                startpos = i
            inter = False
            black=True
        i+=1 
    return str(startpos) +"   " + "1000"

x_v = [np.array(parse_function(i))  for i in files ] 
y_v = action 
font = cv2.FONT_HERSHEY_SIMPLEX

for i in x_v:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1920, 500)
    cv2.putText(i, str(length_to_cactus(i[cactusBase] )) , (5, 15), cv2.FONT_HERSHEY_PLAIN , 0.9, (155, 155, 155))     
    cv2.imshow("image", i)
    cv2.setMouseCallback('image', mouse_callback)
    cv2.waitKey(0)