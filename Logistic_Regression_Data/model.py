import csv
from PIL import Image
import numpy as np
import PIL.ImageOps    
import cv2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import tensorflow as tf


def mouse_callback(event, x, y, flags, params):
    global i
    print y , x , i[y,x]



files = []
action = []

cactusBase = 156
MIN_LENGTH = 100

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
            else:return startpos ,( i - startpos)
        if(imgline[i] == 0 and black == False):
            if(inter):
                startpos = i
            inter = False
            black=True
        i+=1 
    return startpos  , 1000

x_v = [ length_to_cactus(np.array(parse_function(i))[cactusBase])  for i in files ] 
y_v = action 

print "Loaded all Images"

print "Starting Training NN ( Tensorflow ) " 

'''
Using a Neural Network with No Hidden Layers with 55*200 input layers. total of 55 * 200 Values.

The weight dimentions are (55 * 200) * 1 , input is 1 *  ( 55 * 200 )

Output is single neuron ( 1 - Neuron Fired -- Jump ) else ( Leave )

'''


# Parameters
#learning_rate = .00000000001
learning_rate =  .00000000001
training_epochs = 50000
batch_size = x_v.__len__()
display_step = 110

tf.set_random_seed(777)  
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None]) 

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.random_normal([1]))

pred = tf.nn.tanh(tf.add(tf.matmul(x, W) , b) ) 


#cost = tf.reduce_mean(-1 * (tf.reduce_sum(y*tf.log(pred) + tf.reduce_sum( (y-1)*tf.log(pred -1) ) )))
cost = tf.reduce_mean(-1 * (tf.reduce_sum(y*tf.log(pred   ))  ) ) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


correct_prediction = tf.equal(tf.cast(tf.argmax(pred, 1),"float64"),tf.cast(y , "float64") )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        epoch_loss = 0
        for i in range(int(len(files)//batch_size)):
            epoch_x, epoch_y = x_v[ (i * batch_size) : (i +1 ) * batch_size ] , y_v[ (i * batch_size) : (i +1 ) * batch_size ] 
            #print sess.run(max_pool1, feed_dict={x_inp: epoch_x, y: epoch_y}).shape
            _, c , w = sess.run([optimizer, cost, W], feed_dict={x: x_v,y: y_v})
            #print sum(w) , c
            epoch_loss += c

        #print sum(w)
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{}".format(epoch_loss))

    print(sess.run([accuracy , pred] ,feed_dict={x: x_v,y: y_v }))


    # Start Selenium Program

    import math

    driver = webdriver.Chrome()

    driver.get("chrome://dino")
    elem = driver.find_element_by_id("t")

    variable = 1
    while True:
        if( elem.get_attribute("class") == "offline" ):continue
        if(driver.execute_script("return Runner.instance_.playing;") == False ):
            elem.send_keys(Keys.SPACE)
            continue
        
        driver.save_screenshot("temp/screenshot-{}.png".format(variable))
        img = cv2.imread("temp/screenshot-{}.png".format( variable))

        img = img[ 168 : 400 , 105: ]

        ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        cv2.imwrite("temp/train-1.png", thresh_img)

        value =length_to_cactus(np.array(parse_function("temp/train-1.png"))[cactusBase] ) 

        value = sess.run(pred , feed_dict={x : [value]})

        print value

        if(int(value) != 1) :
            elem.send_keys(Keys.SPACE)
