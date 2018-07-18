
import tensorflow as tf
import csv
from PIL import Image
import numpy as np
import PIL.ImageOps    
import cv2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

files = []
action = []

with open("1/out.csv" , "r") as f :
    data = csv.reader(f, delimiter=',')
    for i in data:
        if(i[0] == 1):
            action.append(0)
        else:
            action.append(1)
        files.append(i[1])
        

def parse_function(filename,basewidth):
    img = Image.open(filename).convert('L')
    #img = PIL.ImageOps.invert(img)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)    
    return img


x_v = [np.array(parse_function(i,200)).flatten()  for i in files ] 
y_v = action 

print "Loaded all Images"

print "Starting Training on NN ( Tensorflow ) " 

'''
Using a Neural Network with No Hidden Layers with 55*200 input layers. total of 55 * 200 Values.

The weight dimentions are (55 * 200) * 1 , input is 1 *  ( 55 * 200 )

Output is single neuron ( 1 - Neuron Fired -- Jump ) else ( Leave )

'''

print sum(x_v[0])

# Parameters
#learning_rate = .00000000001
learning_rate =  .0000000001
training_epochs = 100
batch_size = 100
display_step = 1

tf.set_random_seed(777)  
x = tf.placeholder(tf.float32, [None, 11000])
y = tf.placeholder(tf.float32, [None]) 

W = tf.Variable(tf.random_normal([11000, 1]))
b = tf.Variable(tf.random_normal([1]))

pred = tf.nn.sigmoid(tf.matmul(x, W) + b ) 


#cost = tf.reduce_mean(-1 * (tf.reduce_sum(y*tf.log(pred) + tf.reduce_sum( (y-1)*tf.log(pred -1) ) )))
cost = tf.reduce_mean(-1 * (tf.reduce_sum(y*tf.log(pred  + 1e-30 ))  ) ) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


correct_prediction = tf.equal(tf.cast(tf.argmax(pred, 1),"float64"),tf.cast(y , "float64") )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):

        _, c , w = sess.run([optimizer, cost, W], feed_dict={x: x_v,y: y_v})
        print sum(w)
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{}".format(c))

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

        value = parse_function("temp/train-1.png",200)

        if(round(sess.run(pred , feed_dict={x : [np.array(value).flatten()]})) == 0) :
            elem.send_keys(Keys.SPACE)
