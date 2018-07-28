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


x_v = [np.array(parse_function(i,270)).flatten()  for i in files ] 
y_v = action 


print "Loaded all Images"

print '''Starting Training CNN ( Tensorflow ) '''

tf.set_random_seed(777)  



x_inp = tf.placeholder(tf.float32, [None, 20250]) 
y = tf.placeholder(tf.float32, [None])

x = tf.reshape(x_inp, [-1, 75, 270, 1])

weight_conv1  = tf.Variable(tf.zeros([5,5,1,32])) 
weight_conv2  = tf.Variable(tf.zeros([5,5,32,64]))
weight_fc     = tf.Variable(tf.zeros([5 * 18 * 64  ,800]))
weight_out    = tf.Variable(tf.zeros([800,1]))

bias_conv1  = tf.Variable(tf.random_normal([32])) 
bias_conv2  = tf.Variable(tf.random_normal([64]))
bias_fc     = tf.Variable(tf.random_normal([800]))
bias_out    = tf.Variable(tf.random_normal([1]))

conv1     = tf.add(tf.nn.conv2d(x , weight_conv1 , strides=[1,1,1,1] , padding="SAME") , bias_conv1) 
max_pool1 = tf.nn.max_pool(conv1 , ksize=[1,5,5,1] , strides=[1,5,5,1] , padding="SAME") 

conv2     = tf.add(tf.nn.conv2d(max_pool1 , weight_conv2 , strides=[1,1,1,1] , padding="SAME") , bias_conv2)
max_pool2 = tf.nn.max_pool(conv2 , ksize=[1,3,3,1] , strides=[1,3,3,1] , padding="SAME") 

fc = tf.reshape(max_pool2 , [-1 , 5 * 18 * 64  ])
fc = tf.nn.relu( tf.add(tf.matmul(fc , weight_fc) , bias_fc) )

out = tf.matmul(fc , weight_out) + bias_out

cost = tf.reduce_mean(-1 * (tf.reduce_sum(y*tf.log(out  + 1e-30 ))  ) ) 
optimizer = tf.train.AdamOptimizer(learning_rate=.000000001).minimize(cost)

correct_prediction = tf.equal(tf.cast(tf.argmax(out, 1),"float64"),tf.cast(y , "float64") )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

hm_epochs = 2
batch_size = 50



for epoch in range(hm_epochs):
    epoch_loss = 0
    for i in range(int(len(files)//batch_size)):
        epoch_x, epoch_y = x_v[ (i * batch_size) : (i +1 ) * batch_size ] , y_v[ (i * batch_size) : (i +1 ) * batch_size ] 
        #print sess.run(max_pool1, feed_dict={x_inp: epoch_x, y: epoch_y}).shape
        _, c = sess.run([optimizer, cost], feed_dict={x_inp: epoch_x, y: epoch_y})
        epoch_loss += c
        print c
    print 'Completed Epoch# : ', epoch, ' : Epochs Left : ',hm_epochs-epoch - 1,' : loss : ',epoch_loss


#print 'Accuracy For Training is ' ,sess.run(accuracy , feed_dict={x_inp:np.array(x_v[0:500]), y:y_v[0:500]}) , "%"
exit()

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

    value = sess.run(out , feed_dict={x_inp : [np.array(value).flatten()]})

    print value

    if(round(value) == 0) :
        elem.send_keys(Keys.SPACE)
