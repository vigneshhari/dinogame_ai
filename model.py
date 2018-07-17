
import tensorflow as tf

# First Layer 

X = tf.Placeholder( size = (None , 768))
Y = tf.Placeholder( size = (None ))


# layer 1

W1 = tf.Variable(type=tf.float32  )