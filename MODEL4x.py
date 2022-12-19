import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from einops import rearrange

kernel_size = 3
f = 64
f_ca = 8
n_res_blocks=5
n_res_groups=1
weights = []
w_init = tf.variance_scaling_initializer(scale=1., mode='fan_avg', distribution="uniform")
b_init = tf.zeros_initializer()

def conv2d(x, f_in, f_out, k, name):
    conv_w = tf.get_variable(name + "_w" , [k,k,f_in,f_out], initializer=w_init)
    conv_b = tf.get_variable(name + "_b" , [f_out], initializer=b_init)
    weights.append(conv_w)
    weights.append(conv_b)
    return tf.nn.bias_add(tf.nn.conv2d(x, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)                           
            
def feature_extraction(x):
    return conv2d(x, f_in=1, f_out=f, k=kernel_size, name="conv2d-3-0")
	
def sharpening_block(x,count):
    x1 = tf.space_to_depth(x,block_size=2)
    x2 = conv2d(x1, f_in=256, f_out=f, k=kernel_size, name="conv2d-select"+count)
    x3 = tf.depth_to_space(x2,block_size=2)
    x4 = conv2d(x3, f_in=16, f_out=f, k=kernel_size, name="conv2d-return"+count)
    return x4

def channel_attention(x, name):
    y = tf.compat.v1.keras.layers.GlobalAvgPool2D()(x)
    y = tf.expand_dims(y, axis=1)
    y = tf.expand_dims(y, axis=1)
    conv_name = "conv2d-1" + "_" + name
    y1 = conv2d(y, f_in=f, f_out=f_ca, k=1, name=conv_name)
    y2 = tf.nn.relu(y1)
    conv_name = "conv2d-2" + "_" + name
    y3 = conv2d(y2, f_in=f_ca, f_out=f, k=1, name=conv_name)
    y4 = tf.nn.sigmoid(y3)

    return tf.multiply(y4, x)

def residual_block(x, name):
    skip_conn = x  
    conv_name = "conv2d-1" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.nn.relu(x)
    conv_name = "conv2d-2" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.add(x , skip_conn)
    x = channel_attention(x, name+"_CA_")
    
    return x
   
def residual_group(x, name):
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="conv2d-1" + "_" + name)
    skip_conn = x

    for i in range(n_res_blocks):
        x = residual_block(x, name=name + "_" + str(i))
        
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="rg-conv-" + name)
    return tf.add(x , skip_conn ) 
           
def residual_network(x,count):
    head = x

    for i in range(n_res_groups):
        x = residual_group(x, name=str(i)+count )

    body = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="conv2d-body"+count)
    tail = tf.add(body, head) 

    return tail
           
def model(input_tensor):
    with tf.device("/gpu:0"):
        tensor = tf.expand_dims(input_tensor, axis=4)

        tensor = tf.vectorized_map(  lambda x: feature_extraction(x), tensor, fallback_to_while_loop=True)

        tensor_in = rearrange(tensor, 'b (h1 w1) h w c -> b (h h1) (w w1) c', h1=5, w1=5)
        tensor = residual_network(tensor_in,"Raw_Angular")	
        tensor = rearrange(tensor, 'b (h h1) (w w1) c -> b (h1 w1) h w c', h1=5, w1=5)

        tensor = tf.vectorized_map(  lambda x: sharpening_block(x,"First"), tensor, fallback_to_while_loop=True)

        tensor = rearrange(tensor, 'b (h1 w1) h w c -> b (h h1) (w w1) c', h1=5, w1=5)
        tensor = residual_network(tensor,"Second_after-MIN")	
		
        tensor = rearrange(tensor, 'b (h h1) (w w1) c -> b (h1 w1) h w c', h1=5, w1=5)

        tensor = tf.vectorized_map(  lambda x: sharpening_block(x,"Second"), tensor, fallback_to_while_loop=True)

        tensor = rearrange(tensor, 'b (h1 w1) h w c -> b (h h1) (w w1) c', h1=5, w1=5)
        tensor = residual_network(tensor,"Third_after-MAX")	
		
        tensor = rearrange(tensor, 'b (h h1) (w w1) c -> b (h1 w1) h w c', h1=5, w1=5)

        tensor = tf.vectorized_map(  lambda x: sharpening_block(x,"Third"), tensor, fallback_to_while_loop=True)

        tensor = rearrange(tensor, 'b (h1 w1) h w c -> b (h h1) (w w1) c', h1=5, w1=5)
        tensor = residual_network(tensor,"Fourth_after-MAX")
		
        tensor = rearrange(tensor, 'b (h h1) (w w1) c -> b (h1 w1) h w c', h1=5, w1=5)

        tensor = tf.vectorized_map(  lambda x: sharpening_block(x,"Fourth"), tensor, fallback_to_while_loop=True)

        tensor = rearrange(tensor, 'b (h1 w1) h w c -> b (h h1) (w w1) c', h1=5, w1=5)
        tensor = residual_network(tensor,"Fifth_after-MAX")
		
		
        tensor = tf.add(tensor_in, tensor)
        tensor = conv2d(tensor, f_in=f, f_out=1, k=1, name="conv2d-final")
        tensor = tf.transpose(tensor, perm=[0,3,1,2])
        tensor = tf.space_to_depth(tensor, block_size=5, data_format='NCHW')
        
        return tensor, weights
		
        
        
        
       
        