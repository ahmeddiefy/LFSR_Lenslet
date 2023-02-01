import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from einops import rearrange

kernel_size = 3
f = 64
n_res_blocks=3
n_res_groups=1
weights = []
w_init = tf.variance_scaling_initializer(scale=1., mode='fan_avg', distribution="uniform")
b_init = tf.zeros_initializer()



def conv2d(x, f_in, f_out, k, name):
   
    conv_w = tf.get_variable(name + "_w" , [k,k,f_in,f_out], initializer=w_init)
    weights.append(conv_w)
    return tf.nn.conv2d(x, conv_w, strides=[1,1,1,1], padding='SAME')                          


            
def feature_extraction(x):
    return conv2d(x, f_in=1, f_out=f, k=kernel_size, name="conv2d-3-0")
	
def residual_block(x, name):
    skip_conn = x  
    conv_name = "conv2d-1" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.nn.relu(x)
    conv_name = "conv2d-2" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.nn.relu(x)
    conv_name = "conv2d-3" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    x = tf.add(x , skip_conn)
    
    return x
    
    
    
    
def residual_group(x, name):
    conv_name = "conv2d-1" + "_" + name
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name=conv_name)
    skip_conn = x

    for i in range(n_res_blocks):
        x = residual_block(x, name=name + "_" + str(i))
        skip_conn = tf.concat([skip_conn,x],axis=3)
        
    x = conv2d(skip_conn, f_in=(n_res_blocks+1)*f, f_out=f, k=1, name="fuse-1-" + name)
    x = tf.nn.relu(x)
    x = conv2d(x, f_in=f, f_out=f, k=kernel_size, name="fuse-2-" + name)

    return x
            
            
            
            
            

            
def residual_network(x,count):
    # 1. head
    head = x
    # 2. body

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

        tensor = tf.vectorized_map(  lambda x: residual_network(x,"First"), tensor, fallback_to_while_loop=True)

        tensor = rearrange(tensor, 'b (h1 w1) h w c -> b (h h1) (w w1) c', h1=5, w1=5)
        tensor = residual_network(tensor,"Second_after-MIN")	
		
        tensor = rearrange(tensor, 'b (h h1) (w w1) c -> b (h1 w1) h w c', h1=5, w1=5)

        tensor = tf.vectorized_map(  lambda x: residual_network(x,"Second"), tensor, fallback_to_while_loop=True)

        tensor = rearrange(tensor, 'b (h1 w1) h w c -> b (h h1) (w w1) c', h1=5, w1=5)
        tensor = residual_network(tensor,"Third_after-MAX")	
		
        tensor = rearrange(tensor, 'b (h h1) (w w1) c -> b (h1 w1) h w c', h1=5, w1=5)
 
        tensor = tf.vectorized_map(  lambda x: residual_network(x,"Third"), tensor, fallback_to_while_loop=True)

        tensor = rearrange(tensor, 'b (h1 w1) h w c -> b (h h1) (w w1) c', h1=5, w1=5)
        tensor = residual_network(tensor,"Fourth_after-MAX")
		
        tensor = tf.add(tensor_in, tensor)
        tensor = conv2d(tensor, f_in=f, f_out=1, k=1, name="conv2d-final")
        tensor = tf.transpose(tensor, perm=[0,3,1,2])
        tensor = tf.space_to_depth(tensor, block_size=5, data_format='NCHW')
        
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        return tensor, weights
		
        
        
        
       
        