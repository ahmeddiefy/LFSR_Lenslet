import os, glob, re, signal, sys, argparse, threading, time
from random import shuffle
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import numpy as np
import scipy.io
from MODEL2x import model   # for 2x LFSR
#from MODEL4x import model  # for 4x LFSR

DATA_PATH = "./train/"
IMG_SIZE = (25,32,32)
OUT_IMG_SIZE = (25,32,32)
BATCH_SIZE = 16
BASE_LR = 0.0002
LR_RATE = 0.1
LR_STEP_SIZE = 80
MAX_EPOCH = 85

USE_QUEUE_LOADING = True

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

def get_train_list(data_path):
    l = glob.glob(os.path.join(data_path,"*"))
    print(len(l))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    print(len(l))
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat"])
            if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat"])

    return train_list

def get_image_batch(train_list,offset,batch_size):
	target_list = train_list[offset:offset+batch_size]
	input_list = []
	gt_list = []
	cbcr_list = []
	for pair in target_list:
		input_img = scipy.io.loadmat(pair[1])['patch']
		gt_img = scipy.io.loadmat(pair[0])['patch']
		input_list.append(input_img)
		gt_list.append(gt_img)
	input_list = np.array(input_list)
	input_list.resize([BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]])
	gt_list = np.array(gt_list)
	gt_list.resize([BATCH_SIZE, OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], OUT_IMG_SIZE[2]])
	return input_list, gt_list, np.array(cbcr_list)

if __name__ == '__main__':
    train_list = get_train_list(DATA_PATH)
	print("use queue loading")


    ### WITH ASYNCHRONOUS DATA LOADING ###

    train_input_single  = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    train_gt_single  	= tf.placeholder(tf.float32, shape=(OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], OUT_IMG_SIZE[2]))
    q = tf.FIFOQueue(10000, [tf.float32, tf.float32], [[IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]], [OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], OUT_IMG_SIZE[2]]])
    enqueue_op = q.enqueue([train_input_single, train_gt_single])

    train_input, train_gt	= q.dequeue_many(BATCH_SIZE)
    


    shared_model = tf.make_template('shared_model', model)
    train_output, weights 	= shared_model(train_input)
    loss = tf.reduce_sum((tf.abs(tf.subtract(train_output, train_gt))))
    
    for w in weights:
        loss += tf.nn.l2_loss(w)*1e-4
        
    tf.summary.scalar("loss", loss)

    global_step 	= tf.Variable(0, trainable=False)
    learning_rate 	= tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE, len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)
    tf.summary.scalar("learning rate", learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(weights, max_to_keep=0)

    shuffle(train_list)
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
		#TensorBoard open log with "tensorboard --logdir=logs"
        if not os.path.exists('logs'):
            os.mkdir('logs')
        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter('logs', sess.graph)

        tf.initialize_all_variables().run()

        if model_path:
            print("restore model...")
            saver.restore(sess, model_path)
            print("Done")

        def load_and_enqueue(coord, file_list, enqueue_op, train_input_single, train_gt_single, idx=0, num_thread=1):
            count = 0;
            length = len(file_list)
            try:
                while not coord.should_stop():
                    i = count % length;
                    input_img	= scipy.io.loadmat(file_list[i][1])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]])
                    gt_img		= scipy.io.loadmat(file_list[i][0])['patch'].reshape([OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], OUT_IMG_SIZE[2]])
                    sess.run(enqueue_op, feed_dict={train_input_single:input_img, train_gt_single:gt_img})
                    count+=1
            except Exception as e:
                print("stopping...", idx, e)
        threads = []
        def signal_handler(signum,frame):
            sess.run(q.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(threads)
            print("Done")
            sys.exit(1)
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)
      
        # create threads
        num_thread=20
        coord = tf.train.Coordinator()
        for i in range(num_thread):
            length = int(len(train_list)/num_thread)
            print('length:   ', length)
            t = threading.Thread(target=load_and_enqueue, args=(coord, train_list[i*length:(i+1)*length],enqueue_op, train_input_single, train_gt_single,  i, num_thread))
            threads.append(t)
            t.start()
        print("num thread:" , len(threads))

        for epoch in range(0, MAX_EPOCH):
            max_step=len(train_list)//BATCH_SIZE
            for step in range(max_step):
                _,l,output,lr, g_step, summary = sess.run([opt, loss, train_output, learning_rate, global_step, merged])
                print("[epoch %2.4f] loss %.4f\t lr %.5f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr))
                file_writer.add_summary(summary, step+epoch*max_step)
            saver.save(sess, "./checkpoints/LFSR_adam_epoch_%03d.ckpt" % epoch ,global_step=global_step)

