import numpy as np
import cv2
from scipy import misc
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob, os, re
import scipy.io
import pickle
from MODEL2x import model   # for 2x LFSR
#from MODEL4x import model  # for 4x LFSR
from skimage.metrics import structural_similarity as ssim
import time
import math

DATA_PATH = "./data/test/2xSR/"    # for 2x LFSR
# DATA_PATH = "./data/test/4xSR/"  # for 4x LFSR

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
def calculate_psnr_ssim(out,gt):
          
    psnr_total = np.zeros((25))
    ssim_total = np.zeros((25))
    for ax in range(25):
        psnr_total[ax] = psnr(out[ax,:,:] , gt[ax,:,:]) 
        ssim_total[ax] = ssim(out[ax,:,:], gt[ax,:,:])


    average_psnr = np.sum(psnr_total)/25 
    average_ssim = np.sum(ssim_total)/25
    
    return average_psnr, average_ssim
 
 
 
def get_img_list(data_path):
    l = glob.glob(os.path.join(data_path,"*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4]+"_2.mat"): train_list.append([f, f[:-4]+"_2.mat", 2])
            if os.path.exists(f[:-4]+"_4.mat"): train_list.append([f, f[:-4]+"_4.mat", 4])

    return train_list
def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset+batch_size]
    input_list = []
    gt_list = []
    scale_list = []
    for pair in target_list:
        mat_dict = scipy.io.loadmat(pair[1])
        input_img = None
        if ('img_2') in mat_dict: 	input_img = mat_dict['img_2']
        elif ("img_4") in mat_dict: 	input_img = mat_dict["img_4"]
        else: continue
        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        input_list.append(input_img)
        gt_list.append(gt_img)
        scale_list.append(pair[2])
    return input_list, gt_list, scale_list
def test_LFSR_with_sess(epoch, ckpt_path, data_path,sess):
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))
    sub_folders = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
    
    for x in range(len(sub_folders)):
        folder_list = glob.glob(os.path.join(data_path, sub_folders[x]))
        img_list = sorted(get_img_list(os.path.join(data_path, sub_folders[x])))
        avg_psnr = 0
        avg_ssim = 0
        avg_time = 0
        for i in range(len(img_list)):
            input_list, gt_list, scale_list = get_test_image(img_list, i, 1)
            input_y = input_list[0]
            gt_y = gt_list[0]
            start_t = time.time()
            img_out_y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], input_y.shape[2]))})
            img_out_y = np.resize(img_out_y, (gt_y.shape[0], gt_y.shape[1], gt_y.shape[2]))
            end_t = time.time()
            avg_time += end_t-start_t
            psnr_value, ssim_value = calculate_psnr_ssim(img_out_y,gt_y)
            avg_psnr += psnr_value
            avg_ssim += ssim_value
        print("Average PSNR  ", avg_psnr/len(img_list), "and SSIM  ",  avg_ssim/len(img_list), "  Time  ", avg_time/len(img_list), " on  ", sub_folders[x])
   

            

def test_LFSR(epoch, ckpt_path, data_path):
	with tf.Session() as sess:
		test_LFSR_with_sess(epoch, ckpt_path, data_path, sess)
if __name__ == '__main__':
    model_list = sorted(glob.glob("./checkpoints/LFSR_adam_epoch_*"))
    model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
    model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("index")]
    with tf.Session() as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, None))
        shared_model = tf.make_template('shared_model', model)
        output_tensor, weights 	= shared_model(input_tensor)
        saver = tf.compat.v1.train.Saver(weights)
        tf.initialize_all_variables().run()
        for model_ckpt in model_list:
            print(model_ckpt)
            epoch = int(model_ckpt.split('epoch_')[-1].split('.ckpt')[0])
            print(epoch)
            test_LFSR_with_sess(epoch, model_ckpt, DATA_PATH,sess)
