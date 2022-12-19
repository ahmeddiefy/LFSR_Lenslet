# LFSR_Lenslet
Light Field super-resolution (LFSR) on Lenslet LF images
We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for both training and test.
Please first download the dataset from " https://stuxidianeducn-my.sharepoint.com/personal/zyliang_stu_xidian_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzyliang%5Fstu%5Fxidian%5Fedu%5Fcn%2FDocuments%2Fdatasets&ga=1 "
, and place the 5 datasets to the folder ./datasets/.

run generate_test.m then TEST.py to test our model

For training:
place the 5 datasets train images to the folder ./Training/.
run generate_train.m then LFSR.py to train our model
