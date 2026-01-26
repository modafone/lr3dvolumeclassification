# Copyright (c) Masahiro Oda, Nagoya University, Japan.
#
# Title: Left-Right Relationship-Aware 3D Volume Classification Method
# Authors: Masahiro Oda, Yuichiro Hayashi, Yoshito Otake, Masahiro Hashimoto, Toshiaki Akashi, Shigeki Aoki, Kensaku Mori
# Journal: International Journal of Computer Assisted Radiology and Surgery

import os
import cv2
#import csv
import re
import numpy as np
import pandas as pd
import csv
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
from utils import normalize_ctvalue_onechannel, save_image_nifty, Image3DAddPositionEncode
from numpy.random import *
import elasticdeform
from scipy import ndimage
from models import ClassificationModel3D_LR, ClassificationModel3D_CNN, ClassificationModel3D_LR_ablation_noshift


expindex = 0

IMAGE_SIZE_XY = 144    #CNN入力画像サイズ
IMAGE_SIZE_Z = 48
POSITION_ENCODE = False
INTERPOLATION_METHOD = cv2.INTER_CUBIC
REMOVE_OUTSIDE_LUNG = False
AUGMENTATION_NUM = 0
WINDOWCENTER = -550
WINDOWWIDTH = 1500
#WINDOWCENTER = -300
#WINDOWWIDTH = 1600
NUM_CLASS = 2
EPOCH_NUM = 80
BATCH = 8

CROSSVALIDATION = True
CROSSVALIDATION_CURRENT_FOLD = 0

if CROSSVALIDATION == True:
    FILENAME_RESULTCSV = './classify_result_fold%d_exp%d.csv'%(CROSSVALIDATION_CURRENT_FOLD, expindex)
else:
    FILENAME_RESULTCSV = './classify_result_exp%d.csv'%(expindex)

df_org = pd.read_csv('./dataset.csv')
df_fold_org = pd.read_csv('./filenames_4fold.csv')


def load_3dimages_lungvoi(inputpath, imagesize_xy, imagesize_z):
    imglist = []
    labellist = []
    filenamelist = []
    
    for root, dirs, files in os.walk(inputpath):
        for fn in sorted(files):
            #ファイル名に.niiを含み_lungは含まないファイルを処理する
            if not '.nii' in fn:
                continue
            if '_lung' in fn:
                continue
            if '_segmentation' in fn:
                continue
            if '_voi' in fn:
                continue
            
            
            ###File open
            #filenames
            filename_ct = os.path.join(root, fn)
            
            filename_label = fn.split('.')[0] + '_segmentation.nii.gz' #.より前の文字列切り出し
            filename_label = os.path.join(inputpath, filename_label)
            
            if not os.path.exists(filename_ct):
                print('CT image file not exists: ' + filename_ct)
                continue
            if not os.path.exists(filename_label):
                print('Label image file not exists: ' + filename_label)
                continue
            
            print('Processing: ' + filename_ct)
            
            #ct
            imgnifty_ct = nib.load(filename_ct)
            imgdata_ct = imgnifty_ct.get_fdata()
            
            #label
            imgnifty_label = nib.load(filename_label)
            imgdata_label = imgnifty_label.get_fdata()
            imgdata_label = np.asarray(imgdata_label, dtype=np.int8)
            
            
            #ファイルチェック
            if imgdata_ct.shape[2] <= 40:
                print('Skip processing: slice number <= 40')
                continue
            if imgdata_ct.shape[0] != 512 or imgdata_ct.shape[1] != 512:
                print('Skip processing: not axial slice image')
                continue
            
            
            #正解ラベル取得
            #ファイル名を番号に分割
            fn_parts = re.split('[_.]', fn)
            
            #csvから番号が一致する行抽出
            #省略
            df_line = 0
            if df_line.shape[0] == 0:
                print('Record not found in csv')
                continue
            
            #目的ラベルの列番号
            column = df_org.columns.get_loc('FREE_TEXT')
            
            label = df_line.iat[0, column]
            
            labelvalue = 0
            if label in ['1', '2']:
                labelvalue = 0
                print('label 0')
            elif label in ['3', '4']:
                labelvalue = 1
                print('label 1')
            else:
                print('Label value not found!!!!')
                continue
                
            #CTから肺野のあるスライスのみ取り出し，等方解像度に変換
            #(512,512,z) -> (imagesize,imagesize,z)
            imglist_sub1 = []
            for index_slice in range(imgdata_ct.shape[2]):
                #肺野または炎症ラベルのないスライスはスキップ
                img_label = imgdata_label[:,:,index_slice]
                if (np.count_nonzero(img_label) == 0):
                    continue
                     
                if imgdata_ct.ndim == 4:
                    img = imgdata_ct[:,:,index_slice,0]
                else:
                    img = imgdata_ct[:,:,index_slice]
                #img = imgdata_ct[:,:,index_slice]
                if REMOVE_OUTSIDE_LUNG == True:
                    img = np.where(imgdata_label[:,:,index_slice] > 0, img, WINDOWCENTER)
                img = cv2.resize(img, (imagesize_xy, imagesize_xy), interpolation = INTERPOLATION_METHOD)  #主に縮小するのでINTER_AREA使用
                imglist_sub1.append(img)
            
            imglist_sub1 = np.asarray(imglist_sub1, dtype=np.float32)
            imglist_sub1 = imglist_sub1.transpose(1, 2, 0)   #(z,x,y) -> (x,y,z)
            
            #(imagesize,imagesize,z) -> (imagesize, imagesize, imagesize)
            imglist_sub2 = []
            for index_slice in range(imglist_sub1.shape[1]):
                img = imglist_sub1[:,index_slice,:]
                img = cv2.resize(img, (imagesize_z, imagesize_xy), interpolation = INTERPOLATION_METHOD)  #主に拡大する?
                imglist_sub2.append(img)
            
            imglist_sub2 = np.asarray(imglist_sub2, dtype=np.float32)
            imglist_sub2 = imglist_sub2.transpose(1, 0, 2)   #(y,x,z) -> (x,y,z)
            imglist_sub2 = imglist_sub2.reshape((imglist_sub2.shape[0], imglist_sub2.shape[1], imglist_sub2.shape[2], 1))
            
            imglist.append(imglist_sub2)
            labellist.append(labelvalue)
            filenamelist.append(fn)
            
            #save lung VOI for debug
            bn = fn.split('.')[0]   #.より前の文字列切り出し
            filename = bn + '_voi.nii.gz'
            filename = os.path.join(inputpath, filename)
            save_image_nifty(imglist_sub2, imgnifty_ct, dtype = np.float32, filename_out = filename)
            
    imglist = np.asarray(imglist, dtype=np.float32)
    labellist = np.asarray(labellist, dtype=np.int32)
    
    return imglist, labellist, filenamelist

#3D画像をランダムに変形する．変形後の画像集合とラベル値集合を返す
def generate_augmented_3dimages(images, labels, generatenum):
    imageresults = []
    labelresults = []
    
    for i in range(generatenum):
        #元画像からランダムに1枚抽出
        index = np.random.choice(images.shape[0], 1, replace=False)
        image = images[index[0],:,:,:,0]#色の次元はなくして画像取得
        label = labels[index[0]]
        #print('index:%d, label:%d'%(index,label))
        
        #適用確率
        prob_flipx = np.random.uniform(0, 1)
        prob_rotate = np.random.uniform(0, 1)
        prob_shift = np.random.uniform(0, 1)
        prob_elastic = np.random.uniform(0, 1)
        
        #変形パラメータ
        rotatex = np.random.uniform(-20, 20)
        rotatey = np.random.uniform(-20, 20)
        rotatez = np.random.uniform(-20, 20)
        shiftx = np.random.uniform(-image.shape[0]*0.1, image.shape[0]*0.1)
        shifty = np.random.uniform(-image.shape[1]*0.1, image.shape[1]*0.1)
        shiftz = np.random.uniform(-image.shape[2]*0.1, image.shape[2]*0.1)
        
        #変形適用
        if prob_flipx > 0.5:
            image = np.flip(image, axis=0)
        if prob_rotate > 0.3:
            image = ndimage.rotate(image, angle=rotatex, axes=(1, 2), reshape=False, mode='constant', cval=WINDOWCENTER)#rotate x axis
            image = ndimage.rotate(image, angle=rotatey, axes=(2, 0), reshape=False, mode='constant', cval=WINDOWCENTER)#rotate y axis
            image = ndimage.rotate(image, angle=rotatez, axes=(0, 1), reshape=False, mode='constant', cval=WINDOWCENTER)#rotate z axis
        
        if prob_shift > 0.3:
            image = ndimage.shift(image, shift=[shiftx, shifty, shiftz], mode='constant', cval=WINDOWCENTER)#shift
        
        if prob_elastic > 0.7:
            image = elasticdeform.deform_random_grid(image, sigma=3, points=4, mode='constant', cval=WINDOWCENTER)#nonrigid deform
        #image[image == 0] = WINDOWCENTER #余白を埋める値
        
        imageresults.append(image)
        labelresults.append(label)
    
    if generatenum > 0:
        imageresults = np.asarray(imageresults, dtype=np.float32)
        imageresults = imageresults.reshape((imageresults.shape[0], imageresults.shape[1], imageresults.shape[2], imageresults.shape[3], 1))#色の次元を追加
        labelresults = np.asarray(labelresults, dtype=np.int32)
    
    return imageresults, labelresults

   
data_image1, data_label1, filename1 = load_3dimages_lungvoi('/path/to/dataset1/', IMAGE_SIZE_XY, IMAGE_SIZE_Z)
data_image2, data_label2, filename2 = load_3dimages_lungvoi('/path/to/dataset2/', IMAGE_SIZE_XY, IMAGE_SIZE_Z)
data_image3, data_label3, filename3 = load_3dimages_lungvoi('/path/to/dataset3/', IMAGE_SIZE_XY, IMAGE_SIZE_Z)

data_image = np.concatenate([data_image1, data_image2, data_image3], axis=0)
data_label = np.append(data_label1, data_label2)
data_label = np.append(data_label, data_label3)
filenames = filename1 + filename2 + filename3


#train,testデータ分割
data_image_train = []
data_image_test = []
data_image_val = []
data_label_train = []
data_label_test = []
data_label_val = []
data_test_filenames = []

if type(CROSSVALIDATION_CURRENT_FOLD) is not int:
    #print('Type of CROSSVALIDATION_CURRENT_FOLD is not int')
    CROSSVALIDATION_CURRENT_FOLD = int(CROSSVALIDATION_CURRENT_FOLD)

#cross validation
if CROSSVALIDATION == True:
    for i in range(len(filenames)):
        #csvからファイル名が一致する行抽出
        df_fold_line = df_fold_org[(df_fold_org['filename'] == filenames[i])]

        if df_fold_line.shape[0] == 0:
            print('Filename not found in csv')
            continue
        if df_fold_line.shape[0] > 1:
            print('Multiple Filename found in csv. Why?')
        
        #目的ラベルの列番号
        column_trueclass = df_fold_org.columns.get_loc('true_class')
        column_fold = df_fold_org.columns.get_loc('fold')
        
        trueclass = df_fold_line.iat[0, column_trueclass]#正解クラス番号
        fold = df_fold_line.iat[0, column_fold]#fold番号　0スタート
        
        if type(fold) is not int:
            fold = int(fold)
            
        if fold < 0 or fold > 4:
            print('fold:' + fold)
        
        if (fold == CROSSVALIDATION_CURRENT_FOLD):#test data
            data_image_test.append(data_image[i])
            data_label_test.append(data_label[i])
            data_test_filenames.append(filenames[i])#testデータのファイル名リスト作成
        else:#train data
            data_image_train.append(data_image[i])
            data_label_train.append(data_label[i])
            
    #train,valデータ分割
    #省略
            
    data_image_train = np.asarray(data_image_train, dtype=np.float32)
    data_image_test = np.asarray(data_image_test, dtype=np.float32)
    data_image_val = np.asarray(data_image_val, dtype=np.float32)
    data_label_train = np.asarray(data_label_train, dtype=np.int32)
    data_label_test = np.asarray(data_label_test, dtype=np.int32)
    data_label_val = np.asarray(data_label_val, dtype=np.int32)
    
    
#random split
else:
    indices = np.array(range(data_image.shape[0]))
    data_image_train, data_image_test, data_label_train, data_label_test, data_index_train, data_index_test = train_test_split(data_image, data_label, indices, test_size=0.2)

    #testデータのファイル名リスト作成
    for i in range(len(data_index_test)):
        data_test_filenames.append(filenames[data_index_test[i]])
    

    #train,valデータ分割
    data_image_train, data_image_val, data_label_train, data_label_val = train_test_split(data_image_train, data_label_train, test_size=0.25)


#data augmentation
if AUGMENTATION_NUM > 0:
    print('Generating deformed images')
    data_image_deformed, data_label_deformed = generate_augmented_3dimages(data_image_train, data_label_train, AUGMENTATION_NUM)
    
    data_image_train = np.concatenate([data_image_train, data_image_deformed], axis=0)
    data_label_train = np.append(data_label_train, data_label_deformed)


if POSITION_ENCODE == True:
    #Position encode次元追加
    data_image_train = Image3DAddPositionEncode(data_image_train)
    data_image_test = Image3DAddPositionEncode(data_image_test)

print('Train')
print('num of label 0: ' + repr(np.count_nonzero(data_label_train == 0)) + '. label 1: ' + repr(np.count_nonzero(data_label_train == 1)) + ', label 2<=: ' + repr(np.count_nonzero(data_label_train >= 2)))
print('Val')
print('num of label 0: ' + repr(np.count_nonzero(data_label_val == 0)) + '. label 1: ' + repr(np.count_nonzero(data_label_val == 1)) + ', label 2<=: ' + repr(np.count_nonzero(data_label_val >= 2)))
print('Test')
print('num of label 0: ' + repr(np.count_nonzero(data_label_test == 0)) + '. label 1: ' + repr(np.count_nonzero(data_label_test == 1)) + ', label 2<=: ' + repr(np.count_nonzero(data_label_test >= 2)))

#濃度値範囲補正
data_image_train = normalize_ctvalue_onechannel(data_image_train, WINDOWCENTER, WINDOWWIDTH, output_maxvalue = 2.0)
data_image_val = normalize_ctvalue_onechannel(data_image_val, WINDOWCENTER, WINDOWWIDTH, output_maxvalue = 2.0)
data_image_test = normalize_ctvalue_onechannel(data_image_test, WINDOWCENTER, WINDOWWIDTH, output_maxvalue = 2.0)
data_image_train -= 1.0
data_image_val -= 1.0
data_image_test -= 1.0

if np.any(np.isnan(data_image_train)):
    print('data_image_train has nan')
if np.any(np.isnan(data_image_val)):
    print('data_image_val has nan')
if np.any(np.isnan(data_image_test)):
    print('data_image_test has nan')


#正解ラベルをone hot vector形式に変換
data_label_train_binary = to_categorical(data_label_train, num_classes = NUM_CLASS)
data_label_val_binary = to_categorical(data_label_val, num_classes = NUM_CLASS)
data_label_test_binary = to_categorical(data_label_test, num_classes = NUM_CLASS)


#Proposed model
model = ClassificationModel3D_LR(input_shape=data_image_train.shape[1:],
                        num_classes=NUM_CLASS, 
                        use_softmax=True)
#Proposed model for ablation study
# model = ClassificationModel3D_LR_ablation_noshift(input_shape=data_image_train.shape[1:],
#                         num_classes=NUM_CLASS, 
#                         use_softmax=True)

# model = ClassificationModel3D_CNN(input_shape=data_image_train.shape[1:],
#                         num_classes=NUM_CLASS, 
#                         use_softmax=True)

#model.summary()

# trainingの設定
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

#save best model
modelCheckpoint = ModelCheckpoint(filepath = './classify_weight_best%d.hdf5'%(expindex), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)

#learning rate
import math
def lr_scheduler(epoch):
    if epoch < 2000:
        return 0.005
    else:
        return 0.0000005 * math.exp(0.01 * (1000 - epoch))
    
def lr_scheduler2(step):
    base = 0.005
    warmup_steps = 10
    total_steps = EPOCH_NUM
    decay_type = 'cosine'
    
    """Step to learning rate function."""
    lr = base
    
    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = np.clip(progress, 0.0, 1.0)
    if decay_type == 'linear':
        lr = linear_end + (lr - linear_end) * (1.0 - progress)
    elif decay_type == 'cosine':
        lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    else:
        raise ValueError(f'Unknown lr type {decay_type}')
    
    if warmup_steps:
        lr = lr * np.minimum(1., step / float(warmup_steps))

    return lr

lr_callback = LearningRateScheduler(lr_scheduler2)

# training実行
training = model.fit(data_image_train, data_label_train_binary,
                      epochs=EPOCH_NUM, batch_size=BATCH, shuffle=True, validation_data=(data_image_val, data_label_val_binary), verbose=1, callbacks=[lr_callback, modelCheckpoint])

# training結果をファイルに保存
# 重み
if CROSSVALIDATION == True:
    model.save_weights('./classify_weight_fold%d_exp%d.hdf5'%(CROSSVALIDATION_CURRENT_FOLD, expindex))
else:
    model.save_weights('./classify_weight_exp%d.hdf5'%(expindex))

# 学習履歴グラフ表示
def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()
    
plot_history(training)

# 推定
results = list(model.predict(data_image_test, batch_size=2, verbose=1))

# 認識率を計算
scores = model.evaluate(data_image_test, data_label_test_binary, batch_size=2, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# 評価
results = list(np.argmax(model.predict(data_image_test, batch_size=2, verbose=1), axis=-1))

#score = accuracy_score(data_label_test, results)
#print()
#print(score)
cmatrix = confusion_matrix(data_label_test, results)
print(cmatrix)


# ファイル書き出し
result = model.predict(data_image_test, batch_size=2, verbose=1)

#csvファイルに書き出し
f = open(FILENAME_RESULTCSV, 'w')
writer = csv.writer(f, lineterminator='\n')

savedata = ['filename', 'true_class', 'estimate_class', 'estimate_prob', 'estimate_prob_class0', 'estimate_prob_class1']
writer.writerow(savedata)

for i in range(len(data_image_test)):
    savedata = [data_test_filenames[i], data_label_test[i], np.argmax(result[i]), np.max(result[i]), result[i][0], result[i][1]]
    writer.writerow(savedata)
    
f.close()
