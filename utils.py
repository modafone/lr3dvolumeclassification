# Copyright (c) Masahiro Oda, Nagoya University, Japan.
#
# Title: Left-Right Relationship-Aware 3D Volume Classification Method
# Authors: Masahiro Oda, Yuichiro Hayashi, Yoshito Otake, Masahiro Hashimoto, Toshiaki Akashi, Shigeki Aoki, Kensaku Mori
# Journal: International Journal of Computer Assisted Radiology and Surgery

import numpy as np
import nibabel as nib
from copy import deepcopy

def Image3DAddPositionEncode(imagelist):
    #x
    #array1d_x = np.arange(0.0, imagelist.shape[1], 1.0, dtype = 'float32')  #[0,1,2,...,511]
    array1d_x = np.arange(imagelist.shape[1], 0.0, -1.0, dtype = 'float32')  #[511,510,509,...,0]
    array1d_x /= np.max(array1d_x)  #[0,...,1.0], (shape[1],)
    array1d_x = array1d_x.reshape(-1,1,1) #(shape[1],1,1)
    array3d_x = np.tile(array1d_x, (1, imagelist.shape[2], imagelist.shape[3]))
    
    #y
    #array1d_y = np.arange(0.0, imagelist.shape[2], 1.0, dtype = 'float32')
    array1d_y = np.arange(imagelist.shape[2], 0.0, -1.0, dtype = 'float32')
    array1d_y /= np.max(array1d_y)
    array1d_y = array1d_y.reshape(1,-1,1) #(1,shape[2],1)
    array3d_y = np.tile(array1d_y, (imagelist.shape[1], 1, imagelist.shape[3]))
    
    #z
    #array1d_z = np.arange(0.0, imagelist.shape[3], 1.0, dtype = 'float32')
    array1d_z = np.arange(imagelist.shape[3], 0.0, -1.0, dtype = 'float32')
    array1d_z /= np.max(array1d_z)
    array1d_z *= float(imagelist.shape[3]) / float(imagelist.shape[1])   #xy方向の画素数とz方向の画素数の比率により値変更
    array1d_z = array1d_z.reshape(1,1,-1) #(1,1,shape[3])
    array3d_z = np.tile(array1d_z, (imagelist.shape[1], imagelist.shape[2], 1))
    
    imagelist_new = np.zeros((imagelist.shape[0], imagelist.shape[1], imagelist.shape[2], imagelist.shape[3], 4))
    
    for i in range(imagelist.shape[0]):
        imagelist_new[i,:,:,:,0] = imagelist[i,:,:,:,0]
        imagelist_new[i,:,:,:,1] = array3d_x
        imagelist_new[i,:,:,:,2] = array3d_y
        imagelist_new[i,:,:,:,3] = array3d_z
    
    return imagelist_new


#niftyファイルを保存
#img:画像, img_nifty:ヘッダ情報参照用の元画像(get_fdata前), dtype:データ型, filename_out:ファイル名
def save_image_nifty(img, img_nifty, dtype, filename_out):
    #header
    new_header = nib.Nifti1Header()
    new_header.set_data_shape(img_nifty.header.get_data_shape())
    new_header.set_zooms(img_nifty.header.get_zooms())
    new_header.set_data_dtype(dtype)
    
    #save
    img_nifty_out = nib.Nifti1Image(np.asarray(img, dtype=dtype), affine=np.eye(4), header=new_header)
    img_nifty_out.header['pixdim'] = img_nifty.header['pixdim'] #これがないと画素サイズなぜか保存されない
    nib.save(img_nifty_out, filename_out)

#CT像の濃度値範囲補正
#濃度値をwc,wwをもとに0-output_maxvalueの範囲にする
def normalize_ctvalue(img, windowcenter, windowwidth, output_maxvalue = 1.0):
    ctvalue_min = windowcenter - windowwidth/2.0;
    ctvalue_max = windowcenter + windowwidth/2.0;
    
    img = np.where((img < ctvalue_min), ctvalue_min, img)
    img = np.where((img > ctvalue_max), output_maxvalue, (img-ctvalue_min) / (ctvalue_max - ctvalue_min) * output_maxvalue)
    
    return img

#CT像の濃度値範囲補正
#濃度値をwc,wwをもとに0-output_maxvalueの範囲にする
#1 channel only
def normalize_ctvalue_onechannel(img, windowcenter, windowwidth, output_maxvalue = 1.0):
    ctvalue_min = windowcenter - windowwidth/2.0;
    ctvalue_max = windowcenter + windowwidth/2.0;
    
    img_backup = deepcopy(img)
    
    img = np.where((img < ctvalue_min), ctvalue_min, img)
    img = np.where((img > ctvalue_max), output_maxvalue, (img-ctvalue_min) / (ctvalue_max - ctvalue_min) * output_maxvalue)
    
    if img.shape[-1] >= 2:
        img[...,1] = img_backup[...,1]
    if img.shape[-1] >= 3:
        img[...,2] = img_backup[...,2]
    if img.shape[-1] >= 4:
        img[...,3] = img_backup[...,3]
    
    return img
