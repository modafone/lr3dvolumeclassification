# Copyright (c) Masahiro Oda, Nagoya University, Japan.
#
# Title: Left-Right Relationship-Aware 3D Volume Classification Method
# Authors: Masahiro Oda, Yuichiro Hayashi, Yoshito Otake, Masahiro Hashimoto, Toshiaki Akashi, Shigeki Aoki, Kensaku Mori
# Journal: International Journal of Computer Assisted Radiology and Surgery

from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    Dense,
    Layer,
    Permute,
    Softmax,
    Activation,
    Conv3D,
    Conv2D,
    concatenate,
    BatchNormalization,
    MaxPooling3D,
    GlobalAveragePooling3D,
    Layer,
    Reshape,
    Lambda,
    Multiply,
    SpatialDropout3D,
)
from tensorflow import split, roll, tile, reverse
from tensorflow.nn import softmax
from tensorflow.keras.activations import sigmoid


class rSoftMax(Layer):
    def __init__(self, filters, radix, group_size, **kwargs):
        super(rSoftMax, self).__init__(**kwargs)
        
        self.filters = filters
        self.radix = radix
        self.group_size = group_size
        
        if 1 < radix:
            self.seq1 = Reshape([group_size, radix, filters // group_size])
            self.seq2 = Permute([2, 1, 3])
            self.seq3 = Lambda(lambda x: softmax(x, axis = 1))
            self.seq4 = Reshape([1, 1, radix * filters])
            self.seq = [self.seq1, self.seq2, self.seq3, self.seq4]
        else:
            self.seq1 = Activation(sigmoid)
            self.seq = [self.seq1]

    def call(self, inputs):
        out = inputs
        for l in self.seq:
            out = l(out)
        return out
    
    def get_config(self):
        config = super(rSoftMax, self).get_config()
        config["filters"] = self.filters
        config["radix"] = self.radix
        config["group_size"] = self.group_size
        return config

#2つのtensorを受け取りchannel軸で組み合わせてからconvし，結果をcombineaxis軸で組み合わせて返す
#combineaxis=1:x軸
def ConvShareUnit(filternum, droprate, combineaxis):
    def apply(x0, x1):
        #combine
        xconv = concatenate([x0, x1], axis=-1)
        
        #Shared Convolution
        xconv = Conv3D(filternum, kernel_size=(3,3,3), strides=1, kernel_initializer='he_normal', padding='same')(xconv)
        if droprate > 0:
            xconv = SpatialDropout3D(rate=droprate)(xconv)
        xconv = Activation('relu')(xconv)
        xconv = BatchNormalization(axis=-1)(xconv)
        xconv = Conv3D(filternum, kernel_size=(3,3,3), strides=1, kernel_initializer='he_normal', padding='same')(xconv)
        if droprate > 0:
            xconv = SpatialDropout3D(rate=droprate)(xconv)
        xconv = Activation('relu')(xconv)
        xconv = BatchNormalization(axis=-1)(xconv)
        
        #double tensor along channel axis
        xconv = tile(input=xconv, multiples=[1, 1, 1, 1, 2])
        
        #split along channel axis
        xconv0, xconv1 = split(value=xconv, num_or_size_splits=2, axis=-1)
        
        #reverse one tensor along x axis
        xconv1 = reverse(xconv1, [combineaxis])
        
        #conbine
        res = concatenate([xconv0, xconv1], axis=combineaxis)
        
        return res
    
    return apply

#左右や上下で共通のconvolutionを行う, 左右間のずれを考慮したshiftする
#splitaxis=1:x軸
def LRConvShare3D3(filternum, droprate, shiftvalue, splitaxis):
    def apply(x):
        #shiftvalue = 12
        
        #split along x axis
        x_0, x_1 = split(value=x, num_or_size_splits=2, axis=splitaxis)#-4 -> 1
        
        #reverse one tensor along x axis
        x_1_0 = reverse(x_1, [splitaxis])
        
        #shift
        x_1_1 = roll(input=x_1_0, shift= shiftvalue, axis=1)
        x_1_2 = roll(input=x_1_0, shift=-shiftvalue, axis=1)
        x_1_3 = roll(input=x_1_0, shift= shiftvalue+2, axis=1)
        x_1_4 = roll(input=x_1_0, shift=-shiftvalue-2, axis=1)
        x_1_5 = roll(input=x_1_0, shift= shiftvalue+4, axis=1)
        x_1_6 = roll(input=x_1_0, shift=-shiftvalue-4, axis=1)
        x_1_7 = roll(input=x_1_0, shift= shiftvalue+6, axis=1)
        x_1_8 = roll(input=x_1_0, shift=-shiftvalue-6, axis=1)
        x_1_9 = roll(input=x_1_0, shift= shiftvalue, axis=2)
        x_1_10= roll(input=x_1_0, shift=-shiftvalue, axis=2)
        x_1_11= roll(input=x_1_0, shift= shiftvalue+2, axis=2)
        x_1_12= roll(input=x_1_0, shift=-shiftvalue-2, axis=2)
        x_1_13= roll(input=x_1_0, shift= shiftvalue+4, axis=2)
        x_1_14= roll(input=x_1_0, shift=-shiftvalue-4, axis=2)
        x_1_15= roll(input=x_1_0, shift= shiftvalue+6, axis=2)
        x_1_16= roll(input=x_1_0, shift=-shiftvalue-6, axis=2)
        x_1_17= roll(input=x_1_0, shift=[ shiftvalue, shiftvalue], axis=[1,2])
        x_1_18= roll(input=x_1_0, shift=[-shiftvalue,-shiftvalue], axis=[1,2])
        x_1_19= roll(input=x_1_0, shift=[ shiftvalue,-shiftvalue], axis=[1,2])
        x_1_20= roll(input=x_1_0, shift=[-shiftvalue, shiftvalue], axis=[1,2])
        x_1_21= roll(input=x_1_0, shift=[ shiftvalue+2, shiftvalue+2], axis=[1,2])
        x_1_22= roll(input=x_1_0, shift=[-shiftvalue-2,-shiftvalue-2], axis=[1,2])
        x_1_23= roll(input=x_1_0, shift=[ shiftvalue+2,-shiftvalue-2], axis=[1,2])
        x_1_24= roll(input=x_1_0, shift=[-shiftvalue-2, shiftvalue+2], axis=[1,2])
        x_1_25= roll(input=x_1_0, shift=[ shiftvalue+4, shiftvalue+4], axis=[1,2])
        x_1_26= roll(input=x_1_0, shift=[-shiftvalue-4,-shiftvalue-4], axis=[1,2])
        x_1_27= roll(input=x_1_0, shift=[ shiftvalue+4,-shiftvalue-4], axis=[1,2])
        x_1_28= roll(input=x_1_0, shift=[-shiftvalue-4, shiftvalue+4], axis=[1,2])
        x_1_29= roll(input=x_1_0, shift=[ shiftvalue+6, shiftvalue+6], axis=[1,2])
        x_1_30= roll(input=x_1_0, shift=[-shiftvalue-6,-shiftvalue-6], axis=[1,2])
        x_1_31= roll(input=x_1_0, shift=[ shiftvalue+6,-shiftvalue-6], axis=[1,2])
        x_1_32= roll(input=x_1_0, shift=[-shiftvalue-6, shiftvalue+6], axis=[1,2])
        
        x_1_0 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_0)
        x_1_1 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_1)
        x_1_2 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_2)
        x_1_3 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_3)
        x_1_4 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_4)
        x_1_5 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_5)
        x_1_6 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_6)
        x_1_7 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_7)
        x_1_8 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_8)
        x_1_9 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_9)
        x_1_10= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_10)
        x_1_11= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_11)
        x_1_12= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_12)
        x_1_13= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_13)
        x_1_14= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_14)
        x_1_15= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_15)
        x_1_16= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_16)
        x_1_17= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_17)
        x_1_18= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_18)
        x_1_19= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_19)
        x_1_20= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_20)
        x_1_21= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_21)
        x_1_22= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_22)
        x_1_23= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_23)
        x_1_24= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_24)
        x_1_25= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_25)
        x_1_26= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_26)
        x_1_27= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_27)
        x_1_28= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_28)
        x_1_29= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_29)
        x_1_30= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_30)
        x_1_31= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_31)
        x_1_32= ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_32)
        
        
        res = concatenate([x_1_0, x_1_1, x_1_2, x_1_3, x_1_4, x_1_5, x_1_6, x_1_7, x_1_8, x_1_9, x_1_10, x_1_11, x_1_12, x_1_13, x_1_14, x_1_15, x_1_16, x_1_17, x_1_18, x_1_19, x_1_20, x_1_21, x_1_22, x_1_23, x_1_24, x_1_25, x_1_26, x_1_27, x_1_28, x_1_29, x_1_30, x_1_31, x_1_32], axis=-1)
        
        
        #split attention
        #https://github.com/Burf/ResNeSt-Tensorflow2/blob/2f2f5b101df0171a3b24e6702e72982674765f7c/resnest/splat.py#L40
        group_size = 33
        radix = 1
        expansion = 4
        
        #inter_channel = max(x.shape[-1] * radix // expansion, 32)
        inter_channel = max(x.shape[-1] * group_size // expansion, 165)#group_size=7のとき35,group_sizeの5倍の値を書く
        if 1 < radix:
            split_result = split(value=res, num_or_size_splits=radix, axis=-1)
            out = Add()(split_result)
        else:
            out = res
        out = GlobalAveragePooling3D()(out)
        out = keras.layers.Reshape([1, 1, filternum*group_size])(out)#3Dだから1,1,1,...とすべき？
        
        out = Conv2D(inter_channel, kernel_size=(1,1), strides=1, groups=group_size, use_bias = True, kernel_initializer='he_normal', padding='same')(out)#Conv3Dにすべき？
        out = BatchNormalization(axis=-1, momentum = 0.9, epsilon = 1e-5)(out)
        out = Activation('relu')(out)
        out = Conv2D(filternum*group_size*radix, kernel_size=(1,1), strides=1, groups=group_size, use_bias = True, kernel_initializer='he_normal', padding='same')(out)#Conv3Dにすべき？
        
        attention = rSoftMax(filternum, radix, group_size)(out)
        if 1 < radix:
            attention = split(attention, radix, axis = -1)
            out = Add()([o * a for o, a in zip(split_result, attention)])
        else:
            out = Multiply()([res, attention])
            
        out = Conv3D(filternum, kernel_size=(1,1,1), strides=1, kernel_initializer='he_normal', padding='same')(out)
        
        return out
    
    return apply


def LRConvShare3D3_ablation_noshift(filternum, droprate, shiftvalue, splitaxis):
    def apply(x):
        #split along x axis
        x_0, x_1 = split(value=x, num_or_size_splits=2, axis=splitaxis)#-4 -> 1
        
        #reverse one tensor along x axis
        x_1_0 = reverse(x_1, [splitaxis])
        
        #shift
        x_1_0 = ConvShareUnit(filternum=filternum, droprate=droprate, combineaxis=splitaxis)(x_0, x_1_0)
        
        return x_1_0
    
    return apply




#LR別の処理
def ClassificationModel3D_LR(
        input_shape: int,
        num_classes: int,
        use_softmax: bool = False,
):
    #height, width, depth, _ = input_shape
    droprate = 0.1

    inputs = keras.Input(input_shape)
    x = inputs
    
    #depth1
    enc1_1 = LRConvShare3D3(filternum=8, droprate=droprate, shiftvalue=2, splitaxis=1)(x)
    enc1mp = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(enc1_1)
    
    #depth2
    enc2_1 = LRConvShare3D3(filternum=16, droprate=droprate, shiftvalue=2, splitaxis=1)(enc1mp)
    enc2mp = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(enc2_1)
    
    #depth3
    enc3_1 = LRConvShare3D3(filternum=32, droprate=droprate, shiftvalue=2, splitaxis=1)(enc2mp)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(enc3_1)
    
    
    x = GlobalAveragePooling3D()(x)
    x = Dense(num_classes, name='head')(x)

    if use_softmax:
        x = Softmax()(x)
    return keras.Model(inputs, x)

def ClassificationModel3D_LR_ablation_noshift(
        input_shape: int,
        num_classes: int,
        use_softmax: bool = False,
):
    #height, width, depth, _ = input_shape
    droprate = 0.1

    inputs = keras.Input(input_shape)
    x = inputs
    
    #depth1
    enc1_1 = LRConvShare3D3_ablation_noshift(filternum=8, droprate=droprate, shiftvalue=2, splitaxis=1)(x)
    enc1mp = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(enc1_1)
    
    #depth2
    enc2_1 = LRConvShare3D3_ablation_noshift(filternum=16, droprate=droprate, shiftvalue=2, splitaxis=1)(enc1mp)
    enc2mp = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(enc2_1)
    
    #depth3
    enc3_1 = LRConvShare3D3_ablation_noshift(filternum=32, droprate=droprate, shiftvalue=2, splitaxis=1)(enc2mp)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(enc3_1)
    
    
    x = GlobalAveragePooling3D()(x)
    x = Dense(num_classes, name='head')(x)

    if use_softmax:
        x = Softmax()(x)
    return keras.Model(inputs, x)


#3D CNNでの分類
def ClassificationModel3D_CNN(
        input_shape: int,
        num_classes: int,
        use_softmax: bool = False,
):
    droprate = 0.1
    
    inputs = keras.Input(input_shape)
    x = inputs
    

    x = Conv3D(16, kernel_size=(3,3,3), strides=1, kernel_initializer='he_normal', padding='same')(x)
    if droprate > 0:
        x = SpatialDropout3D(rate=droprate)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv3D(16, kernel_size=(3,3,3), strides=1, kernel_initializer='he_normal', padding='same')(x)
    if droprate > 0:
        x = SpatialDropout3D(rate=droprate)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    
    x = Conv3D(32, kernel_size=(3,3,3), strides=1, kernel_initializer='he_normal', padding='same')(x)
    if droprate > 0:
        x = SpatialDropout3D(rate=droprate)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv3D(32, kernel_size=(3,3,3), strides=1, kernel_initializer='he_normal', padding='same')(x)
    if droprate > 0:
        x = SpatialDropout3D(rate=droprate)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    
    x = Conv3D(64, kernel_size=(3,3,3), strides=1, kernel_initializer='he_normal', padding='same')(x)
    if droprate > 0:
        x = SpatialDropout3D(rate=droprate)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv3D(64, kernel_size=(3,3,3), strides=1, kernel_initializer='he_normal', padding='same')(x)
    if droprate > 0:
        x = SpatialDropout3D(rate=droprate)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    
    x = GlobalAveragePooling3D()(x)
    x = Dense(num_classes, name='head')(x)

    if use_softmax:
        x = Softmax()(x)
    return keras.Model(inputs, x)
