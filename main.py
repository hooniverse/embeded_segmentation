import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import matplotlib.pyplot as plt
import random
import time


import data
import build_unet
import parameter
import output

np.random.seed(42)
tf.random.set_seed(42)


#데이터 로드
train_dir = '/split_datasets/train'
val_dir = '/split_datasets/val'
test_dir = '/split_datasets/test'


X_train, y_train = data.load(train_dir)
X_val, y_val = data.load(val_dir)
X_test, y_test= data.load(test_dir)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)


#데이터 증강
X_train, y_train = data.augment(X_train, y_train, num_aug=2)


#모델 생성
model = build_unet.dilated()
model.summary()

model.compile(optimizer='adam',
    loss= parameter.combined_loss,
    metrics=[parameter.dice_coef])


#모델 학습
history = model.fit(X_train, y_train,
                    batch_size=16,  # 기존 16
                    epochs=30,    # 기존 50
                    validation_data=(X_val, y_val),
                    callbacks=[parameter.reduce_lr, parameter.model_checkpoint],
                    shuffle=True              # 데이터 셔플링
)


#최종 모델 평가
best_model = tf.keras.models.load_model('pruned_unet.keras',
    custom_objects={
        'combined_loss': parameter.combined_loss,
        'dice_coef': parameter.dice_coef})

start_time = tf.timestamp()
test_loss, test_dice = best_model.evaluate(X_test, y_test)
end_time = tf.timestamp()
print(f"Inference time: {end_time - start_time} seconds")



#결과 출력
output.loss_and_dice()
output.visualize_results()
flops = output.calculate_flops(best_model)
print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")

