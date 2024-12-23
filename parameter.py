def dice_coef(y_true, y_pred, smooth=1e-6): # 다이스 계수
    y_pred = tf.cast(y_pred > 0.5, dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    intersection = tf.cast(intersection, dtype=tf.float32)
    
    y_true = tf.reduce_sum(y_true)
    y_pred = tf.reduce_sum(y_pred)
    
    result = (2.0 *(intersection) + smooth) / (y_true + y_pred + smooth)
    
    return result
    
def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    
def combined_loss(y_true, y_pred, a=0.95):
    logit_loss_value = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
    dice_loss_value = dice_loss(y_true, y_pred)
    
    return a*dice_loss_value+ (1-a)*logit_loss_value

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_dice_coef', # validation dice coefficient
    factor=0.9, # learning rate￿ 10% ￿￿ 기존 0.9
    patience=3, # 3 epoch ￿￿ ￿￿￿ ￿기존 3￿ ￿
    min_lr=1e-6, # ￿￿ learning rate
    mode='max', # dice coefficient￿ ￿￿￿￿ ￿￿
verbose=1
)
        
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_unet_model.keras', # ￿￿￿ ￿￿￿ ￿￿￿
    monitor='val_dice_coef', # validation dice coefficient ￿￿￿￿
    save_best_only=True, # ￿￿ ￿￿ ￿￿￿ ￿￿
    mode='max',          # dice coefficient￿ ￿￿￿￿ ￿￿
    verbose=1
)
