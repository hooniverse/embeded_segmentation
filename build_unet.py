def original(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up7 = layers.concatenate([layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)
    
    up8 = layers.concatenate([layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)
    
    up9 = layers.concatenate([layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def dilated(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)
    
    # Encoder
    # First block - normal convolution to capture fine details
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', dilation_rate=2)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second block - increasing dilation rate
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2)(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', dilation_rate=4)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Third block - larger dilation rate for broader context
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=4)(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=8)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge - maximum dilation rate for largest receptive field
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same', dilation_rate=8)(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same', dilation_rate=16)(conv4)
    
    # Decoder - gradually decreasing dilation rates
    up7 = layers.concatenate([layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=4)(up7)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=4)(conv7)
    
    up8 = layers.concatenate([layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2)(up8)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2)(conv8)
    
    up9 = layers.concatenate([layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def same_dilated(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)
    
    # Encoder
    # First block
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', dilation_rate=2)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second block
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2)(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Third block
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=2)(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=2)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same', dilation_rate=2)(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same', dilation_rate=2)(conv4)
    
    # Decoder
    up7 = layers.concatenate([layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=2)(up7)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same', dilation_rate=2)(conv7)
    
    up8 = layers.concatenate([layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2)(up8)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2)(conv8)
    
    up9 = layers.concatenate([layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same', dilation_rate=2)(up9)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same', dilation_rate=2)(conv9)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def separable(input_shape=(128, 128, 3)):
    inputs = layers.Input(input_shape)
    
    # Encoder
    # First block
    conv1 = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second block
    conv2 = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Third block
    conv3 = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Bridge
    conv4 = layers.SeparableConv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.SeparableConv2D(256, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    # First block
    up7 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv4)
    up7 = layers.concatenate([up7, conv3], axis=3)
    conv7 = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(conv7)
    
    # Second block
    up8 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2], axis=3)
    conv8 = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(conv8)
    
    # Third block
    up9 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1], axis=3)
    conv9 = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(up9)
    conv9 = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(conv9)
    
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model