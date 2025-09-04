import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, losses, callbacks

def build_pmt_cnn(input_channels=2, weight_decay=1e-4, dropout=0.1):
    """
    Tiny CNN for SS/MS classification on rasterized PMT images.
    - 3x3 convs
    - two stride-2 downsamples
    - global average pooling
    - single-logit head

    Args:
        input_channels (int): e.g., 2 for [counts, mask], or 4 if adding CoordConv [x,y].
        weight_decay (float): L2 kernel regularization for convs/dense.
        dropout (float): Dropout before the final FC.
        use_total_feature (bool): If True, accepts an extra scalar (total photons).

    Returns:
        tf.keras.Model
    """
    l2 = regularizers.l2(weight_decay)

    # Variable spatial dims; channels_last
    img_in = layers.Input(shape=(None, None, input_channels), name="img")

    x = layers.Conv2D(32, 3, padding="same", use_bias=False, kernel_regularizer=l2)(img_in)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    logits = layers.Dense(1, kernel_regularizer=l2, activation='sigmoid')(x)
    inputs = img_in

    model = models.Model(inputs=inputs, outputs=logits, name="pmt_tiny_cnn")

    # Use logits + from_logits=True for numerical stability
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.AUC(name="AUC"), tf.keras.metrics.BinaryAccuracy(name="acc")]
    )
    return model

def scheduler(epoch, lr):
    if epoch % 10 != 0:
        return lr
    else:
        return lr * 0.1

def build_tiny_vgg(input_shape=(20, 20, 3), dropout_rate=0.1):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)  # binary
    model = models.Model(inputs, outputs)
    return model

def _bn_relu_conv(x, filters, kernel_size, stride=1):
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return layers.Conv2D(filters, kernel_size, strides=stride, padding="same",
                         use_bias=False, kernel_initializer="he_normal")(x)

def _resnet_block(x, filters, stride):
    """Basic 2-conv residual block. Projection shortcut if shape/stride changes."""
    shortcut = x
    # First conv may downsample
    x = _bn_relu_conv(x, filters, 3, stride=stride)
    # Second conv
    x = _bn_relu_conv(x, filters, 3, stride=1)
    # Projection if needed
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                 use_bias=False, kernel_initializer="he_normal")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    return x

def ResNet20(input_shape=(32, 32, 3), num_classes=10):
    """
    CIFAR-style ResNet-20: conv(16) -> [3 x (16)] -> [3 x (32, stride2 first)] ->
    [3 x (64, stride2 first)] -> GAP -> Dense
    """
    inputs = layers.Input(shape=input_shape)

    # CIFAR stem: 3x3 conv, 16 filters (no BN before first conv per original paper)
    x = layers.Conv2D(16, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1: 2 blocks, 8 filters, no downsampling
    for _ in range(2):
        x = _resnet_block(x, filters=8, stride=1)

    # Stage 2: 2 blocks, first downsamples to 16 filters
    x = _resnet_block(x, filters=16, stride=2)
    for _ in range(2):
        x = _resnet_block(x, filters=16, stride=1)

    # Stage 3: 2 blocks, first downsamples to 32 filters
    x = _resnet_block(x, filters=32, stride=2)
    for _ in range(2):
        x = _resnet_block(x, filters=32, stride=1)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation=None)(x)  # logits

    model = models.Model(inputs, outputs, name="ResNet20")

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                         tf.keras.metrics.AUC(name="AUC")])
    return model
