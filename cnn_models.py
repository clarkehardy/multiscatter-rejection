import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, losses, callbacks

def scheduler(epoch, lr):
    if epoch % 10 != 0:
        return lr
    else:
        return lr * 0.1

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

def ResNet14(input_shape=(32, 32, 3), num_classes=10):
    """
    CIFAR-style ResNet-14: conv(16) -> [3 x (16)] -> [3 x (32, stride2 first)] ->
    [3 x (64, stride2 first)] -> GAP -> Dense
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(8, 3, padding="same", use_bias=False,
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

    model = models.Model(inputs, outputs, name="ResNet14")

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                           tf.keras.metrics.AUC(name="AUC")])
    return model
