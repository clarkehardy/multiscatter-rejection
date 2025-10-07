import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, losses, callbacks

def scheduler(epoch, lr):
    if (epoch + 1) % 10 != 0:
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

def ResNet14(input_shape=(32, 32, 3)):
    """
    CIFAR-style ResNet-14 (n=2):
    conv(8) ->
    [2 x (8, stride1)] ->
    [2 x (16, stride2 first)] ->
    [2 x (32, stride2 first)] ->
    BN -> ReLU -> GlobalAvgPool -> Dense (logits)

    Depth uses CIFAR convention 6n+2 = 14 (excludes projection 1x1 shortcuts).
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(8, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 1: 2 blocks, 8 filters, no downsampling  (unchanged)
    for _ in range(2):
        x = _resnet_block(x, filters=8, stride=1)

    # Stage 2: 2 blocks total (first downsamples), 16 filters
    x = _resnet_block(x, filters=16, stride=2)
    for _ in range(1):  # was range(2)
        x = _resnet_block(x, filters=16, stride=1)

    # Stage 3: 2 blocks total (first downsamples), 32 filters
    x = _resnet_block(x, filters=32, stride=2)
    for _ in range(1):  # was range(2)
        x = _resnet_block(x, filters=32, stride=1)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(1, activation=None)(x)  # logits
    model = models.Model(inputs, outputs, name="ResNet14")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="AUC")]
    )
    return model

def _conv_block(x, filters, l2=5e-4):
    """VGG-style block: Conv -> Conv -> BN -> ReLU -> MaxPool -> Dropout"""
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Conv2D(filters, (3, 3), padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    return x

def MiniVGG(input_shape, num_classes=1, l2=5e-4, fc_units=128, dropout_fc=0.5):
    """
    Build a compact VGG-like CNN.

    Parameters
    ----------
    input_shape : tuple
        (H, W, C), e.g. (19, 29, 1) for single-channel images.
    num_classes : int
        If 1 -> binary classification with sigmoid.
        If >1 -> multi-class classification with softmax.
    l2 : float
        L2 regularization strength for Conv and Dense kernels.
    fc_units : int
        Hidden units in the fully-connected layer after conv blocks.
    dropout_fc : float
        Dropout rate applied before the classifier.

    Returns
    -------
    model : tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    x = _conv_block(inputs, 16, l2=l2)     # Block 1
    x = _conv_block(x, 32, l2=l2)          # Block 2
    x = _conv_block(x, 64, l2=l2)         # Block 3 (keep compact)

    # For very small inputs, Flatten is fine and preserves capacity
    x = layers.Flatten()(x)
    x = layers.Dense(fc_units, kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_fc)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
        metrics = ["accuracy", "AUC"]
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model = models.Model(inputs, outputs, name="mini_vgg")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=metrics,
    )
    return model

