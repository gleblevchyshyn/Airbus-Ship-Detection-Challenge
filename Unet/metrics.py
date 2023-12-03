import tensorflow as tf


def dice_coff(y_true, y_pred, epsilon=1e-7):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    reduced_sum = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon

    coff = (2.*reduced_sum + epsilon)/union
    return coff


def dice_loss(y_true, y_pred, epsilon=1e-7):
    return 1 - dice_coff(y_true, y_pred, epsilon)