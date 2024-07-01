import tensorflow as tf

smooth = 0.0000001


def jacc_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    jacc = tf.math.divide_no_nan(intersection + smooth, union + smooth)
    return 1 - jacc