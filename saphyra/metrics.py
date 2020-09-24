
__all__ = [
  "auc",
  "f1_score",
  "sp",
  "pd",
  "fa"
]

import numpy as np 
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K
import tensorflow as tf 
tf.executing_eagerly()


def auc(y_true, y_pred, num_thresholds=2000):
  import tensorflow as tf
  auc = tf.metrics.auc(y_true, y_pred,num_thresholds=num_thresholds)[1]
  K.get_session().run(tf.local_variables_initializer())
  return auc


def f1_score(y_true, y_pred):
  import tensorflow as tf
  f1 = tf.contrib.metrics.f1_score(y_true, y_pred)[1]
  K.get_session().run(tf.local_variables_initializer())
  return f1



class sp(AUC):

  # This implementation works with Tensorflow backend tensors.
  # That way, calculations happen faster and results can be seen
  # while training, not only after each epoch
  def result(self):

    # Add K.epsilon() for forbiding division by zero
    fa = self.false_positives / (self.true_negatives + self.false_positives + K.epsilon())
    pd = self.true_positives  / (self.true_positives + self.false_negatives + K.epsilon())

    sp = K.sqrt(  K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))
    knee = K.argmax(sp)
    return sp[knee]

class pd(AUC):

  # This implementation works with Tensorflow backend tensors.
  # That way, calculations happen faster and results can be seen
  # while training, not only after each epoch
  def result(self):

    # Add K.epsilon() for forbiding division by zero
    fa = self.false_positives / (self.true_negatives + self.false_positives + K.epsilon())
    pd = self.true_positives  / (self.true_positives + self.false_negatives + K.epsilon())
    
    sp = K.sqrt(  K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = K.argmax(sp)
    return pd[knee]


class fa(AUC):

  # This implementation works with Tensorflow backend tensors.
  # That way, calculations happen faster and results can be seen
  # while training, not only after each epoch
  def result(self):

    # Add K.epsilon() for forbiding division by zero
    fa = self.false_positives / (self.true_negatives + self.false_positives + K.epsilon())
    pd = self.true_positives  / (self.true_positives + self.false_negatives + K.epsilon())

    sp = K.sqrt(  K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = K.argmax(sp)
    return fa[knee]
