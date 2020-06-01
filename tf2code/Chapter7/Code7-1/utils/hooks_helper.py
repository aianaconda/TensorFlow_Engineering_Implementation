

import tensorflow as tf


_TENSORS_TO_LOG = dict((x, x) for x in ['learning_rate',
                                        'cross_entropy',
                                        'train_accuracy'])


def get_logging_tensor_hook(every_n_iter=100, tensors_to_log=None, **kwargs):  # pylint: disable=unused-argument
  """Function to get LoggingTensorHook.

  Args:
    every_n_iter: `int`, print the values of `tensors` once every N local
      steps taken on the current worker.
    tensors_to_log: List of tensor names or dictionary mapping labels to tensor
      names. If not set, log _TENSORS_TO_LOG by default.
    **kwargs: a dictionary of arguments to LoggingTensorHook.

  Returns:
    Returns a LoggingTensorHook with a standard set of tensors that will be
    printed to stdout.
  """
  if tensors_to_log is None:
    tensors_to_log = _TENSORS_TO_LOG

  return tf.compat.v1.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=every_n_iter)




