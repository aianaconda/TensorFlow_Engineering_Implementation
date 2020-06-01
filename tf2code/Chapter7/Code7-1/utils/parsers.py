

import argparse

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


class BaseParser(argparse.ArgumentParser):
  """Parser to contain flags which will be nearly universal across models.

  Args:
    add_help: Create the "--help" flag. False if class instance is a parent.
    data_dir: Create a flag for specifying the input data directory.
    model_dir: Create a flag for specifying the model file directory.
    train_epochs: Create a flag to specify the number of training epochs.
    epochs_between_evals: Create a flag to specify the frequency of testing.
    stop_threshold: Create a flag to specify a threshold accuracy or other
      eval metric which should trigger the end of training.
    batch_size: Create a flag to specify the batch size.
    multi_gpu: Create a flag to allow the use of all available GPUs.
    hooks: Create a flag to specify hooks for logging.
    export_dir: Create a flag to specify where a SavedModel should be exported.
  """

  def __init__(self, add_help=False, data_dir=True, model_dir=True,
               train_epochs=True, epochs_between_evals=True,
               stop_threshold=True, batch_size=True, multi_gpu=True,
               hooks=True, export_dir=True):
    super(BaseParser, self).__init__(add_help=add_help)

    if data_dir:
      self.add_argument(
          "--data_dir", "-dd", default="/tmp",
          help="[default: %(default)s] The location of the input data.",
          metavar="<DD>",
      )

    if model_dir:
      self.add_argument(
          "--model_dir", "-md", default="/tmp",
          help="[default: %(default)s] The location of the model checkpoint "
               "files.",
          metavar="<MD>",
      )

    if train_epochs:
      self.add_argument(
          "--train_epochs", "-te", type=int, default=1,
          help="[default: %(default)s] The number of epochs used to train.",
          metavar="<TE>"
      )

    if epochs_between_evals:
      self.add_argument(
          "--epochs_between_evals", "-ebe", type=int, default=1,
          help="[default: %(default)s] The number of training epochs to run "
               "between evaluations.",
          metavar="<EBE>"
      )

    if stop_threshold:
      self.add_argument(
          "--stop_threshold", "-st", type=float, default=None,
          help="[default: %(default)s] If passed, training will stop at "
          "the earlier of train_epochs and when the evaluation metric is "
          "greater than or equal to stop_threshold.",
          metavar="<ST>"
      )

    if batch_size:
      self.add_argument(
          "--batch_size", "-bs", type=int, default=32,
          help="[default: %(default)s] Batch size for training and evaluation.",
          metavar="<BS>"
      )

    if multi_gpu:
      self.add_argument(
          "--multi_gpu", action="store_true",
          help="If set, run across all available GPUs."
      )

    if export_dir:
      self.add_argument(
          "--export_dir", "-ed",
          help="[default: %(default)s] If set, a SavedModel serialization of "
               "the model will be exported to this directory at the end of "
               "training. See the README for more details and relevant links.",
          metavar="<ED>"
      )


