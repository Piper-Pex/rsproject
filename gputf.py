import tensorflow as tf
print("GPU可用:", tf.config.list_physical_devices('GPU'))
print("TensorFlow版本:", tf.__version__)