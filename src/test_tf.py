import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPU available:", gpus)
