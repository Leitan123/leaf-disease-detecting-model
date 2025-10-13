import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("../model/cashew_classifier.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save it
with open("../model/cashew_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model successfully converted to TensorFlow Lite!")
