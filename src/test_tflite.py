import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model/cashew_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names
class_names = ["Anthracnose", "Healthy"]

# Path to test images
test_folder = "dataset/test_images"

# Loop through each image
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    img_array = img_array.astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get predictions
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = np.max(output_data) * 100

    print(f"🖼️ {img_name} → {class_names[predicted_index]} ({confidence:.2f}% confidence)")
