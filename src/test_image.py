import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("model/cashew_classifier.h5")

# Define your class names (same order used in training)
class_names = ["Anthracnose", "Healthy"]

# Path to the folder containing test images
test_folder = "dataset/test_images"

# Loop through each image in the folder
for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction[0])           # Index of the class with highest probability
    predicted_class = class_names[pred_index]      # Map index to class name
    confidence = prediction[0][pred_index]         # Confidence of predicted class

    print(f"🖼️ {img_name} → {predicted_class} ({confidence * 100:.2f}% confidence)")
