import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model_path = "/home/stemland/cat_dog_model.h5"

if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    exit()

model = load_model(model_path)
print("Model loaded successfully.")

# Prediction function
def predict_image_class(img_path, model, target_size=(150, 150)):
    if not os.path.exists(img_path):
        return f"Error: Image not found -> {img_path}"

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)[0][0]
        print(f"Raw model output: {prediction:.4f}")
        label = "Dog" if prediction > 0.5 else "Cat"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        return f"{label} (Confidence: {confidence:.4f})"
    except Exception as e:
        return f"Error processing image {img_path}: {e}"

# List of test image paths
test_images = [
    "/home/stemland/dataset/test/dog/00854-3846169016.png",
    "/home/stemland/dataset/test/cat/00314-200124624.png",
    "/home/stemland/dataset/test/dog/00865-3846169027.png",
    "/home/stemland/dataset/test/cat/00475-200124785.png",
    "/home/stemland/dataset/test/dog/00874-3846169036.png",
    "/home/stemland/dataset/test/cat/00315-200124625.png"   
    
]

# Run predictions
for img_path in test_images:
    print(f"\nPredicting: {img_path}")
    result = predict_image_class(img_path, model)
    print(result)

