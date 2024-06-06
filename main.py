# Import necessary libraries
import os
import numpy as np
from keras._tf_keras.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2


model = load_model('FinalModel.keras')
test_dir = 'dataset/Testing'


class_names = sorted(os.listdir(test_dir))
print(f"Class names: {class_names}")

# Function to preprocess the image and convert grayscale to RGB
def preprocess_image(img_path):
    # Load the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to the target size
    img = cv2.resize(img, (150, 150))
    # Normalize the image
    img = img / 255.0
    # Convert grayscale to RGB by stacking the single channel 3 times
    img = np.stack((img,)*3, axis=-1)
    # Expand dimensions to match batch size
    img_array = np.expand_dims(img, axis=0)
    return img_array
# Function to map image file name to actual class

# Loop through the images in the test directory
# Initialize counters for accuracy calculation
total_images = 0
correct_predictions = 0

for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)
    for image_name in os.listdir(class_path):
        img_path = os.path.join(class_path, image_name)
        img_array = preprocess_image(img_path)
        
        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        
        # Print the raw prediction
        print(f"Raw prediction for {image_name}: is {predicted_class_name}")
        
     
        actual_class_name = None
        if "no" in image_name:
            actual_class_name = 'notumor'
        elif "pi" in image_name or "piTr" in image_name:
            actual_class_name = 'pituitary'
        elif "gl" in image_name:
            actual_class_name = 'glioma'
        elif "me" in image_name:
            actual_class_name = 'meningioma'
        
        # Update counters for accuracy calculation
        if predicted_class_name == actual_class_name:
            correct_predictions += 1
        total_images += 1

# Calculate and print accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"Accuracy: {accuracy:.2f}%")

