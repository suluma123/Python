import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

# Set the path to the dataset
dataset_path = 'path_to_xray_dataset'

# Data Preprocessing
def load_data(dataset_path):
    data = []
    labels = []
    for label in os.listdir(dataset_path):
        for img in os.listdir(os.path.join(dataset_path, label)):
            img_path = os.path.join(dataset_path, label, img)
            img_array = plt.imread(img_path)
            img_array = np.resize(img_array, (150, 150, 3))  # Resize images to 150x150
            data.append(img_array)
            labels.append(label)
    return np.array(data), np.array(labels)

# Load the dataset
data, labels = load_data(dataset_path)

# Encode labels
labels = pd.get_dummies(labels).values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # 3 classes: Covid-19, Pneumonia, Normal

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(datagen.flow(X_train, y_train, batch_size=32), 
          validation_data=(X_test, y_test), 
          epochs=50, 
          callbacks=[early_stopping])

# Save the model
model.save('covid_detection_model.h5')


# covid detection using x rays with gui buttons
import tkinter as tk
from tkinter import filedialog, messagebox
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# Load the pre-trained model
model = load_model('covid_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to classify the image
def classify_image():
    if not image_path:
        messagebox.showerror("Error", "Please upload an image first.")
        return
    
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    
    if prediction[0][0] > 0.5:
        result = "Covid"
    else:
        result = "Normal"
    
    messagebox.showinfo("Result", f"The image is classified as: {result}")

# Function to browse and upload an image
def upload_image():
    global image_path
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        messagebox.showinfo("Image Selected", "Image uploaded successfully!")

# Initialize the GUI
root = tk.Tk()
root.title("Covid-19 Detection from X-Rays")
root.geometry("400x300")

image_path = ""

# Create buttons
upload_button = tk.Button(root, text="Browse Image", command=upload_image)
upload_button.pack(pady=20)

classify_button = tk.Button(root, text="Classify Image", command=classify_image)
classify_button.pack(pady=20)

# Run the application
root.mainloop()

