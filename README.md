# CSNN-based-Image-Classification-Model
I would like you to develop

1) A CNN and VCG16-based image classifier that would give us how likely a person has a Heart disease

2) The Heart diseases can be Angina pectoris, Hypotension, Coronary Artery Disease, and Cardiovascular disease, or any other disease whose dataset of echocardiograms is available. A dataset of around a 1000 images per disease would be ideal

3)The dataset used should be Echocardiograms of the diseases, from which the CNN algorithm will extract features

4)we need a trained model something that will readily integrate with a website. And we may require your help in integrating it to our website

5)we need to know the diseases and their respective accuracy.

6) We would likely need the following things from you: the dataset of echocardiograms, the file wherein you built the model, the exported model along with its accuracy scores for each diseases
--------------------------------
To address the problem of creating a CNN (Convolutional Neural Network) based image classifier using VGG16 and classifying various heart diseases from echocardiogram images, I will outline the steps and provide a Python script to build, train, and save the model. We will use the Keras deep learning library (with TensorFlow backend) for the implementation.

Here’s a step-by-step breakdown of the solution:
Steps:

    Dataset Collection: You need to collect echocardiogram images for different heart diseases like Angina pectoris, Hypotension, Coronary Artery Disease, Cardiovascular Disease, etc. These images should be labeled appropriately (with disease names) and grouped into subdirectories for each disease.

    Since you mentioned a dataset of around 1000 images per disease, we will assume that you have the dataset organized in a directory structure like:

    dataset/
      ├── angina/
      ├── hypotension/
      ├── coronary_artery_disease/
      ├── cardiovascular_disease/
      └── ...

    Preprocessing the Data: We will load the images, resize them to the required size (usually 224x224 for VGG16), normalize the pixel values, and split the data into training and validation sets.

    VGG16 Model Integration: We will use VGG16 as a feature extractor (using its pre-trained weights) and add custom layers for classification.

    Model Training: We'll train the model using the training set and validate it using the validation set.

    Model Evaluation: We'll evaluate the accuracy of the model for each disease category.

    Export the Model: Save the trained model to disk so it can be integrated into a website.

Python Code Implementation:

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Set paths
dataset_path = 'path_to_dataset_directory'
train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'val')

# Image size and batch size
img_size = (224, 224)  # VGG16 expects 224x224 images
batch_size = 32

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

# Loading datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')

# Load VGG16 Model pre-trained on ImageNet and use it as a feature extractor
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of VGG16 to retain pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

# Build the CNN Model using VGG16 as base
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())  # Pooling layer to reduce dimensions
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))  # Number of diseases

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size)

# Save the trained model
model.save('heart_disease_classifier_model.h5')

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Accuracy: {val_accuracy*100:.2f}%')

# Generate the classification report for disease-wise accuracy
y_pred = np.argmax(model.predict(val_generator), axis=1)
y_true = val_generator.classes
target_names = list(train_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

Explanation of the Code:

    Dataset Loading:
        The dataset should be organized in a directory structure where each disease has its own folder (e.g., 'angina', 'hypotension', etc.).
        The ImageDataGenerator is used for real-time data augmentation, which helps in improving the generalization ability of the model.

    Model Architecture:
        VGG16 is used as the base model for feature extraction. The layers of the pre-trained VGG16 are frozen, so they are not retrained, and only the new layers are trained.
        GlobalAveragePooling2D: This reduces the spatial dimensions (height and width) of the output from the convolutional layers.
        Fully Connected Layer (Dense): Added a dense layer with 1024 units followed by a dropout layer for regularization.
        Output Layer: The final layer uses a softmax activation function with the number of output neurons equal to the number of diseases (based on the number of directories in the training set).

    Model Training:
        The model is trained using the adam optimizer and categorical_crossentropy loss function since this is a multi-class classification task.

    Evaluation:
        After training, we evaluate the model on the validation data and print the classification report that shows accuracy and other metrics (precision, recall, F1-score) for each disease.

    Saving the Model:
        The trained model is saved as heart_disease_classifier_model.h5, which can be later used for integration with a website.

Integration with Website:

To integrate the trained model with your website, you'll need to deploy it on a server. Here's a basic outline for integrating the model using a Flask API:

    Create a Flask API that loads the model and serves predictions.
    Upload Images: The website can upload echocardiogram images to the Flask server, which will preprocess the image, feed it to the model, and return predictions (heart disease likelihood).

Flask API Example:

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('heart_disease_classifier_model.h5')

# Load class labels
class_labels = ['angina', 'hypotension', 'coronary_artery_disease', 'cardiovascular_disease']  # Update based on actual labels

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image
    
    # Predict the disease
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    
    return jsonify({
        'disease': class_labels[predicted_class[0]],
        'accuracy': float(np.max(prediction) * 100)
    })

if __name__ == '__main__':
    app.run(debug=True)

This Flask API will accept POST requests with an echocardiogram image, preprocess the image, predict the disease using the trained model, and return the result as a JSON response.
Final Deliverables:

    Dataset: A collection of echocardiogram images (1000 images per disease, appropriately labeled).
    Model File: The saved model file (heart_disease_classifier_model.h5).
    Model Evaluation: Accuracy scores and classification report for each disease.
    Flask API: Python code to integrate the model with your website.

This approach provides a robust framework for classifying heart diseases from echocardiogram images using deep learning, and it's ready to integrate with your website or application. Let me know if you need further adjustments or assistance!
