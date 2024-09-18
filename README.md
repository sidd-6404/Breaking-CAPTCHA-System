#CAPTCHA Solver
This repository contains a solution for recognizing CAPTCHA text using a Convolutional Neural Network (CNN) model built with Keras. The project includes the following components:
CAPTCHA Image Processing: Code to load CAPTCHA images, preprocess them (grayscale conversion, padding, thresholding), and extract individual letter regions.
Model Training: A neural network model is trained using extracted letter images to classify each letter/number in the CAPTCHA. The CNN model is built with two convolutional layers, followed by max pooling and a fully connected layer.
CAPTCHA Solver App: A Streamlit-based web app where users can upload CAPTCHA images, and the trained model predicts the CAPTCHA text.

#Files
captcha_solver.py - Contains the Streamlit-based CAPTCHA solver app and the model loading/prediction logic​(captcha_solver).
extract_single_letters_from_captchas.py - Extracts individual letters from CAPTCHA images and prepares training data by saving letter images to directories​(extract_single_letters_…).
helpers.py - A helper function to resize images to a given size while maintaining aspect ratio​(helpers).

#Usage
Step 1: Extract single letters from CAPTCHA images

Run:

python3 extract_single_letters_from_captchas.py

The results will be stored in the "extracted_letter_images" folder.


Step 2: Train the neural network to recognize single letters

Run:

python3 train_model.py

This will write out "captcha_model.hdf5" and "model_labels.dat"


Step 3: Use the model to solve CAPTCHAs!

Run: 

python3 captcha_solver.py
