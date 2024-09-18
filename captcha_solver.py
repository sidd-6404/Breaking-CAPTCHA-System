import streamlit as st
from keras.models import load_model
from helpers import resize_to_fit
import numpy as np
import cv2
import pickle
import pathlib

# Load the model and labels
MODEL_FILENAME = pathlib.Path("D:\solving_captchas_code_examples\captcha_model.hdf5")
#MODEL_LABELS_FILENAME = "model_labels.dat"
MODEL_LABELS_FILENAME = pathlib.Path('D:\solving_captchas_code_examples\model_labels.dat')

with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

model = load_model(MODEL_FILENAME)

# Streamlit UI
st.title("CAPTCHA Solver")
st.write("Upload a CAPTCHA image and let the model predict the text.")

uploaded_file = st.file_uploader("Choose a CAPTCHA image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    image_padded = cv2.copyMakeBorder(image_gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # Threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(image_padded, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letter_image_regions = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters, skip this image
    if len(letter_image_regions) != 4:
        st.error("The CAPTCHA image couldn't be processed correctly.")
    else:
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        predictions = []

        for letter_bounding_box in letter_image_regions:
            x, y, w, h = letter_bounding_box
            letter_image = image_padded[y - 2:y + h + 2, x - 2:x + w + 2]
            letter_image = resize_to_fit(letter_image, 20, 20)
            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)
            prediction = model.predict(letter_image)
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)

        captcha_text = "".join(predictions)
        st.success(f"CAPTCHA text is: {captcha_text}")

        # Display the uploaded image and the result
        st.image(image, caption='Uploaded CAPTCHA', use_column_width=True)