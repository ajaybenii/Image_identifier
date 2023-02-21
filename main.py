import io
import os

import requests

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
import numpy as np
from fastapi import FastAPI
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from urllib.parse import urlparse
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL


app = FastAPI()


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


# Load the model
model_path = "keras_Model.h5"
m_path = os.path.basename(model_path)

model = load_model(m_path, compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)



@app.get('/')
def index():
    """Sample Function"""
    return("Hi User!")


@app.post("/image_identifier_by_file")
async def insert_image_by_file(insert_image: UploadFile=File(...)):

    contents = await insert_image.read() #Building image
    input_image = Image.open(BytesIO(contents)).convert("RGB")
    input_image.save("input_img.jpg")
    img_path = "input_img.jpg"

    # img_path = '8.jpg'

    # Replace this with the path to your image
    image = Image.open(img_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

    # obj_coordinates = class_name[2:]
    if class_name[2:] == "Not_valid\n":

        obj_coordinates = "Not_Valid"
        score = confidence_score*100

    else:
        obj_coordinates = "Valid"
        score = confidence_score*100
    
    return {"response" : obj_coordinates,
            "confidence_score": score}



@app.post("/image_identifier_by_url")
async def insert_image_by_url(insert_image:str):
    
    try:
        response = requests.get(insert_image)

        if response.status_code == 200:

            image_bytes = io.BytesIO(response.content)
            input_image = Image.open(image_bytes).convert("RGB")
            input_image.save("input_img.jpg")
            img_path = "input_img.jpg"

            image = Image.open(img_path).convert("RGB")

            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # Predicts the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Print prediction and confidence score
            # print("Class:", class_name[2:], end="")
            # print("Confidence Score:", confidence_score)
            # obj_coordinates = class_name[2:]

            if class_name[2:] == "Not_valid\n":

                obj_coordinates = "Not_Valid"
                score = confidence_score*100

            else:
                obj_coordinates = "Valid"
                score = confidence_score*100

            return {"response" : obj_coordinates,
                    "confidence_score": score}
            
        else:
            return{"Invalid image url"}
    except:
        return{"Invalid image url"}
