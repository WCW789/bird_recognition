import os
import tensorflow as tf
from tensorflow import keras
import requests
import cv2
import csv
import numpy as np
import pandas as pd
import random
from keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV3Large
 
import os
from dotenv import load_dotenv

load_dotenv()

def get_species(url):
    print("url in flask", url)
    r = requests.get(url)
    print("rrrrr", r)
    working_dir=r'./'
    save_location=os.path.join(working_dir, os.environ['IMAGE_NAME'])
    print("save_location", save_location)
    with open(save_location, 'wb') as f:
        f.write(r.content)

    img = cv2.imread(save_location)
    print("img1", img)
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("got modified img")

    model = MobileNetV3Large(weights='imagenet')

    input_img = image.img_to_array(img)
    input_img = np.expand_dims(input_img, axis=0)
    input_img = tf.keras.applications.mobilenet_v3.preprocess_input(input_img)

    predict_img = model.predict(input_img)

    # Get prediction
    top_prediction = tf.keras.applications.mobilenet_v3.decode_predictions(predict_img, top=1)
    print(top_prediction)

    bird_name_top_prediction = top_prediction[0][0][1]
    print(f'Prediction: {bird_name_top_prediction}')

    def convert_name(input_name):
        edited_name = input_name.replace('_', ' ')
        capitalized_name = edited_name.capitalize()

        return capitalized_name
    
    bird_name = convert_name(bird_name_top_prediction)
    return bird_name
