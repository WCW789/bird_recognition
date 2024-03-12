import os
import tensorflow as tf
from tensorflow import keras
import requests
import cv2
import csv
import numpy as np
import pandas as pd
import random
 
import os
from dotenv import load_dotenv

load_dotenv()

def get_species(url):
    r = requests.get(url)
    working_dir=r'./'
    save_location=os.path.join(working_dir, os.environ['IMAGE_NAME'])
    with open(save_location, 'wb') as f:
        f.write(r.content)

    img = cv2.imread(save_location)
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get answer
    with open(os.environ['BIRD_PATH'], "r") as file:
        df1 = csv.reader(file)

        for row in df1:
            if row[0] == url:
                bird_name = row[1]
                print(f"Actual Bird: {bird_name}")


    img_exp = np.expand_dims(img, axis=0)
    model_path = os.environ['MODEL_PATH']

    model_loaded = keras.models.load_model(model_path, custom_objects={'F1_score':'F1_score'})
    prediction_bird = model_loaded.predict(img_exp)

    index = np.argmax(prediction_bird)
    print (f'The predict class index is {index}')

    # Get prediction
    csvpath2 = os.environ['BIRD_PATH']
    df2 = pd.read_csv(csvpath2)

    klass_filter = df2['class id'] == index
    index_df = df2[klass_filter]
    row_0 = index_df.iloc[0]
    species = row_0['labels']
    print(f'Prediction: {species}')
    return species
