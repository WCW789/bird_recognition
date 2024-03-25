from flask import Flask, jsonify, request
import bird_recognition
import json
import os

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/bird', methods=['POST'])
def bird_images(): 
    if request.method == 'POST':
        data = request.get_json()

        if 'url' in data:            
            url = json.loads(data)['url']
    
        species_name = bird_recognition.get_species(url)
        return f'This is a {species_name}'

    else:
        return jsonify({'error': 'Invalid request.'}), 400
        

    
