import random
import bird_recognition


from flask import Flask, jsonify, request

app = Flask(__name__)

bird_images = ['https://cdn.download.ams.birds.cornell.edu/api/v1/asset/215836881/1800', 'https://cdn.download.ams.birds.cornell.edu/api/v1/asset/263736021/1800', 'https://cdn.download.ams.birds.cornell.edu/api/v1/asset/297949771/1800', 'https://cdn.download.ams.birds.cornell.edu/api/v1/asset/115691751/1800', 'https://cdn.download.ams.birds.cornell.edu/api/v1/asset/275462541/1800']
url = random.choice(bird_images)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/bird', methods=['POST'])
def bird_images(): 
    if request.method == 'POST':
        species_name = bird_recognition.get_species(url)
        print(f'This is the species name! {species_name}')
        return species_name
    
