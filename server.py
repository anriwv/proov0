from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from keras.models import load_model
from numpy.random import normal
from PIL import Image
import io
import base64
import os

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'Downloads', 'proov0'))
model = load_model('generator_modelKaggle-17.h5')

@app.route('/')
def home():
    return render_template('/index.html')

@app.route('/generate', methods=['POST'])
def generate():
    noise = normal(0, 1, (1, 100))
    output_data = model.predict(noise)

    img = Image.fromarray(output_data[0, :, :, 0] * 255).convert('L')
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)