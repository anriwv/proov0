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


def save_image(image):
    save_path = os.path.join(os.getcwd(), 'Downloads', 'generated_image.png')
    image.save(save_path)
    
    
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
    
    img.save("generated_image.png")
    return img_str

@app.route('/save', methods=['POST'])
def save():
    img_data = request.form['image_data']
    img_data = img_data.replace("data:image/png;base64,", "")
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes))
    save_image(img)
    
    return "Image saved successfully!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
