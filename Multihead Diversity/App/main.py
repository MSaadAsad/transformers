from flask import Flask, render_template, jsonify, send_from_directory
import os

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    image_dir = 'Images'  # Directly refer to the Images directory
    try:
        sizes = sorted(set(f.split('_')[2] for f in os.listdir(image_dir) if f.startswith('output_size')))
        dropouts = sorted(set(f.split('_')[4].replace('_', '.') for f in os.listdir(image_dir) if f.startswith('output_size')))
    except FileNotFoundError:
        print(f"Failed to read directory: {image_dir}")
        sizes, dropouts = [], []  # Provide defaults in case of an error
    return render_template('index.html', sizes=sizes, dropouts=dropouts)

@app.route('/images/<size>/<dropout>')
def images(size, dropout):
    dropout_formatted = dropout.replace('.', '_')
    folder_name = f'output_size_{size}_dropout_{dropout_formatted}'
    folder_path = os.path.join('Images', folder_name)
    try:
        images = [img for img in os.listdir(folder_path) if img.endswith('.png')]
        images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except FileNotFoundError:
        print(f"Failed to read directory: {folder_path}")
        images = []
    return jsonify(images)

@app.route('/image/<size>/<dropout>/<image>')
def image(size, dropout, image):
    dropout_formatted = dropout.replace('.', '_')
    folder_name = f'output_size_{size}_dropout_{dropout_formatted}'
    folder_path = os.path.join('Images', folder_name)
    return send_from_directory(folder_path, image)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)