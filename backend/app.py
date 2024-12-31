from flask import Flask, render_template, request, jsonify
import os
from backend.model.caption_model import generate_caption
from backend.database.setup_db import save_to_database

app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/Ansh Varshney/Desktop/Image_Captioning/dataset/Images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    caption = generate_caption(file_path)
    save_to_database(file_path, caption)
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)
