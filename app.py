from flask import Flask, request, jsonify
import json
import cv2
import numpy as np
from bubble_sheet_processing import process_bubble_sheet

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/process', methods=['POST'])
def process_image():
    # Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    # Get all files from the request
    files = request.files.getlist('file')

    # Check if the user submits any empty files
    if not files or any(file.filename == '' for file in files):
        return jsonify({'error': 'No selected files'}), 400
    
    # Extract the answer key from form data
    answer_key = request.form.get('answer_key')
    if not answer_key:
        return jsonify({'error': 'Answer key not provided'}), 400

    # Convert answer_key from JSON string to Python dictionary
    try:
        print(answer_key)
        answer_key = json.loads(answer_key)
        # Convert string keys to integers
        answer_key = {int(k): v for k, v in answer_key.items()}
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid answer key format'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid key format'}), 400

    already_in_frame_str = request.form.get('already_in_frame', 'false')
    already_in_frame = already_in_frame_str.lower() == 'true'
    # Process the image
    combined_results = []

    # Process each file
    for file in files:
        result = process_bubble_sheet(file, answer_key, already_in_frame)
        combined_results.append(result)

    # Return the combined results
    return jsonify(combined_results)

if __name__ == '__main__':
    app.run(debug=True)
