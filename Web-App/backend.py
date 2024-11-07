from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import librosa
# import the method that does the prediction for our CNN here


# Define your genre labels
GENRE_LABELS = ["Pop", "Classical", "Jazz", "Rock", "Hip Hop"]  # Update with actual labels

def preprocess_audio(file_path):
    # process the audio file into whatever features needed to make a prediction
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)  
    mfccs = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0) 

@app.route('/classify', methods=['POST'])
def classify_genre():
    if 'audioFile' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded file
    file = request.files['audioFile']
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)

    try:
        # Preprocess the audio and get model prediction
        features = preprocess_audio(file_path)
        with torch.no_grad():
            output = model(features)
            
        return jsonify({"genre": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
