from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from pydub import AudioSegment  # To handle mp3 to wav conversion
import scipy.io.wavfile as wav
import scipy.signal as signal
from matplotlib import pyplot as plt
import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch.nn.functional as F
import gc
from pathlib import Path




app = Flask(__name__)

# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'}

# Define the model
# Custom Dataset class
class NumpyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.Zsamples = []
        self.Tsamples = []
        self.Fsamples = []
        
        # Load file paths and labels
        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                if (class_path != path):
                    continue
                for file_name in os.listdir(class_path):
                    if file_name.endswith("Z.npy"):
                        file_path = os.path.join(class_path, file_name)
                        self.Zsamples.append((file_path, label))
                    elif file_name.endswith("T.npy"):
                        file_path = os.path.join(class_path, file_name)
                        self.Tsamples.append((file_path, label))
                    elif file_name.endswith("F.npy"):
                        file_path = os.path.join(class_path, file_name)
                        self.Fsamples.append((file_path, label))
    
    def __len__(self):
        return len(self.Zsamples)
    
    def __getitem__(self, idx):
        file_path, label = self.Zsamples[idx]
        data = np.load(file_path)  # Load numpy array
        real = np.real(data)
        imag = np.imag(data)
        mag = np.abs(data)
        angle = np.angle(data)
        real = np.expand_dims(real, axis=0)
        imag = np.expand_dims(imag, axis=0)
        mag = np.expand_dims(mag, axis=0)
        angle = np.expand_dims(angle, axis=0)
        data = np.concatenate((real, imag, mag, angle), axis=0)
        data = torch.tensor(data, dtype=torch.float32)  # Convert to PyTorch tensor
        
        if self.transform:
            data = self.transform(data)
        
        return data, label

    def spectrograms(self, idx):
        Zfile_path, label = self.Zsamples[idx]
        Tfile_path, label = self.Tsamples[idx]
        Ffile_path, label = self.Fsamples[idx]
        Zdata = np.load(Zfile_path)  # Load numpy array
        Tdata = np.load(Tfile_path)  # Load numpy array
        Fdata = np.load(Ffile_path)  # Load numpy array
        print(Zdata.shape)
        # Create a 2x2 subplot grid
        fig, axes = plt.subplots(5, 1, figsize=(10, 16))
        
        # First subplot
        c1 = axes[0].pcolormesh(Tdata, Fdata, np.log(np.abs(Zdata)), cmap='gnuplot')
        fig.colorbar(c1, ax=axes[0])
        axes[0].set_title("Spectrogram Magnitude")
        
        # Second subplot
        c2 = axes[1].pcolormesh(Tdata, Fdata, np.angle(Zdata), cmap='gnuplot')
        fig.colorbar(c2, ax=axes[1])
        axes[1].set_title("Spectrogram Angle")      

        # Third subplot
        c3 = axes[2].pcolormesh(Tdata, Fdata, np.log(np.square(np.real(Zdata))), cmap='gnuplot')
        fig.colorbar(c3, ax=axes[2])
        axes[2].set_title("Spectrogram Real")  

        # Fourth subplot
        c4 = axes[3].pcolormesh(Tdata, Fdata, np.log(np.square(np.imag(Zdata))), cmap='gnuplot')
        fig.colorbar(c4, ax=axes[3])
        axes[3].set_title("Spectrogram Imag")  

        # Fifth subplot
        c5 = axes[4].pcolormesh(Tdata, Fdata, np.log(np.square(np.imag(Zdata)) + np.square(np.real(Zdata))), cmap='gnuplot')
        fig.colorbar(c4, ax=axes[4])
        axes[4].set_title("Spectrogram Imag + Real") 




class MusicNet(nn.Module):

    def __init__(self):
        super(MusicNet, self).__init__()

        self.cLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding='same', groups=4),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.cLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.cLayer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.cLayer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fLayer = nn.Sequential(
            nn.Linear(in_features=512*34*15, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x, return_features=False):
        x = self.cLayer1(x)
        x = self.cLayer2(x)
        x = self.cLayer3(x)
        x = self.cLayer4(x)
        
        x = x.view(x.shape[0], 512*34*15)       # Flatten before passing to fully connected layers

        if not return_features:
            x = self.fLayer(x)
             
        return x

    

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicNet()
model = MusicNet().to(device)
model.load_state_dict(torch.load("C:/Users/olabinmo/CSSE 416/MusicGenreClassifier 1.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# Function to convert mp3 to wav if necessary
# def convert_to_wav(filepath, target_sample_rate=22050):
#     if filepath.endswith(".mp3"):
#         audio = AudioSegment.from_mp3(filepath)
#         wav_path = filepath.rsplit(".", 1)[0] + ".wav"  # Change file extension to .wav
#         audio = audio.set_frame_rate(target_sample_rate).set_channels(1)  # Set sample rate and mono
#         audio.export(wav_path, format="wav")
#         os.remove(filepath)  # Remove the original mp3 file
#         return wav_path
#     return filepath

# Function to convert audio to spectrogram
def audio_to_spectrogram(filepath, sample_rate=22050):
    # Ensure file is in .wav format
    # filepath = convert_to_wav(filepath, sample_rate)
    
    # Read audio file
    print("Going to read the file path: " + filepath)
    sr, samples = wavfile.read(filepath)
    if sr != sample_rate:
        raise ValueError(f"Expected sample rate {sample_rate}, but got {sr}")

    # Mono conversion
    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)

    # Convert audio to spectrogram
    _, _, Zxx = stft(samples, fs=sample_rate, nperseg=sample_rate // 20, noverlap=(sample_rate // 20) // 2)
    Zxx = np.abs(Zxx)  # Take the magnitude
    Zxx = np.expand_dims(Zxx, axis=0)  # Add channel dimension

    # Convert to PyTorch tensor
    print("Done reading the file")
    return torch.tensor(Zxx, dtype=torch.float32).unsqueeze(0).to(device)

@app.route('/classify', methods=['POST'])
def classify_genre():
    if 'audioFile' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Save the uploaded file
    file = request.files['audioFile']
     # Save the file to the server
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
           
    print(file_path)

    transform = transforms.Compose([
    transforms.CenterCrop((544, 240)),
    transforms.Normalize((0,), (0.5,))
])
    
    music_dataset = NumpyDataset(root_dir=root_dir, transform=transform)

    print(len(music_dataset))

    music_loader = DataLoader(dataset=music_dataset, batch_size=1, shuffle=False)

    music_classifier(model, music_loader)

    try:
        # Convert to spectrogram
        print("Going to change the audio to a spectogram")
        spectrogram = audio_to_spectrogram(file_path)
        print("Back from creating the spectogram and going to perform prediction")
        
        # Perform prediction

        with torch.no_grad():
            image, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

        print("Done with prediction " + predicted)
        # Map prediction to genre label
        genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        genre = genres[predicted.item()]

        # Clean up uploaded file
        os.remove(file_path)

        # Return the genre as JSON
        return jsonify({"genre": genre})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
