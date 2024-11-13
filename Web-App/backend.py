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



app = Flask(__name__)

# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'}

# Model architecture
class MusicNet(nn.Module):
    def __init__(self):
        super(MusicNet, self).__init__()
        # C1: Convolutional Layer (input channels: 1, output channels: 6, kernel size: 3x3)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding='same', groups= 4)
        
        # C2: Convolutional Layer (input channels: 6, output channels: 12, kernel size: 3x3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1, padding='same')
        
        # S3: Subsampling (Max Pooling with kernel size: 2x2 and stride: 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # C4: Convolutional Layer (input channels: 12, output channels: 24, kernel size: 3x3)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=14, kernel_size=3, stride=1, padding='same')

        # C5: Convolutional Layer (input channels: 6, output channels: 16, kernel size: 3x3)
        self.conv4 = nn.Conv2d(in_channels=14, out_channels=16, kernel_size=3, stride=1, padding='same')
        
        # S6: Subsampling (Max Pooling with kernel size: 2x2 and stride: 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # C7: Convolutional Layer (input channels: 48, output channels: 96, kernel size: 3x3)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=18, kernel_size=3, stride=1, padding='same')

        # C8: Convolutional Layer (input channels: 6, output channels: 16, kernel size: 3x3)
        self.conv6 = nn.Conv2d(in_channels=18, out_channels=20, kernel_size=3, stride=1, padding='same')
        
        # S9: Subsampling (Max Pooling with kernel size: 2x2 and stride: 2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(in_channels=20, out_channels=22, kernel_size=3, stride=1, padding='same')

        self.conv8 = nn.Conv2d(in_channels=22, out_channels=24, kernel_size=3, stride=1, padding='same')

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
       
        
        # C10: Fully connected convolutional layer (input size: 192, output size: 120)
        self.fc1 = nn.Linear(in_features=24*34*15, out_features=4000)
        
        # F11: Fully connected layer (input size: 10000, output size: 1000)
        self.fc2 = nn.Linear(in_features=4000, out_features=1000)

        # F12: Fully connected layer (input size: 1000, output size: 100)
        self.fc3 = nn.Linear(in_features=1000, out_features=100)
        
        # Output layer (input size: 100, output size: 10)
        self.fc4 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        # Apply the first convolution and activation function
        # print('x dims', x.shape)
        x = F.relu(self.conv1(x))    # C1
        x = F.relu(self.conv2(x))    # C2
        x = self.pool1(x)            # S3
        x = F.relu(self.conv3(x))    # C4
        x = F.relu(self.conv4(x))    # C5
        x = self.pool2(x)            # S6
        x = F.relu(self.conv5(x))    # C7
        x = F.relu(self.conv6(x))    # C8
        x = self.pool3(x)            # S9
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)
        
        x = x.view(x.shape[0], 24*34*15)       # Flatten before passing to fully connected layers
        # Fully connected layers with activation functions
        x = F.relu(self.fc1(x))      # C10
        x = F.relu(self.fc2(x))      # F11
        x = F.relu(self.fc3(x))      # F12
        # Output layer (no activation function because we will use CrossEntropyLoss which includes Softmax)
        x = self.fc4(x)              # Output layer
        return x

    

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicNet()
model = MusicNet().to(device)
model.load_state_dict(torch.load("C:/Users/olabinmo/CSSE 416/MusicGenreClassifier.pth", map_location=device))
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

    try:
        # Convert to spectrogram
        print("Going to change the audio to a spectogram")
        spectrogram = audio_to_spectrogram(file_path)
        print("Back from creating the spectogram and going to perform prediction")
        
        # Perform prediction

        with torch.no_grad():
            outputs = model(spectrogram)
            _, predicted = torch.max(outputs, 1)

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
