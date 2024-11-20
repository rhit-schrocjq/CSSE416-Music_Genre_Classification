from flask import Flask, request, jsonify
import torchaudio
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment  # To handle mp3 to wav conversion
import scipy.signal as signal
import os
import numpy as np
import torch.nn.functional as F
import shutil
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from panns_inference import AudioTagging
import librosa
from flask_cors import CORS
from scipy.fft import fft


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'}

NPY_FOLDER = 'npy/'
app.config['NPY_FOLDER'] = NPY_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'npy'}

class AudioDataset(Dataset):
    def __init__(self, dir, sample_rate=16000, segment_length=5*16000):
        """
        Args:
            root_dir (str): Root directory with genre folders containing audio files.
            sample_rate (int): Target sample rate for audio files.
            segment_length (int): Length of each audio segment in samples (6 seconds).
            set_type (str): Specify "train" for training set and "test" for testing set.
            train_ratio (float): The ratio of the data to use for training (e.g., 0.2 for 20% train, 80% test).
        """
        self.dir = dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length  # Length of each segment (10 seconds)

    
    def __len__(self):
        return len(self.file_paths) * 6  # Each file is split into 6 segments
    
    def load_audio_with_fallback(self, file_path):
        """
        Load audio file using torchaudio, and fall back to librosa if needed.
        """
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)
        return waveform

    def __getitem__(self, idx):
        """
        Get a 5-second audio segment and its label from the dataset.

        Args:
            idx (int): Index of the audio segment to retrieve.

        Returns:
            tuple: A tuple containing the Fourier-transformed waveform segment and its corresponding label.
        """        
        segment_idx = idx % 6     # Segment within the file (for 5-second clips)

        # Load the waveform
        audio_path = self.dir
        waveform = self.load_audio_with_fallback(audio_path)
        if waveform is None:
            print(f"Skipping {audio_path} due to load failure.")
            return None

        # Ensure mono audio
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Calculate start and end for each 5-second segment
        start = segment_idx * self.segment_length
        end = start + self.segment_length
        waveform_segment = waveform[:, start:end]

        # If segment is shorter than segment_length, pad with zeros
        num_samples = waveform_segment.size(1)
        if num_samples < self.segment_length:
            padding = self.segment_length - num_samples
            waveform_segment = torch.nn.functional.pad(waveform_segment, (0, padding))

        # Perform Fourier Transform on the segment and keep only the first half
        waveform_np = waveform_segment.numpy()  # Convert tensor to numpy array
        fourier_transform = np.abs(fft(waveform_np[0]))[:self.segment_length // 2]  # Only the first half


        return torch.tensor(fourier_transform, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    
# Define another 1D CNN model for Fourier-transformed audio data
class AudioCNN(nn.Module):
    def __init__(self, input_length=5*16000 // 2):
        super(AudioCNN, self).__init__()
        
        # First convolutional layer and pooling
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second convolutional layer and pooling
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Third convolutional layer and pooling
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fourth convolutional layer and pooling
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fifth convolutional layer and pooling
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the flattened size after the last pooling layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)  # Batch size 1, 1 channel, input length
            out = F.relu(self.conv1(dummy_input))
            out = self.pool1(out)
            out = F.relu(self.conv2(out))
            out = self.pool2(out)
            out = F.relu(self.conv3(out))
            out = self.pool3(out)
            out = F.relu(self.conv4(out))
            out = self.pool4(out)
            out = F.relu(self.conv5(out))
            out = self.pool5(out)
            self.flattened_size = out.shape[1] * out.shape[2]  # Channels * Width after conv/pooling layers

        # Define fully connected layers using the computed flattened size
        self.fc1 = nn.Linear(in_features=self.flattened_size, out_features=3200)
        self.fc2 = nn.Linear(in_features=3200, out_features=1600)
        self.fc3 = nn.Linear(in_features=1600, out_features=800)
        self.fc4 = nn.Linear(in_features=800, out_features=10)

    def forward(self, x):
        # Pass through convolutional and pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        # Flatten the tensor to match the input of the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))    # Pass through first fully connected layer
        x = F.relu(self.fc2(x))    # Pass through second fully connected layer
        x = F.relu(self.fc3(x))    # Pass through third fully connected layer
        x = self.fc4(x)            # Output layer
        return x    
    
# Evaluation loop to calculate accuracy
def evaluate_single_file(model, state_dict, file_path, classes, segment_length=5*16000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(state_dict, weights_only=False, map_location=device))  # Use map_location for device compatibility
    model.eval()  # Set model to evaluation mode
    model.eval()  # Set the model to evaluation mode
    prediction_list = []
    print(f"Evaluating file: {file_path}")
    
    dataset = AudioDataset(file_path, segment_length=segment_length)
    waveform = dataset.load_audio_with_fallback(file_path)

    # Ensure mono and process segments
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    num_segments = waveform.size(1) // segment_length

    # Calculate the number of segments
    num_segments = (waveform.size(1) + segment_length - 1) // segment_length  # Round up for last segment

    with torch.no_grad():  # Disable gradient calculation
        for segment_idx in range(num_segments):
            start = segment_idx * segment_length
            end = start + segment_length
            waveform_segment = waveform[:, start:end]
            
            # Pad if necessary
            if waveform_segment.size(1) < segment_length:
                padding = segment_length - waveform_segment.size(1)
                waveform_segment = torch.nn.functional.pad(waveform_segment, (0, padding))
            
            # Perform Fourier Transform
            waveform_np = waveform_segment.numpy()
            fourier_transform = np.abs(fft(waveform_np[0]))[:segment_length // 2]
            
            # Convert to tensor
            input_tensor = torch.tensor(fourier_transform, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Forward pass
            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)
            
            # Print prediction for the segment
            print(f"Predicted Genre: {classes[predicted.item()]}")
            prediction_list.append(classes[predicted.item()])

    return prediction_list
class NumpyDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.Zsamples = []
        
        # Load file paths and labels
        if os.path.isdir(dir):
            for file_name in os.listdir(dir):
                if file_name.endswith("Z.npy"):
                    file_path = os.path.join(dir, file_name)
                    self.Zsamples.append(file_path)

    
    def __len__(self):
        return len(self.Zsamples)
    
    def __getitem__(self, idx):
        file_path = self.Zsamples[idx]
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
        
        return data

# Custom Instrment Dataset class
class InstrumentDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.Isamples = []
        
        # Load file paths and labels
        if os.path.isdir(dir):
            for file_name in os.listdir(dir):
                if file_name.endswith("I.npy"):
                        file_path = os.path.join(dir, file_name)
                        self.Isamples.append(file_path)
    
    def __len__(self):
        return len(self.Isamples)
    
    def __getitem__(self, idx):
        file_path = self.Isamples[idx]
        data = np.load(file_path)  # Load numpy array
        data = torch.tensor(data, dtype=torch.float32)  # Convert to PyTorch tensor
        
        return data
    

        
class SimpleDataset(Dataset):
    def __init__(self, X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        data = self.X[idx]
        data = torch.tensor(data, dtype=torch.float32)  # Convert to PyTorch tensor
        return data
    

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

class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()

        self.fLayer = nn.Sequential(
            nn.Linear(in_features=263168, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.fLayer(x)   
        return x

# Function to convert audio to spectrograms (sample_rate=22050, clip_length=6*22050+1=132301)
def audio_to_spectrograms2D(filepath, savepath, sample_rate=22050, clip_length=132301):
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

    # create 6sec samples
    clip_samples=[]
    start = 0
    end = start + clip_length

    while (end < len(samples)):
        clip_samples.append(samples[start: end])
        start = end
        end = end + clip_length

    i=0
    for sample in clip_samples:
        SFT = signal.ShortTimeFFT.from_window(win_param='tukey', 
                                            fs=sample_rate, 
                                            nperseg=sample_rate//20,      #make 20Hz minimum sampled frequency
                                            noverlap=(sample_rate//20)//2,  #50% overlap
                                            fft_mode='onesided', 
                                            scale_to='magnitude', 
                                            phase_shift=None,
                                            symmetric_win=True)
        Zxx = SFT.stft(sample)
        np.save(savepath + "sample_" + str(i) + "_Z.npy", Zxx)
        i+=1        


# Function to convert audio to spectrograms (sample_rate=22050, clip_length=6*22050+1=132301)
def audio_to_spectrograms_mixed(filepath, savepath, sample_rate=22050, music_clip_length=132301, instrument_clip_length=192001):
    # Ensure file is in .wav format
    # filepath = convert_to_wav(filepath, sample_rate)
    
    # Read audio file
    print("Going to read the file path: " + filepath)
    #spectrogram sampling
    music_sr, music_samples = wavfile.read(filepath)
    if music_sr != sample_rate:
        raise ValueError(f"Expected sample rate {sample_rate}, but got {music_sr}")
    # Mono conversion
    if len(music_samples.shape) > 1:
        music_samples = np.mean(music_samples, axis=1)

    #insturment sampling
    instrument_samples, instrument_sr = librosa.load(filepath, sr=32000, mono=True)

    # music clip samples
    music_clip_samples=[]
    start = 0
    end = start + music_clip_length

    while (end < len(music_samples)):
        music_clip_samples.append(music_samples[start: end])
        start = end
        end = end + music_clip_length

    # insturment clip samples    
    instrument_clip_samples = []
    start = 0
    end = start + instrument_clip_length
    
    while (end < len(instrument_samples)):
        instrument_clip_samples.append(instrument_samples[start: end])
        start = end
        end = end + instrument_clip_length

    i=0
    for sample in music_clip_samples:
        SFT = signal.ShortTimeFFT.from_window(win_param='tukey', 
                                            fs=sample_rate, 
                                            nperseg=sample_rate//20,      #make 20Hz minimum sampled frequency
                                            noverlap=(sample_rate//20)//2,  #50% overlap
                                            fft_mode='onesided', 
                                            scale_to='magnitude', 
                                            phase_shift=None,
                                            symmetric_win=True)
        Zxx = SFT.stft(sample)
        Ix = instrument_clip_samples[i]
        np.save(savepath + "sample_" + str(i) + "_Z.npy", Zxx)
        np.save(savepath + "sample_" + str(i) + "_I.npy", Ix)
        i+=1


# Function to convert mp3 to wav if necessary
def convert_mp3_to_wav(input_file: str, output_file: str, sample_rate: int = 22050):
    try: 
        audio = AudioSegment.from_mp3(input_file) # Load the MP3 file
        audio = audio.set_frame_rate(sample_rate) # Set the desired sample rate
        audio.export(output_file, format="wav") # Export as WAV
        
        print(f"Conversion successful! File saved to: {output_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")

def music_features(model, state_dict, loader, save_path):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(state_dict, weights_only=False, map_location=device))  # Use map_location for device compatibility
    model.eval()  # Set model to evaluation mode
    # Dictionary to store features and labels
    features = []

    # Extract features
    with torch.no_grad():  # We don't need gradients for this task
        for images in loader:
            images = images.to(device)
            
            # Extract features from the images
            # The VGG16 model outputs feature maps, which we can treat as image features
            outputs = model(images, return_features=True)
            
            # Flatten the outputs to a 1D vector
            outputs = outputs.view(outputs.size(0), -1)
            
            # Append features and labels to lists
            features.append(outputs.cpu().numpy())

    print("Model Done")
    
    features = np.concatenate(features, axis=0)
    np.savez(save_path, features=features)
    
    print("Features saved")

def instrument_features(loader, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    audio_tagging = AudioTagging(checkpoint_path=None, device=device)
    # Dictionary to store features and labels
    features = []

    # Extract features
    for audio in loader:
       
        # Extract features for the batch
        _, embedding = audio_tagging.inference(audio)
    
        # Flatten the outputs to a 1D vector
        outputs = np.array(embedding.flatten())
        
        # Append features and labels to lists
        features.append(outputs)
    
    print("Model Done")
    
    np.savez(save_path, features=features)
    
    print("Features saved")

def concatenate_features(npz1, npz2):
    data1 = np.load(npz1)
    data2 = np.load(npz2)

    features = np.concatenate((data1['features'], data2['features']), axis=1)
    print(features.shape)

    return features


# Testing loop
def music_classifier(model, state_dict, loader, classes):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(state_dict, weights_only=False, map_location=device))  # Use map_location for device compatibility
    model.eval()  # Set model to evaluation mode
    prediction_list = []

    with torch.no_grad():
        for tensors in loader:  # `tensors` is a batch of inputs
            tensors = tensors.to(device)
            outputs = model(tensors)
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities

            # Get predicted classes and their confidences
            predicted_classes = torch.argmax(probabilities, dim=1)  # Tensor of predicted class indices
            confidences = probabilities.max(dim=1).values  # Tensor of maximum probabilities (confidence scores)

            # Print and collect predictions
            for pred, confidence in zip(predicted_classes, confidences):
                prediction_list.append([classes[pred.item()], confidence.item()])
                print(f"Predicted {classes[pred.item()]:10} with {confidence.item()*100:.2f}% confidence")

    return prediction_list


transform = transforms.Compose([
    transforms.CenterCrop((544, 240)),
    transforms.Normalize((0,), (0.5,))
])

@app.route('/classify_1D', methods=['POST'])
def classify_1D():
    if 'audioFile' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    #clear folders
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'])

    if os.path.exists(app.config['NPY_FOLDER']):
        shutil.rmtree(app.config['NPY_FOLDER'])
    os.makedirs(app.config['NPY_FOLDER'])

    # Save the uploaded file
    file = request.files['audioFile']
     # Save the file to the server
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
           
    print(file_path)
    if (file_path.endswith('.mp3')):
        convert_mp3_to_wav(file_path, file_path[:-4] + ".wav")
        file_path = file_path[:-4] + ".wav"
    elif (not file_path.endswith(".wav")):
        return jsonify({"error": "Incompatible file type"}), 401

    # Perform prediction
    classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    predictions = evaluate_single_file(AudioCNN(), "MusicGenreClassifier1D.pth", file_path, classes)

    # Clean up uploaded file
    os.remove(file_path)

    # Return the genre as JSON
    return jsonify({"predictions":predictions})
    #except Exception as e:
    #    return jsonify({"error": str(e)}), 500


@app.route('/classify_2D', methods=['POST'])
def classify_2D():
    if 'audioFile' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    #clear folders
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'])

    if os.path.exists(app.config['NPY_FOLDER']):
        shutil.rmtree(app.config['NPY_FOLDER'])
    os.makedirs(app.config['NPY_FOLDER'])

    # Save the uploaded file
    file = request.files['audioFile']
     # Save the file to the server
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
           
    print(file_path)
    if (file_path.endswith('.mp3')):
        convert_mp3_to_wav(file_path, file_path[:-4] + ".wav")
        file_path = file_path[:-4] + ".wav"
    elif (not file_path.endswith(".wav")):
        return jsonify({"error": "Incompatible file type"}), 401

    #try:
    # Convert to spectrogram
    print("Going to change the audio to a spectogram")
    audio_to_spectrograms2D(file_path, app.config['NPY_FOLDER'])
    print("Back from creating the spectogram and going to perform prediction")

    music_dataset = NumpyDataset(dir=app.config['NPY_FOLDER'], transform=transform)
    music_loader = DataLoader(dataset=music_dataset, batch_size=1, shuffle=False)
    
    # Perform prediction
    classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    predictions = music_classifier(MusicNet(), "MusicGenreClassifier2D.pth", music_loader, classes)

    # Clean up uploaded file
    os.remove(file_path)

    # Return the genre as JSON
    return jsonify({"predictions":predictions})
    #except Exception as e:
    #    return jsonify({"error": str(e)}), 500


@app.route('/classify_mixed', methods=['POST'])
def classify_mixed():
    if 'audioFile' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    #clear folders
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'])

    if os.path.exists(app.config['NPY_FOLDER']):
        shutil.rmtree(app.config['NPY_FOLDER'])
    os.makedirs(app.config['NPY_FOLDER'])

    # Save the uploaded file
    file = request.files['audioFile']
     # Save the file to the server
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
           
    print(file_path)
    if (file_path.endswith('.mp3')):
        convert_mp3_to_wav(file_path, file_path[:-4] + ".wav")
        file_path = file_path[:-4] + ".wav"
    elif (not file_path.endswith(".wav")):
        return jsonify({"error": "Incompatible file type"}), 401

    #try:
    # Convert to spectrogram
    print("Going to change the audio to a spectogram")
    audio_to_spectrograms_mixed(file_path, app.config['NPY_FOLDER'])
    print("Back from creating the spectogram and going to perform prediction")

    #generate music features
    print("Creating Music Features")
    music_dataset = NumpyDataset(dir=app.config['NPY_FOLDER'], transform=transform)
    music_loader = DataLoader(dataset=music_dataset, batch_size=1, shuffle=False)
    music_features(MusicNet(), "MusicGenreClassifier2D.pth", music_loader, os.path.join(app.config['NPY_FOLDER'], 'music_features.npz'))

    #generate instrument features
    print("Creating Instrument Features")
    instrument_dataset = InstrumentDataset(dir=app.config['NPY_FOLDER'])
    instrument_loader = DataLoader(dataset=instrument_dataset, batch_size=1, shuffle=False)
    instrument_features(instrument_loader, os.path.join(app.config['NPY_FOLDER'], 'instrument_features.npz'))

    # Perform prediction on combined features
    print("Performing Classifications")
    features = concatenate_features(os.path.join(app.config['NPY_FOLDER'], 'music_features.npz'), os.path.join(app.config['NPY_FOLDER'], 'instrument_features.npz'))
    combined_dataset = SimpleDataset(features)
    combined_loader = DataLoader(dataset=combined_dataset, batch_size=1, shuffle=False)
    classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    predictions = music_classifier(FFNet(), "MixedMusicGenreClassifier.pth", combined_loader, classes)

    # Clean up uploaded file
    os.remove(file_path)

    # Return the genre as JSON
    return jsonify({"predictions":predictions})
    #except Exception as e:
    #    return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
