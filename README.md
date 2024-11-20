
# Music Genre Classification Web Application

This repository contains a web-based music genre classification application that leverages machine learning models to analyze audio files and predict their genres. We are unable to host this site non-locally because of the large storage requirements needed because of our models, which are challenging to manage on Google Cloud. The application supports both 1D-CNN and 2D-based classification methods. 

---

## **Table of Contents**

1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Setup and Installation](#setup-and-installation)  
4. [How to Run](#how-to-run)  
5. [Troubleshooting](#troubleshooting)
  

---

## **Overview**

This project implements a Flask backend that interfaces with machine learning models for music genre classification. It uses:

- **1D-CNN**: For genre classification based on audio waveform features.  
- **2D-CNN**: Utilizing spectrograms for classification.  

The web frontend allows users to upload audio files in `.wav` or `.mp3` format and view predictions with genre and confidence levels.

---

## **Prerequisites**

Before running this application, ensure you have the following installed:

- Python 3.8 or later
- Pip (Python's package installer)
- ffmpeg (for audio processing, required by `pydub`)

You will also need to download:
1. Pre-trained models from the provided Google Drive link.
2. Necessary files for `panns_data` as outlined below.

---

## **Setup and Installation**

1. Clone this repository:
   ```bash
   git clone (https://github.com/rhit-schrocjq/CSSE416-Music_Genre_Classification.git)
   
2. Install the necessary python dependencies:
    ```bash
    pip install flask torchaudio torch numpy scipy pydub torchvision librosa flask-cors
3. Download the pre-trained models from the provided Google Drive link and place them in the appropriate folder:
    - https://drive.google.com/file/d/17_4z8vzs6h2xd-52qq37zoqpv5MFH23J/view?usp=sharing
    - place those downloaded files in Web-App - flask-backend
4. Download the files for panns_data:
   - https://drive.google.com/file/d/17_4z8vzs6h2xd-52qq37zoqpv5MFH23J/view?usp=sharing
   - create a directorty called C:users/<you>/panns_data where you will place these downloaded files
5. Ensure ffmpeg is installed and accessible on your system. For example:
   - On Ubuntu/Debian:
   ```bash
   sudo apt-get install ffmpeg
 - On macOS:
    ```bash
    brew install ffmpeg
    
  On Windows: Download and add ffmpeg to your system path from FFmpeg.org.

## **How To Run**
1. Download a song as a wav with a sample rate of 22050 or as a mp3
2. Navigate to the flask-backend directory
3. Start the backend server
   ```bash
   python backend.py
4. Start the Frontend
   - Open the index.html file in your browser to launch the web interface
5. Upload your audio file
6. Select a model: 1D CNN, 2D CNN, or 2D Mixed CNN
7. Click on the "Classify" button
8. View the predicted genre results in the browser


## **Troubleshooting** ##
- Ensure that the backend is running before interacting with the frontend.
- If you encounter CORS issues, ensure that the flask-cors package is installed and properly configured in the backend.py.
- Ensure that you are uploading a mp3 or a wav file with a sample rate of 22050






