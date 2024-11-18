const audioFileInput = document.getElementById('audioFile');
const audioPlayer = document.getElementById('audioPlayer');
const modelSelector = document.getElementById('modelSelector');

// Show and load the audio player when a file is selected
audioFileInput.addEventListener('change', () => {
    const audioFile = audioFileInput.files[0];

    if (audioFile) {
        // Create a URL for the selected audio file and set it as the source for the audio player
        const fileURL = URL.createObjectURL(audioFile);
        audioPlayer.src = fileURL;
        audioPlayer.style.display = 'block';  // Show the audio player
    }
});

// Handle the form submission to upload and classify the file
document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent the default form submission

    const resultDiv = document.getElementById('result');
    const audioFile = audioFileInput.files[0];
    const selectedModel = modelSelector.value; // Get the selected model from the dropdown

    if (!audioFile) {
        resultDiv.innerHTML = "Please upload a valid audio file.";
        return;
    }

    // Prepare the form data for upload
    const formData = new FormData();
    formData.append('audioFile', audioFile);

    // Determine the API endpoint based on the selected model
    const endpoint = selectedModel === '1D-cnn' 
        ? 'http://127.0.0.1:5000/classify_1D' 
        : 'http://127.0.0.1:5000/classify_2D';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to classify the audio.');
        }

        const data = await response.json();
        resultDiv.innerHTML = `Predicted Genre: ${data.genre}`;
    } catch (error) {
        resultDiv.innerHTML = `Error: ${error.message}`;
    }
});