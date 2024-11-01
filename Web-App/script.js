document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const resultDiv = document.getElementById('result');
    const audioFile = document.getElementById('audioFile').files[0];

    if (!audioFile) {
        resultDiv.innerHTML = 'Please upload a valid audio file.';
        return;
    }

    const formData = new FormData();
    formData.append('audioFile', audioFile);

    try {
        const response = await fetch('/classify', {
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
