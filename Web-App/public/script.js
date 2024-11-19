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
        audioPlayer.style.display = 'block'; // Show the audio player
    }
});


// Function to clear the existing chart and recreate the canvas
function clearChartContainer() {
    const chartContainer = document.getElementById('genreChart');
    chartContainer.innerHTML = ''; // Clear all child nodes
    const newCanvas = document.createElement('canvas'); // Create a new canvas
    newCanvas.id = 'genreChartCanvas'; // Give it an ID
    chartContainer.appendChild(newCanvas); // Append the new canvas to the container
}

// Function to count the frequency of genres
function countGenres(predictions) {
    const genreCounts = {};
    predictions.forEach(([genre]) => {
        genreCounts[genre] = (genreCounts[genre] || 0) + 1;
    });
    return genreCounts;
}

// Function to render the chart
function renderChart(genres, frequencies) {
    const ctx = document.getElementById('genreChartCanvas').getContext('2d');

    // Create a new chart with better configurations
    window.genreChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: genres,
            datasets: [{
                label: 'Frequency of Genres',
                data: frequencies,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false, // Allow resizing
            plugins: {
                legend: {
                    display: false // Hide the legend for simplicity
                }
            },
            scales: {
                x: {
                    labels: genres
                },
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Adjust canvas size dynamically
    document.getElementById('genreChartCanvas').style.height = '400px';
    document.getElementById('genreChartCanvas').style.width = '100%';
}


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


    console.log("I am going to backend.py to performe classification");

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to classify the audio.');
        }

        const data = await response.json();

        console.log("This was returned from the backend.py: " + data.predictions);

        // Format and display predictions with timestamps in a scrollable container
        if (data.predictions && Array.isArray(data.predictions)) {
            let currentTime = 0; // Start from 0:00
            const predictionsHtml = data.predictions
                .map(prediction => {
                    const timestamp = new Date(currentTime * 1000).toISOString().substr(14, 5); // Format as mm:ss
                    currentTime += 6; // Increment by 6 seconds
                    return `<li>${timestamp} - ${prediction[0]}: ${(prediction[1] * 100).toFixed(2)}%</li>`;
                })
                .join('');

            // Create a scrollable container
            resultDiv.innerHTML = `
        <div style="max-height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
            <ul>${predictionsHtml}</ul>
        </div>
    `;
        } else {
            resultDiv.innerHTML = "No predictions returned from the server.";
        }


        // Extract genres and confidences
        const genres = data.predictions.map(pred => pred[0]);
        const confidences = data.predictions.map(pred => pred[1]);
        const genreCounts = countGenres(genres); // Count genre frequencies
        const genreKeys = Object.keys(genreCounts); // Genre labels
        const frequencies = Object.values(genreCounts); // Frequency values

        // Render the chart
        clearChartContainer();
        renderChart(genreKeys, frequencies);

    } catch (error) {
        resultDiv.innerHTML = `Error: ${error.message}`;
    }
});
