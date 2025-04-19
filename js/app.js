document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('videoInput');
    const videoPreview = document.getElementById('videoPreview');
    const runButton = document.getElementById('runInference');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');

    // Initialize the WASM module
    initializeWasm().then(() => {
        console.log('WASM module initialized');
        runButton.disabled = false;
    }).catch(err => {
        console.error('Failed to initialize WASM:', err);
    });

    videoInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            videoPreview.src = url;
            videoPreview.style.display = 'block';
            runButton.disabled = false;
        }
    });

    runButton.addEventListener('click', async () => {
        if (!videoInput.files[0]) {
            alert('Please select a video first');
            return;
        }

        try {
            loadingDiv.style.display = 'block';
            runButton.disabled = true;
            resultsDiv.style.display = 'none';

            const results = await runVideoInference(videoInput.files[0]);
            
            // Display results
            resultsDiv.innerHTML = '<h3>Classification Results:</h3>' +
                results.map((result, i) => 
                    `<p>${i + 1}. ${result.class_name}: ${(result.probability * 100).toFixed(2)}%</p>`
                ).join('');
            resultsDiv.style.display = 'block';
        } catch (error) {
            console.error('Inference failed:', error);
            alert('Failed to process video: ' + error.message);
        } finally {
            loadingDiv.style.display = 'none';
            runButton.disabled = false;
        }
    });
});