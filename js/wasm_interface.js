let videoMAEModule = null;
let inferenceInstance = null;

async function initializeWasm() {
    try {
        // Load and instantiate the WebAssembly module
        videoMAEModule = await Module();
        inferenceInstance = videoMAEModule._create_inference();
        console.log('VideoMAE WebAssembly module initialized');
        return true;
    } catch (error) {
        console.error('Failed to initialize WASM module:', error);
        throw error;
    }
}

async function runVideoInference(videoFile) {
    if (!videoMAEModule || !inferenceInstance) {
        throw new Error('WASM module not initialized');
    }

    // Extract frames from video
    const frames = await extractVideoFrames(videoFile);
    
    // Convert frames to flat array and copy to WASM memory
    const frameData = new Float32Array(frames.length * 224 * 224 * 3);
    let offset = 0;
    for (const frame of frames) {
        frameData.set(new Float32Array(frame.data.buffer), offset);
        offset += frame.data.length;
    }
    
    // Allocate memory in WASM and copy frame data
    const frameDataPtr = videoMAEModule._malloc(frameData.byteLength);
    new Float32Array(videoMAEModule.HEAPF32.buffer, frameDataPtr, frameData.length)
        .set(frameData);

    try {
        // Run inference
        const resultsPtr = videoMAEModule._run_inference(
            inferenceInstance,
            frameDataPtr,
            frames.length
        );

        if (!resultsPtr) {
            throw new Error('Inference failed');
        }

        // Parse results
        const results = [];
        let currentPtr = resultsPtr;
        for (let i = 0; i < 5; i++) {
            const classNamePtr = videoMAEModule.getValue(currentPtr, 'i32');
            const probability = videoMAEModule.getValue(currentPtr + 4, 'float');
            
            results.push({
                class_name: videoMAEModule.UTF8ToString(classNamePtr),
                probability: probability
            });
            
            // Free the class name string
            videoMAEModule._free(classNamePtr);
            currentPtr += 8;
        }

        // Free allocated memory
        videoMAEModule._free(frameDataPtr);
        videoMAEModule._free(resultsPtr);

        return results;
    } catch (error) {
        // Clean up on error
        videoMAEModule._free(frameDataPtr);
        throw error;
    }
}

// Helper function to extract frames from video
async function extractVideoFrames(videoFile) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        const frames = [];
        
        video.onloadedmetadata = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 224;
            canvas.height = 224;
            
            const duration = video.duration;
            const framePoints = Array.from({length: 16}, (_, i) => 
                (duration * i) / 16
            );
            
            let framesCaptured = 0;
            
            video.onseeked = () => {
                ctx.drawImage(video, 0, 0, 224, 224);
                frames.push(ctx.getImageData(0, 0, 224, 224));
                
                framesCaptured++;
                if (framesCaptured < framePoints.length) {
                    video.currentTime = framePoints[framesCaptured];
                } else {
                    resolve(frames);
                }
            };
            
            video.currentTime = framePoints[0];
        };
        
        video.onerror = reject;
        video.src = URL.createObjectURL(videoFile);
    });
}

window.initializeWasm = initializeWasm;
window.runVideoInference = runVideoInference;