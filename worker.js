import init, { initThreadPool, gif_to_avif } from "./pkg/giftoavif.js";

async function run() {
    await init();
    
    await initThreadPool(navigator.hardwareConcurrency);

    self.onmessage = async (event) => {
        const { fileData, fps, crf, encSpeed, playbackSpeed, interpolate } = event.data;
        
        try {
            const result = gif_to_avif(
                fileData, 
                fps, 
                crf, 
                encSpeed, 
                playbackSpeed, 
                interpolate
            );
            
            self.postMessage({ status: 'success', data: result }, [result.buffer]);
        } catch (e) {
            self.postMessage({ status: 'error', error: e.toString() });
        }
    };
    
    self.postMessage({ status: 'ready' });
}

run();
