let isProcessing = false;
let stream = null;

async function startCamera() {
    const videoContainer = document.getElementById('live-feed-container');
    const videoFeed = document.getElementById('video_feed');
    const clientVideo = document.getElementById('client_video');
    const canvas = document.getElementById('capture_canvas');
    const context = canvas.getContext('2d');

    try {
        // Show container
        videoContainer.classList.remove('hidden');
        videoContainer.scrollIntoView({ behavior: 'smooth' });

        // Get user camera
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user', // or 'environment' for back camera
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        clientVideo.srcObject = stream;
        isProcessing = true;

        // Start processing loop
        processFrame();
    } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Could not access camera. Please ensure you are using HTTPS or localhost and have given permission.");
    }

    async function processFrame() {
        if (!isProcessing) return;

        // Set canvas size to match video
        if (canvas.width !== clientVideo.videoWidth) {
            canvas.width = clientVideo.videoWidth;
            canvas.height = clientVideo.videoHeight;
        }

        // Draw video to canvas
        context.drawImage(clientVideo, 0, 0, canvas.width, canvas.height);

        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg', 0.5); // 0.5 quality for speed

        try {
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            if (response.ok) {
                const data = await response.json();
                videoFeed.src = data.image;
            }
        } catch (err) {
            console.error("Processing error:", err);
        }

        // Next frame
        if (isProcessing) {
            requestAnimationFrame(processFrame);
        }
    }
}

function stopCamera() {
    isProcessing = false;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    const videoContainer = document.getElementById('live-feed-container');
    videoContainer.classList.add('hidden');

    // Stop feed
    document.getElementById('video_feed').src = "";
}

function uploadVideo() {
    document.getElementById("uploadForm").submit();
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});
