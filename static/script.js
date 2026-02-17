let isProcessing = false;
let stream = null;
let progressInterval = null;

async function startCamera() {
    const videoContainer = document.getElementById('live-feed-container');
    const videoFeed = document.getElementById('video_feed');
    const clientVideo = document.getElementById('client_video');
    const canvas = document.getElementById('capture_canvas');
    const context = canvas.getContext('2d');

    try {
        videoContainer.classList.remove('hidden');
        videoContainer.scrollIntoView({ behavior: 'smooth' });

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        clientVideo.srcObject = stream;
        isProcessing = true;
        processFrame();
    } catch (err) {
        console.error("Error accessing camera:", err);
        alert("Could not access camera. Please ensure you are using HTTPS or localhost and have given permission.");
    }

    async function processFrame() {
        if (!isProcessing) return;
        if (canvas.width !== clientVideo.videoWidth) {
            canvas.width = clientVideo.videoWidth;
            canvas.height = clientVideo.videoHeight;
        }
        context.drawImage(clientVideo, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg', 0.5);

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
    document.getElementById('live-feed-container').classList.add('hidden');
    document.getElementById('video_feed').src = "";
}

async function uploadVideo() {
    console.log("uploadVideo triggered");
    const fileInput = document.getElementById('file-upload');
    if (!fileInput.files.length) {
        console.log("No file selected");
        return;
    }
    console.log("File selected:", fileInput.files[0].name);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Show processing modal immediately
    const modal = document.getElementById('processing-modal');
    const streamImg = document.getElementById('processing_stream');
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    const status = document.getElementById('modal-status');
    const actions = document.getElementById('modal-actions');
    const downloadBtn = document.getElementById('download-btn');

    modal.classList.remove('hidden');
    fill.style.width = '0%';
    text.innerText = '0%';
    status.innerText = 'جاري رفع الملف...';
    actions.classList.add('hidden');
    streamImg.src = "";

    try {
        const response = await fetch('/upload_video', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            const taskId = data.task_id;

            status.innerText = 'جاري المعالجة...';
            // Start the processing stream
            streamImg.src = `/stream_processing/${taskId}`;

            // Poll for progress
            progressInterval = setInterval(async () => {
                const progRes = await fetch(`/get_progress/${taskId}`);
                if (progRes.ok) {
                    const progData = await progRes.json();
                    fill.style.width = `${progData.progress}%`;
                    text.innerText = `${progData.progress}%`;

                    if (progData.complete) {
                        clearInterval(progressInterval);
                        status.innerText = 'اكتملت المعالجة!';
                        actions.classList.remove('hidden');

                        downloadBtn.onclick = () => {
                            window.location.href = `/download_video/${taskId}`;
                        };
                    }
                }
            }, 1000);
        } else {
            alert("حدث خطأ أثناء الرفع.");
            closeModal();
        }
    } catch (err) {
        console.error("Upload error:", err);
        alert("حدث خطأ في الاتصال بالسيرفر.");
        closeModal();
    }
}

function closeModal() {
    document.getElementById('processing-modal').classList.add('hidden');
    document.getElementById('processing_stream').src = "";
    if (progressInterval) clearInterval(progressInterval);
}

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});
