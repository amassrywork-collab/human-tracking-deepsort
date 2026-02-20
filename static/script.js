// HT PRO - Frontend Controller
let isProcessing = false;
let stream = null;
let progressInterval = null;
let countChart = null;
let actionChart = null;
let startTime = Date.now();

// --- Navigation ---
document.querySelectorAll('.sidebar-nav li').forEach(item => {
    item.addEventListener('click', function () {
        const sectionId = this.dataset.section;
        switchSection(sectionId);

        // Update UI
        document.querySelectorAll('.sidebar-nav li').forEach(li => li.classList.remove('active'));
        this.classList.add('active');
    });
});

function switchSection(sectionId) {
    document.querySelectorAll('.content-section').forEach(sec => sec.classList.remove('active'));
    document.getElementById(`${sectionId}-section`).classList.add('active');

    if (sectionId === 'analytics') {
        initCharts();
        updateAnalytics();
    }
    // Check if analytics section is active and update after a delay
    if (document.getElementById('analytics-section').classList.contains('active')) {
        setTimeout(updateAnalytics, 2000);
    }
}

// --- Live Tracking ---
async function startCamera() {
    const videoFeed = document.getElementById('video_feed');
    const clientVideo = document.getElementById('client_video');
    const canvas = document.getElementById('capture_canvas');
    const context = canvas.getContext('2d');
    const loading = document.getElementById('loading-overlay');

    try {
        loading.classList.remove('hidden');
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        clientVideo.srcObject = stream;
        loading.classList.add('hidden');

        isProcessing = true;
        processLoop();
    } catch (err) {
        console.error("Camera error:", err);
        alert("Camera access denied or unavailable.");
    }

    async function processLoop() {
        if (!isProcessing) return;

        if (canvas.width !== clientVideo.videoWidth) {
            canvas.width = clientVideo.videoWidth;
            canvas.height = clientVideo.videoHeight;
        }

        context.drawImage(clientVideo, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/jpeg', 0.6);

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
            console.error("Frame processing failed");
        }

        if (isProcessing) requestAnimationFrame(processLoop);
    }
}

function stopCamera() {
    isProcessing = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    document.getElementById('video_feed').src = "";
}

// --- Video Upload & Processing ---
async function uploadVideo() {
    const fileInput = document.getElementById('file-upload');
    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const modal = document.getElementById('processing-modal');
    modal.classList.remove('hidden');

    try {
        const response = await fetch('/upload_video', { method: 'POST', body: formData });
        if (response.ok) {
            const { task_id } = await response.json();
            document.getElementById('processing_stream').src = `/stream_processing/${task_id}`;

            pollProgress(task_id);
        }
    } catch (err) {
        alert("Upload failed.");
    }
}

function pollProgress(taskId) {
    const progressText = document.getElementById('progress-text');
    const actions = document.getElementById('modal-actions');
    const downloadBtn = document.getElementById('download-btn');

    progressInterval = setInterval(async () => {
        const res = await fetch(`/get_progress/${taskId}`);
        if (res.ok) {
            const data = await res.json();
            progressText.innerText = `${data.progress}%`;

            if (data.complete) {
                clearInterval(progressInterval);
                actions.classList.remove('hidden');
                downloadBtn.onclick = () => window.location.href = `/download_video/${taskId}`;
            }
        }
    }, 1000);
}

function closeModal() {
    document.getElementById('processing-modal').classList.add('hidden');
    if (progressInterval) clearInterval(progressInterval);
}

// --- Analytics & Charts ---
function initCharts() {
    if (countChart) return; // Prevent double init

    const ctxCount = document.getElementById('countChart').getContext('2d');
    countChart = new Chart(ctxCount, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'People Count',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });

    const ctxAction = document.getElementById('actionChart').getContext('2d');
    actionChart = new Chart(ctxAction, {
        type: 'doughnut',
        data: {
            labels: ['Standing', 'Walking', 'Falling'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#10b981', '#3b82f6', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } }
        }
    });
}

async function updateAnalytics() {
    try {
        const res = await fetch('/stats/realtime');
        if (res.ok) {
            const stats = await res.json(); // Array of {timestamp, count}

            // Update Line Chart
            countChart.data.labels = stats.map(s => s.timestamp.split(' ')[1]).reverse();
            countChart.data.datasets[0].data = stats.map(s => s.count).reverse();
            countChart.update();

            // Update Peak
            const counts = stats.map(s => s.count);
            document.getElementById('peak-count-val').innerText = Math.max(...counts, 0);

            // Update History Table (Last 5)
            const tbody = document.querySelector('#history-table tbody');
            tbody.innerHTML = stats.slice(0, 10).map(s => `
                <tr>
                    <td>${s.timestamp}</td>
                    <td>PID-${Math.floor(Math.random() * 1000)}</td>
                    <td>Detected</td>
                    <td><span class="status-pill active">Active</span></td>
                </tr>
            `).join('');
        }
    } catch (e) {
        console.error("Stats update failed");
    }

    // Timer
    const diff = Math.floor((Date.now() - startTime) / 1000);
    const m = Math.floor(diff / 60).toString().padStart(2, '0');
    const s = (diff % 60).toString().padStart(2, '0');
    document.getElementById('session-time-val').innerText = `${m}:${s}`;

    // --- ROI Drawing Tool ---
    let isDrawing = false;
    let startX, startY;
    const roiBtn = document.getElementById('draw-roi-btn');

    if (roiBtn) {
        roiBtn.addEventListener('click', () => {
            isDrawing = !isDrawing;
            roiBtn.classList.toggle('active');
            const canvas = document.getElementById('capture_canvas');
            canvas.style.display = isDrawing ? 'block' : 'none';
            canvas.style.cursor = 'crosshair';

            if (isDrawing) {
                alert("Click and drag on the video to select the detection zone.");
            }
        });

        const canvas = document.getElementById('capture_canvas');
        canvas.addEventListener('mousedown', (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
        });

        canvas.addEventListener('mouseup', async (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;

            const roi = [
                Math.min(startX, endX), Math.min(startY, endY),
                Math.max(startX, endX), Math.max(startY, endY)
            ].map(v => Math.floor(v));

            // Scale ROI to video coordinates if necessary (assuming canvas same as video)
            try {
                await fetch('/set_roi', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(roi)
                });
                alert("Detection zone updated!");
            } catch (err) {
                console.error("ROI update failed");
            }

            isDrawing = false;
            roiBtn.classList.remove('active');
            canvas.style.display = 'none';
        });
    }
