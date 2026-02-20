/**
 * HT PRO - Frontend Controller
 * Robust Version for Production
 */

console.log("🚀 HT PRO: Initializing System...");

// State Management
let isProcessing = false;
let stream = null;
let progressInterval = null;
let countChart = null;
let actionChart = null;
let startTime = Date.now();
let activeSource = 'webcam';

// 1. Global Navigation Controller
window.switchSection = function (sectionId) {
    console.log("🚀 Switching to ->", sectionId);
    try {
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
            // Reset animations on children so they re-reveal
            section.querySelectorAll('.reveal').forEach(el => el.classList.remove('active'));
        });

        const activeSection = document.getElementById(sectionId + '-section');
        if (activeSection) {
            activeSection.classList.add('active');
            // Small delay to ensure browser paints first
            setTimeout(() => {
                activeSection.querySelectorAll('.reveal').forEach(el => {
                    // Trigger reveal if it's already in viewport or just by being the active section
                    el.classList.add('active');
                });
            }, 100);
        }

        // Update nav ui
        document.querySelectorAll('.sidebar-nav li').forEach(li => {
            li.classList.toggle('active', li.dataset.section === sectionId);
        });

        if (sectionId === 'analytics' || sectionId === 'history') {
            initCharts();
            updateAnalytics();
        }
    } catch (err) {
        console.error("❌ Navigation error:", err);
    }
};

// Mobile Sidebar Controller
window.toggleSidebar = function () {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');

    if (sidebar) sidebar.classList.toggle('active');
    if (overlay) overlay.classList.toggle('active');
};

// 2. Initialization on Load
document.addEventListener('DOMContentLoaded', () => {
    console.log("✅ DOM fully loaded");

    // Create Sidebar Overlay if missing
    if (!document.querySelector('.sidebar-overlay')) {
        const overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        overlay.onclick = toggleSidebar;
        document.body.appendChild(overlay);
    }

    // Attach Toggle Listener
    const toggleBtn = document.getElementById('mobile-sidebar-toggle');
    if (toggleBtn) toggleBtn.onclick = toggleSidebar;

    document.querySelectorAll('.sidebar-nav li').forEach(item => {
        item.addEventListener('click', () => {
            window.switchSection(item.dataset.section);
            // Auto close sidebar on mobile after click
            if (window.innerWidth <= 1024) toggleSidebar();
        });
    });

    initROI();
    initScrollReveal();
    checkFirstVisit();

    const heroBtn = document.querySelector('.hero-card .btn-primary');
    if (heroBtn) heroBtn.addEventListener('click', () => window.switchSection('tracking'));

    console.log("✨ HT PRO: Ready");
});

// Animation Controller
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.content-section, .viewport-card, .controls-card, .chart-card, .summary-box, .history-card').forEach(el => {
        el.classList.add('reveal');
        observer.observe(el);
    });
}

// 3. Camera & AI Logic
window.setSource = function (source) {
    activeSource = source;
    const buttons = document.querySelectorAll('#source-toggle button');
    buttons.forEach(btn => btn.classList.remove('active'));
    if (source === 'webcam') buttons[0].classList.add('active');
    else buttons[1].classList.add('active');

    if (isProcessing) {
        stopCamera();
        setTimeout(startCamera, 500);
    }
};

window.startCamera = async function () {
    const videoFeed = document.getElementById('video_feed');
    const clientVideo = document.getElementById('client_video');
    const canvas = document.getElementById('capture_canvas');
    const context = canvas.getContext('2d');
    const loading = document.getElementById('loading-overlay');

    if (activeSource === 'server') {
        videoFeed.src = "/video_feed";
        isProcessing = true;
        return;
    }

    try {
        if (loading) loading.classList.remove('hidden');
        if (stream) stream.getTracks().forEach(t => t.stop());

        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720 }
        });

        clientVideo.srcObject = stream;
        clientVideo.onloadedmetadata = () => {
            if (loading) loading.classList.add('hidden');
            isProcessing = true;
            processLoop();
        };
    } catch (err) {
        console.error("❌ Camera failed:", err);
        if (loading) loading.classList.add('hidden');
    }

    async function processLoop() {
        if (!isProcessing || activeSource !== 'webcam') return;

        if (canvas.width !== clientVideo.videoWidth) {
            canvas.width = clientVideo.videoWidth || 640;
            canvas.height = clientVideo.videoHeight || 480;
        }

        if (clientVideo.readyState === clientVideo.HAVE_ENOUGH_DATA) {
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
            } catch (err) { }
        }
        if (isProcessing && activeSource === 'webcam') requestAnimationFrame(processLoop);
    }
};

window.stopCamera = function () {
    isProcessing = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    const vid = document.getElementById('client_video');
    if (vid) vid.srcObject = null;
    const feed = document.getElementById('video_feed');
    if (feed) feed.src = "";
};

// 4. Analytics & Charting
function initCharts() {
    if (countChart) return;

    const countEl = document.getElementById('countChart');
    if (countEl) {
        countChart = new Chart(countEl.getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Person Count', data: [], borderColor: '#0ea5e9', tension: 0.4 }] },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }

    const actionEl = document.getElementById('actionChart');
    if (actionEl) {
        actionChart = new Chart(actionEl.getContext('2d'), {
            type: 'doughnut',
            data: { labels: [], datasets: [{ data: [], backgroundColor: ['#0ea5e9', '#6366f1', '#10b981', '#f59e0b'] }] },
            options: { responsive: true, maintainAspectRatio: false, cutout: '70%' }
        });
    }
}

async function updateAnalytics() {
    try {
        // Live Occupancy
        const res = await fetch('/stats/realtime');
        if (res.ok) {
            const stats = await res.json();
            if (stats.length > 0 && countChart) {
                countChart.data.labels = stats.map(s => s.timestamp.split(' ')[1]).reverse();
                countChart.data.datasets[0].data = stats.map(s => s.count).reverse();
                countChart.update('none');

                const peakEl = document.getElementById('peak-count-val');
                if (peakEl) peakEl.innerText = Math.max(...stats.map(s => s.count));
            }
        }

        // Behavior
        const behaviorRes = await fetch('/stats/behavior');
        if (behaviorRes.ok && actionChart) {
            const bData = await behaviorRes.json();
            actionChart.data.labels = Object.keys(bData);
            actionChart.data.datasets[0].data = Object.values(bData);
            actionChart.update('none');
        }

        // History
        const historyRes = await fetch('/stats/history');
        if (historyRes.ok) {
            const logs = await historyRes.json();
            const tbody = document.querySelector('#history-table tbody');
            if (tbody) {
                tbody.innerHTML = logs.map(log => `
                    <tr>
                        <td>${log.timestamp.split(' ')[1]}</td>
                        <td><span class="id-pill">ID-${log.track_id}</span></td>
                        <td><span class="status-pill ${log.action.toLowerCase().includes('fall') ? 'warning' : 'success'}">${log.action}</span></td>
                        <td>${(log.confidence * 100).toFixed(0)}%</td>
                    </tr>
                `).join('');
            }
        }
    } catch (e) { console.error("⚠️ Analytics failed:", e); }

    // Session Timer
    const timeEl = document.getElementById('session-time-val');
    if (timeEl) {
        const diff = Math.floor((Date.now() - startTime) / 1000);
        const m = Math.floor(diff / 60).toString().padStart(2, '0');
        const s = (diff % 60).toString().padStart(2, '0');
        timeEl.innerText = `${m}:${s}`;
    }

    const analyticsSec = document.getElementById('analytics-section');
    if (analyticsSec && analyticsSec.classList.contains('active')) setTimeout(updateAnalytics, 3000);
}

// 5. Utilities
function initROI() {
    let isDrawing = false;
    let startX, startY;
    const roiBtn = document.getElementById('draw-roi-btn');
    const canvas = document.getElementById('capture_canvas');

    if (roiBtn && canvas) {
        roiBtn.addEventListener('click', () => {
            isDrawing = !isDrawing;
            roiBtn.classList.toggle('active');
            canvas.style.display = isDrawing ? 'block' : 'none';
        });

        canvas.addEventListener('mousedown', (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
        });

        canvas.addEventListener('mouseup', async (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const roi = [Math.min(startX, e.clientX - rect.left), Math.min(startY, e.clientY - rect.top), Math.max(startX, e.clientX - rect.left), Math.max(startY, e.clientY - rect.top)].map(v => Math.floor(v));

            await fetch('/set_roi', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(roi) });
            isDrawing = false;
            roiBtn.classList.remove('active');
            canvas.style.display = 'none';
        });
    }
}

window.uploadVideo = async function () {
    const fileInput = document.getElementById('file-upload');
    if (!fileInput || !fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    document.getElementById('processing-modal').classList.remove('hidden');

    try {
        const response = await fetch('/upload_video', { method: 'POST', body: formData });
        if (response.ok) {
            const { task_id } = await response.json();
            document.getElementById('processing_stream').src = `/stream_processing/${task_id}`;
            pollProgress(task_id);
        }
    } catch (err) { }
};

function pollProgress(taskId) {
    progressInterval = setInterval(async () => {
        try {
            const res = await fetch(`/get_progress/${taskId}`);
            if (res.ok) {
                const data = await res.json();
                document.getElementById('progress-bar-fill').style.width = `${data.progress}%`;
                document.getElementById('progress-text').innerText = `${data.progress}% Completed`;

                if (data.complete) {
                    clearInterval(progressInterval);
                    document.getElementById('modal-status').innerText = "Complete!";
                    document.getElementById('modal-actions').classList.remove('hidden');
                    document.getElementById('download-btn').onclick = () => window.location.href = `/download_video/${taskId}`;
                }
            }
        } catch (e) { }
    }, 1000);
}

window.closeModal = () => {
    document.getElementById('processing-modal').classList.add('hidden');
    clearInterval(progressInterval);
};

window.resetROI = async () => {
    await fetch('/reset_roi', { method: 'POST' });
};

window.setAnalysisMode = async function (mode) {
    console.log("🧠 Analysis Mode ->", mode);

    // Update UI
    const buttons = document.querySelectorAll('#analysis-toggle button');
    buttons.forEach(btn => btn.classList.remove('active'));
    if (mode === 'human') document.getElementById('mode-human').classList.add('active');
    else document.getElementById('mode-behavior').classList.add('active');

    // Notify Backend
    try {
        await fetch('/set_analysis_mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode })
        });
    } catch (err) {
        console.error("❌ Mode switch failed");
    }
};

// Guided Tour Controller
window.startTour = function () {
    const driverObj = window.driver.js.driver({
        showProgress: true,
        animate: true,
        padding: 10,
        popoverClass: 'glass-popover',
        progressText: 'Step {{current}} of {{total}}',
        nextBtnText: 'Next',
        prevBtnText: 'Previous',
        doneBtnText: 'Finish',
        steps: [
            {
                element: '.sidebar',
                popover: {
                    title: 'Navigation Sidebar',
                    description: '<div class="tour-character"><i class="fas fa-robot"></i></div>Use this menu to switch between Tracking, Analytics, and Activity History.',
                    side: "right",
                    align: 'start'
                }
            },
            {
                element: '.viewport-card',
                popover: {
                    title: 'Live Viewport',
                    description: '<div class="tour-character"><i class="fas fa-robot"></i></div>Here you can see the real-time video processing and human tracking with accurate counts.',
                    side: "bottom",
                    align: 'center'
                },
                onHighlightStarted: () => window.switchSection('tracking')
            },
            {
                element: '.controls-card',
                popover: {
                    title: 'Control Panel',
                    description: '<div class="tour-character"><i class="fas fa-robot"></i></div>Choose your camera source, toggle "Behavior Mode", or upload a video for intelligent processing.',
                    side: "left",
                    align: 'start'
                },
                onHighlightStarted: () => window.switchSection('tracking')
            },
            {
                element: '.roi-controls',
                popover: {
                    title: 'Region of Interest (ROI)',
                    description: '<div class="tour-character"><i class="fas fa-robot"></i></div>Use these tools to define specific areas for the system to focus on, ignoring the rest.',
                    side: "top",
                    align: 'center'
                },
                onHighlightStarted: () => window.switchSection('tracking')
            },
            {
                element: '.analytics-grid',
                popover: {
                    title: 'Live Analytics',
                    description: '<div class="tour-character"><i class="fas fa-robot"></i></div>Interactive charts showing peak occupancy and behavioral distributions (Standing, Walking, etc.).',
                    side: "top",
                    align: 'center'
                },
                onHighlightStarted: () => window.switchSection('analytics')
            },
            {
                element: '.history-card',
                popover: {
                    title: 'Engagement History',
                    description: '<div class="tour-character"><i class="fas fa-robot"></i></div>A detailed log of all detected activities with the ability to export as a CSV file.',
                    side: "top",
                    align: 'center'
                },
                onHighlightStarted: () => window.switchSection('history')
            },
            {
                element: '#chatbot-toggle-btn',
                popover: {
                    title: 'AI Assistant',
                    description: '<div class="tour-character"><i class="fas fa-robot"></i></div>If you need any help, I am always here to answer your questions and guide you.',
                    side: "left",
                    align: 'center'
                }
            },
        ]
    });
    driverObj.drive();
};

// Help Bot Controller
window.toggleChatbot = function () {
    const window = document.getElementById('chatbot-window');
    window.classList.toggle('hidden');
};

window.botResponse = function (action) {
    const chatContent = document.getElementById('chat-content');
    let response = "";
    let targetSection = "";

    switch (action) {
        case 'start':
            response = "لبدء التتبع، اختر مصدر الفيديو من لوحة التحكم (Control Panel) ثم اضغط على زر البث. يمكنك أيضاً رفع ملف فيديو!";
            targetSection = "tracking";
            break;
        case 'analytics':
            response = "قسم الإحصائيات يعرض لك تحليلات حية عن أعداد الأشخاص وسلوكياتهم المكتشفة بمرور الوقت.";
            targetSection = "analytics";
            break;
        case 'roi':
            response = "يمكنك رسم منطقة محددة (ROI) لتركيز التتبع عليها فقط وتجاهل باقي المشهد لزيادة الدقة.";
            targetSection = "tracking";
            break;
        case 'export':
            response = "يمكنك تصدير كافة البيانات المسجلة بصيغة CSV من خلال زر التجارة الموجود في صفحة السجل (History).";
            targetSection = "history";
            break;
    }

    if (response) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'bot-msg';
        msgDiv.innerHTML = `<p>${response}</p>`;
        chatContent.appendChild(msgDiv);
        chatContent.scrollTop = chatContent.scrollHeight;

        if (targetSection) {
            setTimeout(() => window.switchSection(targetSection), 1500);
        }
    }
};

// Check for first-time visit
function checkFirstVisit() {
    if (!localStorage.getItem('ht_pro_tour_seen')) {
        setTimeout(() => {
            window.startTour();
            localStorage.setItem('ht_pro_tour_seen', 'true');
        }, 2000);
    }
}
