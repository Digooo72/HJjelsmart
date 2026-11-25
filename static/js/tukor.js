const video = document.getElementById('mirrorVideo');
const result = document.getElementById('result');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

// Same-origin by default now that the API and frontend are one Flask app.
// Override by setting window.JELSMART_API_URL before this script loads.
const API_URL = window.JELSMART_API_URL || '/api/tukor';

// Throttle: only send one frame every FRAME_INTERVAL_MS instead of every
// requestAnimationFrame tick (which is 60/sec and needlessly hammers the
// model + network for a task that doesn't need that resolution in time).
const FRAME_INTERVAL_MS = 400;

let stream;
let running = false;
let requestInFlight = false;
let lastSendTime = 0;

startBtn.addEventListener('click', startMirror);
stopBtn.addEventListener('click', stopMirror);

function startMirror() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(s => {
            stream = s;
            video.srcObject = stream;
            running = true;
            result.style.display = 'block';
            result.textContent = 'Kamera indul...';
            processFrames();
        })
        .catch(err => {
            result.style.display = 'block';
            result.textContent = 'Kamera hiba: ' + err;
        });
}

function stopMirror() {
    running = false;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
        result.style.display = 'none';
    }
}

function processFrames() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    function tick(timestamp) {
        if (!running) return;

        if (!requestInFlight && timestamp - lastSendTime >= FRAME_INTERVAL_MS) {
            lastSendTime = timestamp;
            sendFrame(canvas, context);
        }

        requestAnimationFrame(tick);
    }

    video.addEventListener('loadeddata', () => {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        requestAnimationFrame(tick);
    });
}

function sendFrame(canvas, context) {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        if (!blob) return;

        requestInFlight = true;
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        fetch(API_URL, { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    result.textContent = data.error;
                    return;
                }
                const confidence = typeof data.confidence === 'number'
                    ? ` (${Math.round(data.confidence * 100)}%)`
                    : '';
                result.textContent = (data.message || 'Nincs felismeres') + confidence;
            })
            .catch(err => {
                result.textContent = 'API hiba: ' + err;
            })
            .finally(() => {
                requestInFlight = false;
            });
    }, 'image/jpeg', 0.85);
}
