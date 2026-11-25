const video = document.getElementById('mirrorVideo');
const result = document.getElementById('result');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
let stream;

startBtn.addEventListener('click', startMirror);
stopBtn.addEventListener('click', stopMirror);

function startMirror() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(s => {
            stream = s;
            video.srcObject = stream;
            result.style.display = 'block';
            processFrames();
        })
        .catch(err => result.textContent = 'Kamera hiba: ' + err);
}

function stopMirror() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        result.style.display = 'none';
    }
}

function processFrames() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d'); 

    function sendFrame() {
        if (video.srcObject) {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('frame', blob, 'frame.jpg');
                fetch('https://api.jelsmart.hu/api/tukor', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => result.textContent = data.message || 'Nincs felismerés')
                .catch(err => result.textContent = 'API hiba: ' + err);
            }, 'image/jpeg');
            requestAnimationFrame(sendFrame);
        }
    }

    video.addEventListener('loadeddata', () => {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        sendFrame();
    });
}