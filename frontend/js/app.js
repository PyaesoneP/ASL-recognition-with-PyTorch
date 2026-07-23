/**
 * Main app: camera capture, MediaPipe hand detection, canvas ROI crop, API calls.
 */

const API_BASE = window.location.origin;

class ASLApp {
    constructor() {
        this.video = document.getElementById('video');
        this.overlay = document.getElementById('overlay');
        this.overlayCtx = this.overlay.getContext('2d');
        this.roiBox = document.getElementById('roiBox');
        this.predictionLabel = document.getElementById('predictionLabel');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusDot = this.statusIndicator.querySelector('.status-dot');
        this.statusText = this.statusIndicator.querySelector('.status-text');

        this.sentence = new SentenceManager();
        this.sentence.onSentenceUpdate = (action, prediction) => this.handleSentenceAction(action, prediction);

        this.mediaPipeHands = null;
        this.camera = null;
        this.ws = null;
        this.isRunning = false;
        this.frameCount = 0;
        this.predictInterval = 150;

        this.init();
    }

    async init() {
        try {
            await this.setupMediaPipe();
            await this.startCamera();
            this.connectWebSocket();
            this.setStatus('ready', 'Ready');
        } catch (err) {
            console.error('Init error:', err);
            this.setStatus('error', 'Error: ' + err.message);
        }
    }

    setupMediaPipe() {
        return new Promise((resolve, reject) => {
            this.mediaPipeHands = new Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            });

            this.mediaPipeHands.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.5
            });

            this.mediaPipeHands.onResults((results) => {
                this.onMediaPipeResults(results);
            });

            resolve();
        });
    }

    startCamera() {
        return new Promise((resolve, reject) => {
            this.camera = new Camera(this.video, {
                onFrame: async () => {
                    if (this.mediaPipeHands && this.isRunning) {
                        await this.mediaPipeHands.send({ image: this.video });
                    }
                },
                width: 640,
                height: 480
            });
            this.camera.start()
                .then(() => {
                    this.isRunning = true;
                    this.setStatus('ready', 'Camera active');
                    resolve();
                })
                .catch(reject);
        });
    }

    onMediaPipeResults(results) {
        this.overlay.width = this.video.videoWidth;
        this.overlay.height = this.video.videoHeight;

        this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height);

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];

            drawLandmarks(this.overlayCtx, landmarks);
            drawConnectors(this.overlayCtx, landmarks, HAND_CONNECTIONS, {
                color: '#00ff88',
                lineWidth: 2
            });

            this.roiBox.classList.add('active');

            const bbox = this.getBounds(landmarks);
            this.updateROIPosition(bbox);

            this.frameCount++;
            if (this.frameCount % Math.round(this.predictInterval / 16) === 0) {
                this.cropAndPredict(bbox);
            }
        } else {
            this.roiBox.classList.remove('active');
            this.predictionLabel.textContent = 'No gesture';
            this.confidenceFill.style.width = '0%';
            this.confidenceValue.textContent = '0%';
        }
    }

    getBounds(landmarks) {
        const xs = landmarks.map(l => l.x);
        const ys = landmarks.map(l => l.y);
        const min_x = Math.min(...xs);
        const max_x = Math.max(...xs);
        const min_y = Math.min(...ys);
        const max_y = Math.max(...ys);

        const w = max_x - min_x;
        const h = max_y - min_y;
        const padding = 0.25;

        const cx = (min_x + max_x) / 2;
        const cy = (min_y + max_y) / 2;
        const size = Math.max(w, h) * (1 + padding);

        return {
            x: cx - size / 2,
            y: cy - size / 2,
            w: size,
            h: size
        };
    }

    updateROIPosition(bbox) {
        const videoRect = this.video.getBoundingClientRect();
        const scaleX = videoRect.width / this.video.videoWidth;
        const scaleY = videoRect.height / this.video.videoHeight;

        this.roiBox.style.left = (bbox.x * scaleX * 100) / videoRect.width * 100 + '%';
        this.roiBox.style.top = (bbox.y * scaleY * 100) / videoRect.height * 100 + '%';
        this.roiBox.style.width = (bbox.w * scaleX * 100) / videoRect.width * 100 + '%';
        this.roiBox.style.height = (bbox.h * scaleY * 100) / videoRect.height * 100 + '%';
    }

    cropAndPredict(bbox) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;

        const x = Math.max(0, (bbox.x * vw) | 0);
        const y = Math.max(0, (bbox.y * vh) | 0);
        const w = Math.min(vw - x, (bbox.w * vw) | 0);
        const h = Math.min(vh - y, (bbox.h * vh) | 0);

        if (w <= 0 || h <= 0) return;

        canvas.width = 224;
        canvas.height = 224;

        ctx.drawImage(this.video, x, y, w, h, 0, 0, 224, 224);

        canvas.toBlob((blob) => {
            if (blob) {
                const reader = new FileReader();
                reader.onload = () => {
                    const base64 = reader.result;
                    this.sendPrediction(base64);
                };
                reader.readAsDataURL(blob);
            }
        }, 'image/jpeg', 0.8);
    }

    sendPrediction(imageBase64) {
        fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageBase64 })
        })
            .then(res => res.json())
            .then(data => {
                this.updatePrediction(data);
            })
            .catch(err => {
                console.error('Predict error:', err);
            });
    }

    updatePrediction(data) {
        if (data.prediction && data.prediction !== 'nothing' && data.prediction !== 'error') {
            this.predictionLabel.textContent = data.prediction;
            const pct = Math.round((data.confidence || 0) * 100);
            this.confidenceFill.style.width = pct + '%';
            this.confidenceValue.textContent = pct + '%';

            if (pct > 70) {
                this.confidenceFill.style.background = 'linear-gradient(90deg, #4caf50, #8bc34a)';
            } else if (pct > 40) {
                this.confidenceFill.style.background = 'linear-gradient(90deg, #ff9800, #ffc107)';
            } else {
                this.confidenceFill.style.background = 'linear-gradient(90deg, #f44336, #ff5722)';
            }
        } else {
            this.predictionLabel.textContent = 'No gesture';
            this.confidenceFill.style.width = '0%';
            this.confidenceValue.textContent = '0%';
        }
    }

    connectWebSocket() {
        const wsUrl = `${this.getWsUrl()}/api/stream`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'prediction') {
                if (data.prediction && data.prediction !== 'nothing') {
                    this.sentence.updateFromServer(data.sentence || '');
                }
            } else if (data.type === 'sentence') {
                this.sentence.updateFromServer(data.sentence || '');
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected, reconnecting in 3s...');
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (err) => {
            console.error('WebSocket error:', err);
        };
    }

    getWsUrl() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${window.location.host}`;
    }

    handleSentenceAction(action, prediction) {
        fetch(`${API_BASE}/api/sentence/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, prediction })
        })
            .then(res => res.json())
            .then(data => {
                this.sentence.updateFromServer(data.sentence);
            })
            .catch(err => console.error('Sentence update error:', err));
    }

    setStatus(state, text) {
        this.statusDot.className = 'status-dot ' + state;
        this.statusText.textContent = text;
    }
}

function drawLandmarks(ctx, landmarks) {
    for (let i = 0; i < landmarks.length; i++) {
        const x = landmarks[i].x * ctx.canvas.width;
        const y = landmarks[i].y * ctx.canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
    }
}

const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17]
];

document.addEventListener('DOMContentLoaded', () => {
    window.app = new ASLApp();
});
