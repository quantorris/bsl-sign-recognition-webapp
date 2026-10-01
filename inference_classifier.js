// Browser port of inference_classifier.py. Recognition behaviour is kept the same;
// OpenCV VideoCapture/imshow is replaced by browser camera + canvas APIs.
import { CONFIG } from './sign_config.js';
import { HandSelectiveClassifier, MODEL_SCHEMA, UNKNOWN_LABEL } from './hand_classifier.js';
import { FEATURE_COUNT, FEATURE_SCHEMA, extractFeatures } from './hand_features.js';

const MEDIAPIPE_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/';
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17],
];

const video = document.querySelector('#camera');
const output = document.querySelector('#output');
const processing = document.querySelector('#processing');
const startButton = document.querySelector('#start');
const stopButton = document.querySelector('#stop');
const switchCameraButton = document.querySelector('#switch-camera');
const message = document.querySelector('#message');
const outputContext = output.getContext('2d');
const processingContext = processing.getContext('2d');

let model;
let hands;
let stream;
let running = false;
let processingFrame = false;
let animationFrame = 0;
let facingMode = 'user';

function setMessage(text) {
  message.textContent = text;
}

async function loadModel() {
  const response = await fetch('./model.json', { cache: 'no-cache' });
  if (!response.ok) throw new Error(`Could not load model.json (${response.status}).`);
  const saved = await response.json();

  // Same compatibility checks as load_model() in inference_classifier.py.
  if (saved.modelSchema !== MODEL_SCHEMA ||
      saved.featureSchema !== FEATURE_SCHEMA ||
      saved.featureCount !== FEATURE_COUNT) {
    throw new Error('Model/settings changed. Re-export the supplied model.p.');
  }
  for (const label of saved.classes) {
    if (saved.models[label]?.mode !== CONFIG.handModes[label]) {
      throw new Error('Model/settings changed. Re-export the supplied model.p.');
    }
  }

  const loaded = new HandSelectiveClassifier(CONFIG.handModes, CONFIG.minScore, CONFIG.minMargin)
    .loadExportedModel(saved);
  loaded.minScore = CONFIG.minScore;
  loaded.minMargin = CONFIG.minMargin;
  return loaded;
}

function drawLandmarks(hand) {
  outputContext.lineCap = 'round';
  outputContext.strokeStyle = '#00ff00';
  outputContext.fillStyle = '#ff0000';
  outputContext.lineWidth = 2;

  for (const [a, b] of HAND_CONNECTIONS) {
    outputContext.beginPath();
    outputContext.moveTo(hand[a].x * output.width, hand[a].y * output.height);
    outputContext.lineTo(hand[b].x * output.width, hand[b].y * output.height);
    outputContext.stroke();
  }
  for (const point of hand) {
    outputContext.beginPath();
    outputContext.arc(point.x * output.width, point.y * output.height, 3, 0, Math.PI * 2);
    outputContext.fill();
  }
}

function drawPredictionBox(features, mode) {
  let start = 0;
  let end = FEATURE_COUNT;
  if (mode === 'right') start = 42;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let index = start; index < end; index += 2) {
    const x = features[index] * output.width;
    const y = features[index + 1] * output.height;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  const left = Math.max(0, Math.min(output.width - 1, minX - 10));
  const top = Math.max(0, Math.min(output.height - 1, minY - 10));
  const right = Math.max(0, Math.min(output.width - 1, maxX + 10));
  const bottom = Math.max(0, Math.min(output.height - 1, maxY + 10));

  outputContext.strokeStyle = 'rgb(0,255,0)';
  outputContext.lineWidth = 2;
  outputContext.strokeRect(left, top, right - left, bottom - top);
}

function drawStatus(status) {
  outputContext.font = '20px sans-serif';
  outputContext.textBaseline = 'alphabetic';
  outputContext.lineJoin = 'round';
  outputContext.lineWidth = 5;
  outputContext.strokeStyle = 'black';
  outputContext.strokeText(status, 10, 30);
  outputContext.lineWidth = 2;
  outputContext.strokeStyle = 'white';
  outputContext.strokeText(status, 10, 30);
}

function drawRecognisedStatus(label, mode, score) {
  // Keep the recognised sign large on screen even when a high-resolution
  // camera frame is being scaled down to phone width. This is display-only.
  const displayedWidth = output.getBoundingClientRect().width || output.clientWidth || output.width;
  const scale = displayedWidth > 0 ? output.width / displayedWidth : 1;
  const x = 12 * scale;
  const headlineY = 44 * scale;
  const detailsY = 70 * scale;
  const availableWidth = output.width - (24 * scale);
  const headline = `Sign: ${label}`;

  let headlineSize = 34 * scale;
  outputContext.font = `700 ${headlineSize}px sans-serif`;
  const measured = outputContext.measureText(headline).width;
  if (measured > availableWidth) {
    headlineSize *= availableWidth / measured;
    outputContext.font = `700 ${headlineSize}px sans-serif`;
  }

  outputContext.textBaseline = 'alphabetic';
  outputContext.lineJoin = 'round';
  outputContext.lineWidth = 5 * scale;
  outputContext.strokeStyle = 'black';
  outputContext.strokeText(headline, x, headlineY);
  outputContext.fillStyle = 'white';
  outputContext.fillText(headline, x, headlineY);

  outputContext.font = `600 ${16 * scale}px sans-serif`;
  const details = `(${mode}; score ${score.toFixed(2)})`;
  outputContext.lineWidth = 4 * scale;
  outputContext.strokeStyle = 'black';
  outputContext.strokeText(details, x, detailsY);
  outputContext.fillStyle = 'white';
  outputContext.fillText(details, x, detailsY);
}

function onResults(results) {
  if (!running) return;
  const width = results.image?.videoWidth || results.image?.width || video.videoWidth || 640;
  const height = results.image?.videoHeight || results.image?.height || video.videoHeight || 480;
  if (output.width !== width || output.height !== height) {
    output.width = width;
    output.height = height;
  }

  outputContext.clearRect(0, 0, width, height);
  outputContext.drawImage(results.image, 0, 0, width, height);

  let status = 'No hands detected';
  let recognised = null;
  if (results.multiHandLandmarks?.length) {
    status = 'Hands unclear: adjust their position';
    for (const hand of results.multiHandLandmarks) drawLandmarks(hand);
  }

  const features = extractFeatures(results, CONFIG.inputIsMirrored);
  if (features !== null) {
    const { predictions, scores } = model.predictDetails(features);
    const label = predictions[0];
    status = 'Sign unclear or required hand missing';
    if (label !== UNKNOWN_LABEL) {
      const mode = model.handModes_[label];
      const displayLabel = CONFIG.labels[label] ?? label;
      status = `Sign: ${displayLabel} (${mode}; score ${scores[0].toFixed(2)})`;
      recognised = { label: displayLabel, mode, score: scores[0] };
      drawPredictionBox(features, mode);
    }
  }

  if (recognised) {
    drawRecognisedStatus(recognised.label, recognised.mode, recognised.score);
  } else {
    drawStatus(status);
  }
}

async function initialiseHands() {
  if (hands) return;
  if (typeof window.Hands !== 'function') {
    throw new Error('MediaPipe Hands did not load. Check the internet connection and reload.');
  }
  hands = new window.Hands({ locateFile: file => `${MEDIAPIPE_BASE}${file}` });
  hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.3,
    minTrackingConfidence: 0.5,
  });
  hands.onResults(onResults);
}

function frameForInference() {
  // Mirrors the cv2.flip() condition in inference_classifier.py.
  if (CONFIG.cameraIsMirrored === CONFIG.inputIsMirrored) return video;

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (processing.width !== width || processing.height !== height) {
    processing.width = width;
    processing.height = height;
  }
  processingContext.save();
  processingContext.clearRect(0, 0, width, height);
  processingContext.translate(width, 0);
  processingContext.scale(-1, 1);
  processingContext.drawImage(video, 0, 0, width, height);
  processingContext.restore();
  return processing;
}

async function loop() {
  if (!running) return;
  if (!processingFrame && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    processingFrame = true;
    try {
      await hands.send({ image: frameForInference() });
    } catch (error) {
      console.error(error);
      setMessage('Could not process the camera frame.');
    } finally {
      processingFrame = false;
    }
  }
  animationFrame = requestAnimationFrame(loop);
}

async function openCamera(requestedFacingMode) {
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: requestedFacingMode } },
  });
}

function stopCurrentStream() {
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
}

function updateSwitchCameraButton() {
  const target = facingMode === 'user' ? 'back' : 'front';
  switchCameraButton.setAttribute('aria-label', `Switch to ${target} camera`);
  switchCameraButton.title = `Switch to ${target} camera`;
}

async function attachCamera(requestedFacingMode) {
  stream = await openCamera(requestedFacingMode);
  facingMode = requestedFacingMode;
  video.srcObject = stream;
  await video.play();
  updateSwitchCameraButton();
}

async function startCamera() {
  startButton.disabled = true;
  setMessage('Starting camera…');
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser does not provide camera access.');
    }
    await initialiseHands();
    await attachCamera(facingMode);
    running = true;
    startButton.hidden = true;
    switchCameraButton.hidden = false;
    stopButton.hidden = false;
    setMessage('');
    animationFrame = requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    stopCurrentStream();
    startButton.disabled = false;
    setMessage(error.message || 'Could not open the camera.');
  }
}

async function switchCamera() {
  if (!running || !stream) return;

  const previousFacingMode = facingMode;
  const requestedFacingMode = facingMode === 'user' ? 'environment' : 'user';
  const requestedName = requestedFacingMode === 'environment' ? 'back' : 'front';

  switchCameraButton.disabled = true;
  stopButton.disabled = true;
  setMessage(`Switching to ${requestedName} camera…`);
  running = false;
  cancelAnimationFrame(animationFrame);
  stopCurrentStream();

  try {
    await attachCamera(requestedFacingMode);
    running = true;
    setMessage('');
    animationFrame = requestAnimationFrame(loop);
  } catch (error) {
    console.error(error);
    // If the requested camera is unavailable, reopen the camera that was working.
    try {
      await attachCamera(previousFacingMode);
      running = true;
      setMessage(`Could not switch to the ${requestedName} camera.`);
      animationFrame = requestAnimationFrame(loop);
    } catch (restoreError) {
      console.error(restoreError);
      stopCurrentStream();
      startButton.hidden = false;
      startButton.disabled = false;
      switchCameraButton.hidden = true;
      stopButton.hidden = true;
      setMessage('Could not reopen the camera.');
    }
  } finally {
    switchCameraButton.disabled = false;
    stopButton.disabled = false;
  }
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(animationFrame);
  stopCurrentStream();
  outputContext.clearRect(0, 0, output.width, output.height);
  startButton.hidden = false;
  startButton.disabled = false;
  switchCameraButton.hidden = true;
  stopButton.hidden = true;
  setMessage('Camera stopped.');
}

async function main() {
  try {
    model = await loadModel();
    startButton.disabled = false;
    setMessage('Press Start camera.');
  } catch (error) {
    console.error(error);
    setMessage(error.message || 'Could not load the trained model.');
  }
}

startButton.addEventListener('click', startCamera);
switchCameraButton.addEventListener('click', switchCamera);
stopButton.addEventListener('click', stopCamera);
window.addEventListener('pagehide', stopCurrentStream);

main();
