// Direct browser port of hand_features.py. The feature schema and maths are unchanged.
import { CONFIG } from './sign_config.js';

export const FEATURE_SCHEMA = CONFIG.featureSchema;
export const FEATURE_COUNT = CONFIG.featureCount;

function handednessLabel(description) {
  // Python MediaPipe exposes description.classification[0].label.
  // Legacy MediaPipe JS usually exposes the equivalent label directly.
  if (description?.classification?.length) return description.classification[0]?.label ?? null;
  if (typeof description?.label === 'string') return description.label;
  if (Array.isArray(description) && description.length) return handednessLabel(description[0]);
  return null;
}

function validLandmarks(points) {
  if (!Array.isArray(points) || points.length !== 21) return false;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return false;
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  return Math.max(maxX - minX, maxY - minY) !== 0;
}

export function extractFeatures(results, inputIsMirrored = CONFIG.inputIsMirrored) {
  const landmarks = results?.multiHandLandmarks || [];
  const handedness = results?.multiHandedness || [];
  if (!landmarks.length || landmarks.length !== handedness.length) return null;

  const detected = new Map();
  const rejected = new Set();

  for (let i = 0; i < landmarks.length; i += 1) {
    let side = handednessLabel(handedness[i]);
    if (side !== 'Left' && side !== 'Right') return null;

    if (!inputIsMirrored) {
      side = side === 'Left' ? 'Right' : 'Left';
    }

    if (detected.has(side) || rejected.has(side)) {
      detected.delete(side);
      rejected.add(side);
      continue;
    }

    const points = landmarks[i];
    if (!validLandmarks(points)) {
      rejected.add(side);
      continue;
    }
    detected.set(side, points);
  }

  if (!detected.size) return null;

  const features = new Float32Array(FEATURE_COUNT);
  for (const [slot, side] of ['Left', 'Right'].entries()) {
    const points = detected.get(side);
    if (!points) continue;
    const start = slot * 42;
    for (let point = 0; point < 21; point += 1) {
      features[start + point * 2] = Math.fround(points[point].x);
      features[start + point * 2 + 1] = Math.fround(points[point].y);
    }
  }
  return features;
}

export function sampleMatrix(samples) {
  let rows;
  if (ArrayBuffer.isView(samples)) {
    rows = [samples];
  } else if (Array.isArray(samples) && (samples.length === 0 || typeof samples[0] === 'number')) {
    rows = [samples];
  } else if (Array.isArray(samples)) {
    rows = samples;
  } else {
    throw new Error('Expected samples containing 84 raw coordinate values.');
  }

  return rows.map(row => {
    if (!row || row.length !== FEATURE_COUNT) {
      throw new Error('Expected finite samples with 84 raw coordinate values.');
    }
    const output = Float32Array.from(row);
    for (const value of output) {
      if (!Number.isFinite(value)) {
        throw new Error('Expected finite samples with 84 raw coordinate values.');
      }
    }
    return output;
  });
}

export function handPresence(samples) {
  return sampleMatrix(samples).map(row => {
    let left = false;
    let right = false;
    for (let i = 0; i < 42; i += 1) left ||= row[i] !== 0;
    for (let i = 42; i < 84; i += 1) right ||= row[i] !== 0;
    return [left, right];
  });
}

export function eligibleSamples(samples, mode) {
  const present = handPresence(samples);
  if (mode === 'right') return present.map(value => value[1]);
  if (mode === 'both') return present.map(value => value[0] && value[1]);
  throw new Error(`Unknown hand mode: ${mode}`);
}

export function normaliseFeatures(samples, mode) {
  const rows = sampleMatrix(samples);

  if (mode === 'right') {
    return rows.map(row => {
      let minX = Infinity;
      let minY = Infinity;
      for (let i = 42; i < 84; i += 2) {
        minX = Math.min(minX, row[i]);
        minY = Math.min(minY, row[i + 1]);
      }
      const output = new Float32Array(42);
      for (let i = 0; i < 42; i += 2) {
        output[i] = Math.fround(row[42 + i] - minX);
        output[i + 1] = Math.fround(row[42 + i + 1] - minY);
      }
      return output;
    });
  }

  if (mode !== 'both') throw new Error(`Unknown hand mode: ${mode}`);

  const present = handPresence(rows);
  return rows.map((row, rowIndex) => {
    let minX = Infinity;
    let minY = Infinity;
    for (let hand = 0; hand < 2; hand += 1) {
      if (!present[rowIndex][hand]) continue;
      const start = hand * 42;
      for (let i = start; i < start + 42; i += 2) {
        minX = Math.min(minX, row[i]);
        minY = Math.min(minY, row[i + 1]);
      }
    }

    if (!Number.isFinite(minX)) {
      minX = 0;
      minY = 0;
    }

    const output = new Float32Array(FEATURE_COUNT);
    for (let hand = 0; hand < 2; hand += 1) {
      if (!present[rowIndex][hand]) continue;
      const start = hand * 42;
      for (let i = start; i < start + 42; i += 2) {
        output[i] = Math.fround(row[i] - minX);
        output[i + 1] = Math.fround(row[i + 1] - minY);
      }
    }
    return output;
  });
}
