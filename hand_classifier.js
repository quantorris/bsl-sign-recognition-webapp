// Browser inference port of hand_classifier.py.
// Training is not reimplemented: the website evaluates the already-trained model.p.
import { FEATURE_COUNT, eligibleSamples, normaliseFeatures, sampleMatrix } from './hand_features.js';

export const MODEL_SCHEMA = 'per_label_hand_selection_v1';
export const UNKNOWN_LABEL = 'unknown';

function treePositiveProbability(nodes, features) {
  let node = 0;
  while (true) {
    const current = nodes[node];
    if (current[0] < 0) return current[1];
    node = features[current[0]] <= current[1] ? current[2] : current[3];
  }
}

function calibratedPositiveProbability(savedModel, features) {
  let probability = 0;
  for (const tree of savedModel.trees) {
    probability += treePositiveProbability(tree, features);
  }
  probability /= savedModel.trees.length;

  // scikit-learn's fitted sigmoid calibration: expit(-(a * prediction + b)).
  const z = savedModel.calibrationA * probability + savedModel.calibrationB;
  if (z >= 0) {
    const e = Math.exp(-z);
    return e / (1 + e);
  }
  return 1 / (1 + Math.exp(z));
}

export class HandSelectiveClassifier {
  constructor(handModes, minScore = 0.55, minMargin = 0.05) {
    this.handModes = { ...handModes };
    this.minScore = minScore;
    this.minMargin = minMargin;
  }

  loadExportedModel(spec) {
    if (spec.featureCount !== FEATURE_COUNT) {
      throw new Error(`Expected model with ${FEATURE_COUNT} input values.`);
    }
    this.classes_ = [...spec.classes];
    this.handModes_ = Object.fromEntries(this.classes_.map(label => [label, this.handModes[label]]));
    this.models_ = spec.models;
    this.nFeaturesIn_ = FEATURE_COUNT;
    return this;
  }

  classScores(samples) {
    const data = sampleMatrix(samples);
    const modes = [...new Set(Object.values(this.handModes_))];
    const views = Object.fromEntries(modes.map(mode => [mode, normaliseFeatures(data, mode)]));
    const eligible = Object.fromEntries(modes.map(mode => [mode, eligibleSamples(data, mode)]));
    const scores = Array.from({ length: data.length }, () => new Array(this.classes_.length).fill(0));

    for (let column = 0; column < this.classes_.length; column += 1) {
      const label = this.classes_[column];
      const mode = this.handModes_[label];
      const savedModel = this.models_[label];
      for (let row = 0; row < data.length; row += 1) {
        if (!eligible[mode][row]) continue;
        scores[row][column] = calibratedPositiveProbability(savedModel, views[mode][row]);
      }
    }
    return scores;
  }

  predictDetails(samples) {
    const scores = this.classScores(samples);
    const predictions = [];
    const bestScores = [];

    for (const row of scores) {
      let winner = 0;
      for (let i = 1; i < row.length; i += 1) {
        if (row[i] > row[winner]) winner = i;
      }
      const best = row[winner];
      let runnerUp = -Infinity;
      for (let i = 0; i < row.length; i += 1) {
        if (i !== winner && row[i] > runnerUp) runnerUp = row[i];
      }

      const accepted = best > 0 && best >= this.minScore && (best - runnerUp) >= this.minMargin;
      predictions.push(accepted ? this.classes_[winner] : UNKNOWN_LABEL);
      bestScores.push(best);
    }
    return { predictions, scores: bestScores };
  }

  predict(samples) {
    return this.predictDetails(samples).predictions;
  }
}
