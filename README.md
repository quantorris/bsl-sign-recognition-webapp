# Phone web port of the supplied sign-language project

This folder is the GitHub Pages version of the supplied project. It is an inference port, not a new recogniser.

## What was kept unchanged

The browser code follows the supplied Python inference path:

1. MediaPipe Hands finds up to two hands with the same detection and tracking confidence values.
2. `hand_features.js` uses the same 84-value `left_right_xy_raw_v2` layout, handedness correction, missing-hand zero slots, eligibility rules and right/both normalisation as `hand_features.py`.
3. `hand_classifier.js` uses the same per-label classifier selection, calibrated class score, highest-score winner, `MIN_SCORE` and `MIN_MARGIN` rejection logic as `hand_classifier.py`.
4. `model.json` is a serialization of the already-trained `model(1).p`. It was not retrained.
5. `sign_config.js` was generated from the supplied `sign_config.py`.
6. `inference_classifier.js` retains the original camera-frame status states and bounding-box behaviour. Browser camera/canvas APIs replace OpenCV `VideoCapture`, `imshow`, `waitKey` and window shutdown.

## Phone display and camera controls

The recognition logic and trained model are unchanged. Two interface-only improvements are included:

- Accepted sign labels are drawn much larger so they remain readable when a high-resolution camera frame is scaled down to a phone screen. The hand mode and score remain visible underneath.
- The camera starts with the front-facing camera. While it is running, **Switch camera** (↻) alternates between the browser's `user` and `environment` cameras, so supported phones can use either the front or back camera.

The saved model was checked against the browser classifier on all 2,095 samples in the supplied `data(1).pickle`: 0 prediction mismatches, with a maximum class-score difference of about `2.22e-16` in the verification environment.

## Files to publish on GitHub Pages

Publish this folder exactly as it is. `index.html` must be at the repository root alongside:

- `inference_classifier.js`
- `hand_classifier.js`
- `hand_features.js`
- `sign_config.js`
- `model.json`
- `styles.css`

The page loads the legacy `@mediapipe/hands` JavaScript package from jsDelivr because the supplied Python project uses the legacy `mp.solutions.hands` pipeline.

## Camera access

Phone browsers require a secure context for camera access. GitHub Pages is served over HTTPS, so after deployment open the Pages URL, press **Start camera**, and grant camera permission.

The browser does not expose OpenCV camera indices such as `CAMERA_INDEX = 0`. The web port therefore starts with the browser's front-facing (`user`) camera and can request the back-facing (`environment`) camera with the switch button. This is a platform substitution only; it does not change feature extraction or classification.
