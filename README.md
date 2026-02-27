# HandTrack-enhanced-
FINALLY,  FULL ASL RECOGNITION 
# PHASE 1 — ASL Data Collector
─────────────────────────────
Records your hand landmarks for each ASL letter and saves them to asl_dataset.csv.

HOW TO USE:
  1. Run this script:       python datacollect.py
  2. Press a letter key (A-Z) to select which letter you want to record.
  3. Hold your ASL gesture in front of the camera.
  4. Press SPACE to capture a sample (30-50 samples per letter).
  5. Repeat for all letters you want to recognize.
  6. Press Q to quit and save.


Requirements:
  pip install opencv-python mediapipe
# PHASE 2 — ASL Classifier Trainer (v2 — engineered features)
─────────────────────────────────────────────────────────────
Reads asl_dataset.csv, engineers rich hand features, trains a
GradientBoosting classifier, and saves asl_model.pkl + asl_labels.pkl.

Run after collecting data:
  python train.py

Requirements:
  pip install scikit-learn pandas numpy
