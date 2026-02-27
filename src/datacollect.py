"""
PHASE 1 — ASL Data Collector
─────────────────────────────
Records your hand landmarks for each ASL letter and saves them to asl_dataset.csv.

HOW TO USE:
  1. Run this script:       python 1_collect_data.py
  2. Press a letter key (A-Z) to select which letter you want to record.
  3. Hold your ASL gesture in front of the camera.
  4. Press SPACE to capture a sample (aim for 30-50 samples per letter).
  5. Repeat for all letters you want to recognize.
  6. Press Q to quit and save.

TIPS:
  - Vary your hand position slightly between captures (slight rotation, distance).
  - Good lighting helps MediaPipe track landmarks accurately.
  - You can re-run this script to ADD more samples — it appends to the CSV.

Requirements:
  pip install opencv-python mediapipe
"""

import cv2
import mediapipe as mp
import csv
import os
import time
from collections import defaultdict

OUTPUT_FILE = "asl_dataset.csv"

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def landmarks_to_row(hand_landmarks):
    """Flatten 21 landmarks (x, y, z) into a list of 63 floats."""
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)])
    return row


def load_existing_counts():
    counts = defaultdict(int)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    counts[row[0]] += 1
    return counts


def main():
    cap = cv2.VideoCapture(1)
    current_letter = None
    sample_counts  = load_existing_counts()
    flash_msg      = ""
    flash_until    = 0

    # Open CSV (append mode)
    file_exists = os.path.exists(OUTPUT_FILE)
    csv_file    = open(OUTPUT_FILE, "a", newline="")
    writer      = csv.writer(csv_file)
    if not file_exists:
        # Write header: label + 63 landmark values (x0,y0,z0 ... x20,y20,z20)
        header = ["label"] + [f"{c}{i}" for i in range(21) for c in ("x", "y", "z")]
        writer.writerow(header)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            h, w, _ = frame.shape

            # Draw landmarks
            hand_detected = False
            hand_landmarks_data = None
            if result.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks_data = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks_data, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

            # ── UI ──────────────────────────────────────────────────────────
            # Top bar
            cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
            cv2.putText(frame, "ASL Data Collector  |  Press letter key to select  |  SPACE = capture  |  Q = quit",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Current letter + count
            if current_letter:
                cnt = sample_counts[current_letter]
                bar_color = (0, 200, 100) if cnt >= 30 else (0, 180, 255)
                cv2.rectangle(frame, (0, 45), (w, 95), (30, 30, 30), -1)
                cv2.putText(frame, f"Recording: {current_letter}   Samples: {cnt}  {'✓ GOOD' if cnt >= 30 else '(aim for 30+)'}",
                            (10, 78), cv2.FONT_HERSHEY_DUPLEX, 0.85, bar_color, 2)
            else:
                cv2.rectangle(frame, (0, 45), (w, 95), (30, 30, 30), -1)
                cv2.putText(frame, "Press a letter key (A-Z) to start recording",
                            (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 2)

            # Hand status dot
            dot_color = (0, 255, 80) if hand_detected else (0, 0, 200)
            cv2.circle(frame, (w - 20, 110), 9, dot_color, -1)
            cv2.putText(frame, "hand" if hand_detected else "no hand",
                        (w - 75, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1)

            # Flash message (after capture)
            if time.time() < flash_until:
                cv2.rectangle(frame, (0, h - 55), (w, h), (0, 120, 0), -1)
                cv2.putText(frame, flash_msg,
                            (10, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

            # Sample count summary (bottom right)
            total = sum(sample_counts.values())
            cv2.putText(frame, f"Total samples: {total}",
                        (w - 220, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

            # Letter grid showing progress
            grid_y = 110
            col    = 0
            for letter in LETTERS:
                cnt = sample_counts.get(letter, 0)
                color = (0, 220, 80) if cnt >= 30 else (0, 160, 255) if cnt > 0 else (80, 80, 80)
                x = 10 + col * 38
                if letter == current_letter:
                    cv2.rectangle(frame, (x - 3, grid_y - 20), (x + 32, grid_y + 8), (255, 255, 255), 1)
                cv2.putText(frame, f"{letter}", (x, grid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, str(cnt), (x, grid_y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                col += 1
                if col >= 13:
                    col     = 0
                    grid_y += 45

            cv2.imshow("ASL Data Collector", frame)

            # ── Key handling ────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord(']'):
                break
            elif key == ord(' '):
                # Capture sample
                if current_letter and hand_landmarks_data:
                    row = [current_letter] + landmarks_to_row(hand_landmarks_data)
                    writer.writerow(row)
                    csv_file.flush()
                    sample_counts[current_letter] += 1
                    flash_msg   = f" [+] Captured sample {sample_counts[current_letter]} for '{current_letter}'"
                    flash_until = time.time() + 0.6
                elif not current_letter:
                    flash_msg   = "<< Select a letter first (press A-Z)"
                    flash_until = time.time() + 1.0
                elif not hand_landmarks_data:
                    flash_msg   = "<< No hand detected!"
                    flash_until = time.time() + 1.0
            elif 97 <= key <= 122:  # a-z
                current_letter = chr(key).upper()
            elif 65 <= key <= 90:   # A-Z
                current_letter = chr(key)

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDataset saved to {OUTPUT_FILE}")
    print("Sample counts:")
    for letter in LETTERS:
        cnt = sample_counts.get(letter, 0)
        print(f"  {letter}: {cnt} {'✓' if cnt >= 30 else '(needs more)'}")


if __name__ == "__main__":
    main()
