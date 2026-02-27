"""
features.py — ASL Hand Feature Engineering
────────────────────────────────────────────
Converts raw MediaPipe landmarks into a rich feature vector that makes
similar-looking letters (M/N, U/V, R/K, S/T etc.) much more distinguishable.

Features extracted (per hand):
  1. Normalized landmark positions  (63)  — wrist-relative, scale-invariant
  2. Fingertip distances to wrist   (5)   — how extended each finger is
  3. Fingertip distances to thumb   (4)   — spread/pinch detection
  4. Inter-fingertip distances      (10)  — finger groupings (M vs N vs S etc.)
  5. Joint bend angles              (15)  — knuckle curl per finger
  ─────────────────────────────────────
  Total: 97 features
"""

import numpy as np

# MediaPipe landmark indices
WRIST = 0
THUMB  = [1, 2, 3, 4]
INDEX  = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING   = [13, 14, 15, 16]
PINKY  = [17, 18, 19, 20]

FINGERTIPS   = [4, 8, 12, 16, 20]
FINGER_BASES = [1, 5, 9, 13, 17]   # MCP joints

# All finger joint triplets for angle calculation (base, mid, tip)
JOINT_TRIPLETS = [
    (1,  2,  3),   # thumb MCP-IP-tip
    (2,  3,  4),
    (5,  6,  7),   # index MCP-PIP-DIP
    (6,  7,  8),
    (9,  10, 11),  # middle
    (10, 11, 12),
    (13, 14, 15),  # ring
    (14, 15, 16),
    (17, 18, 19),  # pinky
    (18, 19, 20),
    # cross-finger angles that help distinguish M/N/S/T
    (5,  9,  13),
    (6,  10, 14),
    (8,  12, 16),
    (0,  5,  9),
    (0,  9,  13),
]

# Fingertip pairs for inter-tip distances (index through pinky)
TIP_PAIRS = [
    (8,  12),
    (8,  16),
    (8,  20),
    (12, 16),
    (12, 20),
    (16, 20),
    (4,  8),
    (4,  12),
    (4,  16),
    (4,  20),
]


def _angle(a, b, c):
    """Angle at point b formed by vectors b→a and b→c (radians)."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def extract(landmarks_array):
    """
    landmarks_array: numpy array of shape (21, 3) — raw x,y,z from MediaPipe.
    Returns: 1D numpy feature vector of length 97.
    """
    pts = landmarks_array.copy()

    # ── 1. Normalize: translate wrist to origin, scale by hand size ──────────
    wrist = pts[WRIST].copy()
    pts  -= wrist

    # Hand size = mean distance from wrist to each fingertip
    hand_size = np.mean([np.linalg.norm(pts[t]) for t in FINGERTIPS]) + 1e-6
    pts /= hand_size

    flat_coords = pts.flatten()  # 63 features

    # ── 2. Fingertip distances to wrist (normalized, so = relative extension) ─
    tip_wrist_dists = np.array([np.linalg.norm(pts[t]) for t in FINGERTIPS])  # 5

    # ── 3. Fingertip distances to thumb tip ───────────────────────────────────
    thumb_tip = pts[4]
    tip_thumb_dists = np.array([
        np.linalg.norm(pts[t] - thumb_tip) for t in FINGERTIPS[1:]  # index→pinky
    ])  # 4

    # ── 4. Inter-fingertip distances ──────────────────────────────────────────
    inter_tip_dists = np.array([
        np.linalg.norm(pts[a] - pts[b]) for a, b in TIP_PAIRS
    ])  # 10

    # ── 5. Joint bend angles ──────────────────────────────────────────────────
    angles = np.array([
        _angle(pts[a], pts[b], pts[c]) for a, b, c in JOINT_TRIPLETS
    ])  # 15

    return np.concatenate([
        flat_coords,        # 63
        tip_wrist_dists,    #  5
        tip_thumb_dists,    #  4
        inter_tip_dists,    # 10
        angles,             # 15
    ])  # = 97


def from_mediapipe(hand_landmarks):
    """Convenience: extract features directly from a MediaPipe hand_landmarks object."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    return extract(pts).reshape(1, -1)


def from_csv_row(row_values):
    """
    row_values: list/array of 63 floats (x0,y0,z0,...,x20,y20,z20)
    as stored in asl_dataset.csv (excluding the label column).
    """
    pts = np.array(row_values, dtype=float).reshape(21, 3)
    return extract(pts)
