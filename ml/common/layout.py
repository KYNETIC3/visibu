"""Packed landmark frame layout. Must match src/tracking/frame.ts exactly.

One frame is 553 points x (x, y, z), in this order:
pose (33) | left hand (21) | right hand (21) | face (478).
A body part that was not detected is filled with NaN.
"""

POSE_POINTS = 33
HAND_POINTS = 21
FACE_POINTS = 478
DIMS = 3

POSE_START = 0
LEFT_START = POSE_POINTS
RIGHT_START = POSE_POINTS + HAND_POINTS
FACE_START = POSE_POINTS + 2 * HAND_POINTS
TOTAL_POINTS = POSE_POINTS + 2 * HAND_POINTS + FACE_POINTS  # 553
