// One tracked frame, packed into a flat Float32Array so it can be transferred
// from the worker without copying. Missing body parts are filled with NaN.

export const POSE_POINTS = 33;
export const HAND_POINTS = 21;
export const FACE_POINTS = 478;
const DIMS = 3; // x, y, z

export const LAYOUT = {
  pose: { offset: 0, points: POSE_POINTS },
  leftHand: { offset: POSE_POINTS * DIMS, points: HAND_POINTS },
  rightHand: { offset: (POSE_POINTS + HAND_POINTS) * DIMS, points: HAND_POINTS },
  face: { offset: (POSE_POINTS + 2 * HAND_POINTS) * DIMS, points: FACE_POINTS },
} as const;

export type Part = keyof typeof LAYOUT;
export const FRAME_LENGTH = (POSE_POINTS + 2 * HAND_POINTS + FACE_POINTS) * DIMS;

export interface Point {
  x: number;
  y: number;
  z: number;
}

/** The subset of MediaPipe's HolisticLandmarkerResult that we keep. */
export interface HolisticLike {
  poseLandmarks: Point[][];
  leftHandLandmarks: Point[][];
  rightHandLandmarks: Point[][];
  faceLandmarks: Point[][];
}

export function packFrame(result: HolisticLike): Float32Array {
  const out = new Float32Array(FRAME_LENGTH).fill(NaN);
  write(out, 'pose', result.poseLandmarks[0]);
  write(out, 'leftHand', result.leftHandLandmarks[0]);
  write(out, 'rightHand', result.rightHandLandmarks[0]);
  write(out, 'face', result.faceLandmarks[0]);
  return out;
}

function write(out: Float32Array, part: Part, points: Point[] | undefined) {
  if (!points) return;
  const { offset, points: count } = LAYOUT[part];
  for (let i = 0; i < Math.min(count, points.length); i++) {
    out[offset + i * DIMS] = points[i].x;
    out[offset + i * DIMS + 1] = points[i].y;
    out[offset + i * DIMS + 2] = points[i].z;
  }
}

export function hasPart(frame: Float32Array, part: Part): boolean {
  return !Number.isNaN(frame[LAYOUT[part].offset]);
}

export function readPart(frame: Float32Array, part: Part): Point[] | null {
  if (!hasPart(frame, part)) return null;
  const { offset, points: count } = LAYOUT[part];
  const points: Point[] = [];
  for (let i = 0; i < count; i++) {
    const at = offset + i * DIMS;
    points.push({ x: frame[at], y: frame[at + 1], z: frame[at + 2] });
  }
  return points;
}
