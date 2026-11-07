import { describe, expect, it } from 'vitest';
import { FRAME_LENGTH, HAND_POINTS, hasPart, packFrame, readPart, type Point } from './frame';

const pts = (n: number, seed: number): Point[] =>
  Array.from({ length: n }, (_, i) => ({ x: seed + i, y: seed + i + 0.5, z: -i }));

describe('packFrame', () => {
  it('packs every part and reads it back', () => {
    const left = pts(HAND_POINTS, 1);
    const frame = packFrame({
      poseLandmarks: [pts(33, 100)],
      leftHandLandmarks: [left],
      rightHandLandmarks: [pts(21, 200)],
      faceLandmarks: [pts(478, 300)],
    });
    expect(frame.length).toBe(FRAME_LENGTH);
    expect(readPart(frame, 'leftHand')).toEqual(left);
    expect(readPart(frame, 'face')![477]).toEqual({ x: 777, y: 777.5, z: -477 });
  });

  it('marks missing parts as absent', () => {
    const frame = packFrame({
      poseLandmarks: [pts(33, 0)],
      leftHandLandmarks: [],
      rightHandLandmarks: [pts(21, 0)],
      faceLandmarks: [],
    });
    expect(hasPart(frame, 'leftHand')).toBe(false);
    expect(hasPart(frame, 'rightHand')).toBe(true);
    expect(readPart(frame, 'face')).toBeNull();
  });
});
