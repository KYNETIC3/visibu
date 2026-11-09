import { useEffect, useRef, useState, useCallback } from 'react';
import { Tracker } from '../tracking/tracker';
import { frameBus } from '../lib/signRecognizer';
import type { Delegate } from '../tracking/messages';

interface UseTrackerOptions {
  /** Video element to track. */
  video: HTMLVideoElement | null;
  /** Whether tracking should be active. */
  active: boolean;
  /** Called when tracker is ready. */
  onReady?: (delegate: Delegate, loadMs: number) => void;
  /** Called on fatal tracking error. */
  onError?: (message: string) => void;
}

/**
 * Hook that manages a MediaPipe Holistic tracker for the given video element.
 * Publishes frames to the global frameBus for sign recognition.
 * Returns tracker status and stats.
 */
export function useTracker({
  video,
  active,
  onReady,
  onError,
}: UseTrackerOptions) {
  const trackerRef = useRef<Tracker | null>(null);
  const [status, setStatus] = useState<'idle' | 'starting' | 'running' | 'error'>('idle');
  const [delegate, setDelegate] = useState<Delegate | null>(null);
  const [loadMs, setLoadMs] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Keep callbacks in refs so new inline functions from the caller don't
  // recreate (and reload) the tracker on every render.
  const onReadyRef = useRef(onReady);
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onReadyRef.current = onReady;
    onErrorRef.current = onError;
  }, [onReady, onError]);

  const startTracker = useCallback(() => {
    if (!video || trackerRef.current) return;

    setStatus('starting');
    setError(null);

    const tracker = new Tracker({
      onReady: (d, ms) => {
        setDelegate(d);
        setLoadMs(ms);
        setStatus('running');
        tracker.start(video);
        onReadyRef.current?.(d, ms);
      },
      onFrame: (frame, timestamp, _inferenceMs) => {
        // Publish to frame bus for sign recognition
        frameBus.publish(frame, timestamp);
      },
      onError: (message) => {
        setStatus('error');
        setError(message);
        onErrorRef.current?.(message);
      },
    });

    trackerRef.current = tracker;
  }, [video]);

  const stopTracker = useCallback(() => {
    trackerRef.current?.dispose();
    trackerRef.current = null;
    setStatus('idle');
    setDelegate(null);
    setLoadMs(null);
  }, []);

  useEffect(() => {
    if (active && video) {
      startTracker();
    } else {
      stopTracker();
    }
    return stopTracker;
  }, [active, video, startTracker, stopTracker]);

  return { status, delegate, loadMs, error };
}