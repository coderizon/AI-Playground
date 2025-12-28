import { useCallback, useEffect, useRef, useState } from 'react';

import { FilesetResolver, ObjectDetector } from '@mediapipe/tasks-vision';

const OBJECT_DETECTOR_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite';
const OBJECT_DETECTOR_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

let detectorPromise = null;

async function loadObjectDetectorOnce() {
  if (detectorPromise) return detectorPromise;

  detectorPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(OBJECT_DETECTOR_WASM_URL);

    try {
      return await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: OBJECT_DETECTOR_MODEL_URL,
          delegate: 'GPU',
        },
        scoreThreshold: 0.5,
        runningMode: 'VIDEO',
      });
    } catch (error) {
      console.warn('[ObjectDetector] GPU delegate failed, falling back to CPU.', error);
      return ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: OBJECT_DETECTOR_MODEL_URL,
        },
        scoreThreshold: 0.5,
        runningMode: 'VIDEO',
      });
    }
  })();

  return detectorPromise;
}

export function useObjectDetector({ enabled = true } = {}) {
  const [status, setStatus] = useState(enabled ? 'loading' : 'idle');
  const [error, setError] = useState(null);
  const detectorRef = useRef(null);

  useEffect(() => {
    if (!enabled) {
      setStatus('idle');
      return undefined;
    }

    let cancelled = false;
    setStatus('loading');

    loadObjectDetectorOnce()
      .then((detector) => {
        if (cancelled) return;
        detectorRef.current = detector;
        setStatus('ready');
      })
      .catch((loadError) => {
        if (cancelled) return;
        console.error(loadError);
        setError(loadError);
        setStatus('error');
      });

    return () => {
      cancelled = true;
    };
  }, [enabled]);

  const detect = useCallback((videoElement, timestampMs) => {
    const detector = detectorRef.current;
    if (!detector || !videoElement) return [];

    const timestamp =
      typeof timestampMs === 'number'
        ? timestampMs
        : typeof performance !== 'undefined' && typeof performance.now === 'function'
          ? performance.now()
          : Date.now();

    const result = detector.detectForVideo(videoElement, timestamp);
    return result?.detections ?? [];
  }, []);

  return {
    status,
    error,
    detector: detectorRef.current,
    detect,
  };
}
