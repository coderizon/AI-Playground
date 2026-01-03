import { useCallback, useEffect, useRef, useState } from 'react';

import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';

const DEPTH_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';
const VISION_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

let segmenterPromise = null;

async function loadDepthEstimatorOnce() {
  if (segmenterPromise) return segmenterPromise;

  segmenterPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);

    try {
      return await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: DEPTH_MODEL_URL,
          delegate: 'GPU',
        },
        outputCategoryMask: true,
        outputConfidenceMasks: false,
        runningMode: 'VIDEO',
      });
    } catch (error) {
      console.warn('[DepthEstimation] GPU delegate failed, falling back to CPU.', error);
      return ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: DEPTH_MODEL_URL,
        },
        outputCategoryMask: true,
        outputConfidenceMasks: false,
        runningMode: 'VIDEO',
      });
    }
  })();

  return segmenterPromise;
}

export function useDepthEstimation({ enabled = true } = {}) {
  const [status, setStatus] = useState(enabled ? 'loading' : 'idle');
  const [error, setError] = useState(null);
  const segmenterRef = useRef(null);

  useEffect(() => {
    if (!enabled) {
      setStatus('idle');
      return undefined;
    }

    let cancelled = false;
    setStatus('loading');

    loadDepthEstimatorOnce()
      .then((segmenter) => {
        if (cancelled) return;
        segmenterRef.current = segmenter;
        setStatus('ready');
      })
      .catch((loadError) => {
        if (cancelled) return;
        console.error('[DepthEstimation] Fehler beim Laden des Modells:', loadError);
        setError(loadError);
        setStatus('error');
      });

    return () => {
      cancelled = true;
    };
  }, [enabled]);

  const predict = useCallback((videoElement, timestampMs) => {
    const segmenter = segmenterRef.current;
    if (!segmenter || !videoElement) return null;

    const timestamp =
      typeof timestampMs === 'number'
        ? timestampMs
        : typeof performance !== 'undefined' && typeof performance.now === 'function'
          ? performance.now()
          : Date.now();

    try {
      const result = segmenter.segmentForVideo(videoElement, timestamp);
      return result?.categoryMask ?? null;
    } catch (predictError) {
      console.error('[DepthEstimation] Fehler bei der Vorhersage:', predictError);
      return null;
    }
  }, []);

  return {
    status,
    error,
    model: segmenterRef.current,
    predict,
  };
}
