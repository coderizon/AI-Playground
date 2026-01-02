import { useCallback, useEffect, useRef, useState } from 'react';

import * as poseDetection from '@tensorflow-models/pose-detection';
import { initTensorFlowBackend } from '../utils/tensorflow-init.js';

const MOVENET_MODEL = poseDetection.SupportedModels.MoveNet;
const MOVENET_CONFIG = {
  modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
};
const MOVENET_OUTPUT_DIM = 34;
const MOVENET_KEYPOINTS = 17;
const MOVENET_ADJACENT_PAIRS = poseDetection.util.getAdjacentPairs(MOVENET_MODEL);

let detectorPromise = null;

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function getInputSize(input) {
  if (!input) return { width: 0, height: 0 };

  if (typeof input.videoWidth === 'number' && typeof input.videoHeight === 'number') {
    return { width: input.videoWidth, height: input.videoHeight };
  }

  if (typeof input.naturalWidth === 'number' && typeof input.naturalHeight === 'number') {
    return { width: input.naturalWidth, height: input.naturalHeight };
  }

  if (typeof input.width === 'number' && typeof input.height === 'number') {
    return { width: input.width, height: input.height };
  }

  if (Array.isArray(input.shape) && input.shape.length >= 2) {
    const [height, width] = input.shape;
    return { width: width ?? 0, height: height ?? 0 };
  }

  return { width: 0, height: 0 };
}

function flattenKeypoints(keypoints, width, height) {
  if (!width || !height) return null;
  if (!Array.isArray(keypoints) || keypoints.length === 0) return null;

  const output = [];

  for (let index = 0; index < MOVENET_KEYPOINTS; index += 1) {
    const keypoint = keypoints[index] ?? {};
    const x = typeof keypoint.x === 'number' ? keypoint.x : 0;
    const y = typeof keypoint.y === 'number' ? keypoint.y : 0;
    output.push(clamp01(x / width), clamp01(y / height));
  }

  return output;
}

async function loadMoveNetOnce() {
  if (detectorPromise) return detectorPromise;

  detectorPromise = (async () => {
    await initTensorFlowBackend();

    const detector = await poseDetection.createDetector(MOVENET_MODEL, MOVENET_CONFIG);
    return detector;
  })();

  return detectorPromise;
}

export function useMoveNet({ enabled = true } = {}) {
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

    loadMoveNetOnce()
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

  const getPoseFeatures = useCallback(async (input) => {
    const detector = detectorRef.current;
    if (!detector || !input) return { pose: null, features: null };

    const poses = await detector.estimatePoses(input, {
      maxPoses: 1,
      flipHorizontal: false,
    });

    const pose = poses?.[0] ?? null;
    if (!pose?.keypoints?.length) return { pose: null, features: null };

    const { width, height } = getInputSize(input);
    const features = flattenKeypoints(pose.keypoints, width, height);

    return { pose, features };
  }, []);

  return {
    status,
    error,
    detector: detectorRef.current,
    outputDim: MOVENET_OUTPUT_DIM,
    adjacentPairs: MOVENET_ADJACENT_PAIRS,
    getPoseFeatures,
  };
}
