import { useCallback, useEffect, useRef, useState } from 'react';

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const HAND_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const HAND_LANDMARKER_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const HAND_LANDMARK_COUNT = 21;
const HAND_FEATURE_SIZE = 84;
const HAND_FEATURE_STRIDE = HAND_LANDMARK_COUNT * 2;
const HAND_LANDMARK_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS ?? [];

let landmarkerPromise = null;

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function createEmptyFeatures() {
  return Array(HAND_FEATURE_SIZE).fill(0);
}

function normalizeHandedness(label) {
  if (!label) return null;
  const normalized = String(label).toLowerCase();
  if (normalized.includes('left')) return 'Left';
  if (normalized.includes('right')) return 'Right';
  return null;
}

function getHandednessLabel(handednessEntry) {
  if (!handednessEntry) return null;
  const categories = Array.isArray(handednessEntry?.categories)
    ? handednessEntry.categories
    : Array.isArray(handednessEntry)
      ? handednessEntry
      : [];
  if (!categories.length) return null;
  const top = categories[0];
  return normalizeHandedness(top?.categoryName ?? top?.displayName ?? top?.label);
}

function getAverageX(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length === 0) return null;

  let sum = 0;
  let count = 0;

  for (const landmark of landmarks) {
    const x = clamp01(landmark?.x ?? 0);
    sum += x;
    count += 1;
  }

  return count ? sum / count : null;
}

function sortHands(hands) {
  const order = { Left: 0, Right: 1 };

  return [...hands].sort((a, b) => {
    const aKey = order[a.handedness] ?? 2;
    const bKey = order[b.handedness] ?? 2;
    if (aKey !== bKey) return aKey - bKey;

    if (
      Number.isFinite(a.averageX) &&
      Number.isFinite(b.averageX) &&
      a.averageX !== b.averageX
    ) {
      return a.averageX - b.averageX;
    }

    return 0;
  });
}

function flattenHandLandmarks(landmarks) {
  const output = [];

  for (let index = 0; index < HAND_LANDMARK_COUNT; index += 1) {
    const landmark = landmarks?.[index] ?? {};
    const x = clamp01(landmark?.x ?? 0);
    const y = clamp01(landmark?.y ?? 0);
    output.push(x, y);
  }

  return output;
}

function buildHandFeatures(hands) {
  const features = createEmptyFeatures();
  const limit = Math.min(hands.length, 2);

  for (let handIndex = 0; handIndex < limit; handIndex += 1) {
    const landmarks = hands[handIndex]?.landmarks;
    const flattened = flattenHandLandmarks(landmarks);
    const offset = handIndex * HAND_FEATURE_STRIDE;

    for (let index = 0; index < flattened.length; index += 1) {
      features[offset + index] = flattened[index];
    }
  }

  return features;
}

async function loadHandLandmarkerOnce() {
  if (landmarkerPromise) return landmarkerPromise;

  landmarkerPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(HAND_LANDMARKER_WASM_URL);
    const landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_LANDMARKER_MODEL_URL,
      },
      runningMode: 'VIDEO',
      numHands: 2,
    });
    console.info('[HandLandmarker] Loaded model:', HAND_LANDMARKER_MODEL_URL);
    return landmarker;
  })();

  return landmarkerPromise;
}

export function useHandLandmarker({ enabled = true } = {}) {
  const [status, setStatus] = useState(enabled ? 'loading' : 'idle');
  const [error, setError] = useState(null);
  const landmarkerRef = useRef(null);
  const runningModeRef = useRef('VIDEO');

  useEffect(() => {
    if (!enabled) {
      setStatus('idle');
      return undefined;
    }

    let cancelled = false;
    setStatus('loading');

    loadHandLandmarkerOnce()
      .then((landmarker) => {
        if (cancelled) return;
        landmarkerRef.current = landmarker;
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

  const ensureRunningMode = useCallback(async (mode) => {
    const landmarker = landmarkerRef.current;
    if (!landmarker) return;
    if (runningModeRef.current === mode) return;

    await landmarker.setOptions({ runningMode: mode });
    runningModeRef.current = mode;
  }, []);

  const getHandFeatures = useCallback(
    async (input) => {
      const landmarker = landmarkerRef.current;
      if (!landmarker || !input) return { hands: null, features: null };

      const isVideo = typeof input.videoWidth === 'number';
      let result = null;

      if (isVideo) {
        await ensureRunningMode('VIDEO');
        const timestamp =
          typeof performance !== 'undefined' && typeof performance.now === 'function'
            ? performance.now()
            : Date.now();
        result = landmarker.detectForVideo(input, timestamp);
      } else {
        await ensureRunningMode('IMAGE');
        result = landmarker.detect(input);
      }

      const rawHands = result?.landmarks ?? result?.handLandmarks ?? [];
      const rawHandedness = result?.handednesses ?? result?.handedness ?? [];

      const hands = Array.isArray(rawHands) ? rawHands : [];
      const handednesses = Array.isArray(rawHandedness) ? rawHandedness : [];

      const handEntries = hands
        .map((landmarks, index) => {
          if (!Array.isArray(landmarks) || landmarks.length === 0) return null;
          return {
            landmarks,
            handedness: getHandednessLabel(handednesses[index]),
            averageX: getAverageX(landmarks),
          };
        })
        .filter(Boolean);

      const sortedHands = sortHands(handEntries);
      const features = buildHandFeatures(sortedHands);

      return {
        hands: sortedHands.map(({ landmarks, handedness }) => ({ landmarks, handedness })),
        features,
      };
    },
    [ensureRunningMode],
  );

  return {
    status,
    error,
    landmarker: landmarkerRef.current,
    outputDim: HAND_FEATURE_SIZE,
    connections: HAND_LANDMARK_CONNECTIONS,
    getHandFeatures,
  };
}
