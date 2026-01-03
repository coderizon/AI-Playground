import { useCallback, useEffect, useRef, useState } from 'react';

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

const FACE_LANDMARKER_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const FACE_LANDMARKER_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_BLENDSHAPE_COUNT = 52;
const FACE_LANDMARK_CONNECTIONS = FaceLandmarker.FACE_LANDMARKS_TESSELATION ?? [];

let landmarkerPromise = null;

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function getBlendshapeScores(blendshapes) {
  const categories = blendshapes?.[0]?.categories;
  if (!Array.isArray(categories) || categories.length === 0) return null;

  const sorted = [...categories].sort((a, b) => {
    const aIndex = Number.isFinite(a?.index) ? a.index : null;
    const bIndex = Number.isFinite(b?.index) ? b.index : null;
    if (aIndex !== null && bIndex !== null && aIndex !== bIndex) return aIndex - bIndex;

    const aName = a?.categoryName ?? '';
    const bName = b?.categoryName ?? '';
    return aName.localeCompare(bName);
  });

  const scores = sorted.map((category) => clamp01(category?.score ?? 0));

  if (scores.length >= FACE_BLENDSHAPE_COUNT) {
    return scores.slice(0, FACE_BLENDSHAPE_COUNT);
  }

  return [...scores, ...Array(FACE_BLENDSHAPE_COUNT - scores.length).fill(0)];
}

async function loadFaceLandmarkerOnce() {
  if (landmarkerPromise) return landmarkerPromise;

  landmarkerPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(FACE_LANDMARKER_WASM_URL);
    const landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: FACE_LANDMARKER_MODEL_URL,
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      refineLandmarks: true,
    });

    return landmarker;
  })();

  return landmarkerPromise;
}

export function useFaceLandmarker({ enabled = true } = {}) {
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

    loadFaceLandmarkerOnce()
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

  const getFaceFeatures = useCallback(
    async (input) => {
      const landmarker = landmarkerRef.current;
      if (!landmarker || !input) return { face: null, features: null };

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

      const landmarks = result?.faceLandmarks?.[0] ?? null;
      if (!landmarks?.length) return { face: null, features: null };

      const features = getBlendshapeScores(result?.faceBlendshapes);

      return {
        face: { landmarks },
        features,
      };
    },
    [ensureRunningMode],
  );

  return {
    status,
    error,
    landmarker: landmarkerRef.current,
    outputDim: FACE_BLENDSHAPE_COUNT,
    connections: FACE_LANDMARK_CONNECTIONS,
    getFaceFeatures,
  };
}
