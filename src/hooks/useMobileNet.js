import { useEffect, useRef, useState } from 'react';

import * as tf from '@tensorflow/tfjs';

const MOBILENET_URL =
  'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
const MOBILENET_IMAGE_SIZE = 224;

let mobilenetPromise = null;

async function loadMobileNetOnce() {
  if (mobilenetPromise) return mobilenetPromise;

  mobilenetPromise = (async () => {
    await tf.ready();

    try {
      await tf.setBackend('webgl');
      await tf.ready();
    } catch {
      // Fall back to the default backend.
    }

    const model = await tf.loadGraphModel(MOBILENET_URL, { fromTFHub: true });

    const warmupInput = tf.zeros([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3]);
    const warmupOutput = model.predict(warmupInput);
    warmupInput.dispose();

    const warmupTensor = Array.isArray(warmupOutput) ? warmupOutput[0] : warmupOutput;
    const outputDim = warmupTensor?.shape?.at(-1) ?? 1024;

    if (Array.isArray(warmupOutput)) {
      warmupOutput.forEach((tensor) => tensor.dispose());
    } else {
      warmupOutput.dispose();
    }

    return { model, outputDim };
  })();

  return mobilenetPromise;
}

export function useMobileNet({ enabled = true } = {}) {
  const [status, setStatus] = useState(enabled ? 'loading' : 'idle');
  const [error, setError] = useState(null);
  const modelRef = useRef(null);
  const outputDimRef = useRef(1024);

  useEffect(() => {
    if (!enabled) {
      setStatus('idle');
      return undefined;
    }

    let cancelled = false;
    setStatus('loading');

    loadMobileNetOnce()
      .then(({ model, outputDim }) => {
        if (cancelled) return;
        modelRef.current = model;
        outputDimRef.current = outputDim;
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

  return {
    status,
    error,
    model: modelRef.current,
    outputDim: outputDimRef.current,
    imageSize: MOBILENET_IMAGE_SIZE,
  };
}
