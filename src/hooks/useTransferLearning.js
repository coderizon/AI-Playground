import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import * as tf from '@tensorflow/tfjs';

const CAPTURE_INTERVAL_MS = 200;
const PREDICTION_THROTTLE_MS = 100;
const CAPTURE_HAPTIC_MS = 12;

function createTransferModel({ inputDim, numClasses, learningRate }) {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [inputDim], units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

function getDefaultClassName(index) {
  return `Klasse ${index + 1}`;
}

function makeDefaultClass(index) {
  return {
    id: `class-${Date.now()}-${Math.random().toString(16).slice(2)}-${index}`,
    name: getDefaultClassName(index),
    exampleCount: 0,
  };
}

function drawSquareFrame(videoEl, ctx, imageSize) {
  const width = videoEl?.videoWidth ?? 0;
  const height = videoEl?.videoHeight ?? 0;
  if (!width || !height) return false;

  const size = Math.min(width, height);
  const sx = Math.max(0, Math.floor((width - size) / 2));
  const sy = Math.max(0, Math.floor((height - size) / 2));

  ctx.drawImage(videoEl, sx, sy, size, size, 0, 0, imageSize, imageSize);
  return true;
}

export function useTransferLearning({
  featureExtractor,
  featureSize = 1024,
  imageSize = 224,
  videoRef,
  modelStatus = 'idle',
  isWebcamReady = false,
  epochs = 50,
  batchSize = 16,
  learningRate = 0.001,
  captureIntervalMs = CAPTURE_INTERVAL_MS,
  predictionThrottleMs = PREDICTION_THROTTLE_MS,
  shouldPredict = false,
  onTrainingComplete,
} = {}) {
  const initialClass = useMemo(() => makeDefaultClass(0), []);
  const [classes, setClasses] = useState(() => [initialClass]);
  const [probabilities, setProbabilities] = useState(() => [0]);
  const [collectingClassIndex, setCollectingClassIndex] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingPercent, setTrainingPercent] = useState(0);
  const [isTrained, setIsTrained] = useState(false);
  const [pendingExampleCount, setPendingExampleCount] = useState(0);

  const trainingInputsRef = useRef([]);
  const trainingLabelsRef = useRef([]);
  const pendingExamplesRef = useRef([]);
  const captureCanvasRef = useRef(null);
  const captureCanvasContextRef = useRef(null);
  const transferModelRef = useRef(null);
  const captureIntervalRef = useRef(null);
  const predictLoopRafRef = useRef(null);
  const lastPredictionUpdateRef = useRef(0);

  const canCollect = modelStatus !== 'error' && isWebcamReady && !isTraining;

  const canTrain = useMemo(() => {
    if (isTraining) return false;
    if (modelStatus !== 'ready') return false;
    if (pendingExampleCount > 0) return false;
    if (classes.length < 2) return false;

    return classes.every((cls) => cls.exampleCount > 0);
  }, [classes, isTraining, modelStatus, pendingExampleCount]);

  const resetTrainingState = useCallback(() => {
    if (transferModelRef.current) {
      transferModelRef.current.dispose();
      transferModelRef.current = null;
    }

    if (isTrained) {
      setIsTrained(false);
      setProbabilities((prev) => prev.map(() => 0));
    }
  }, [isTrained]);

  const updateClassName = useCallback((index, nextName) => {
    setClasses((prev) =>
      prev.map((cls, clsIndex) => (clsIndex === index ? { ...cls, name: nextName } : cls)),
    );
  }, []);

  const clearDefaultClassName = useCallback((index) => {
    const defaultName = getDefaultClassName(index);
    const legacyDefaultName = `Class ${index + 1}`;

    setClasses((prev) =>
      prev.map((cls, clsIndex) => {
        if (clsIndex !== index) return cls;
        if (cls.name !== defaultName && cls.name !== legacyDefaultName) return cls;
        return { ...cls, name: '' };
      }),
    );
  }, []);

  const normalizeClassName = useCallback((index) => {
    const defaultName = getDefaultClassName(index);

    setClasses((prev) =>
      prev.map((cls, clsIndex) => {
        if (clsIndex !== index) return cls;
        const nextName = cls.name.trim();
        return { ...cls, name: nextName.length ? nextName : defaultName };
      }),
    );
  }, []);

  const addClass = useCallback(() => {
    let nextClass = null;

    setClasses((prev) => {
      nextClass = makeDefaultClass(prev.length);
      return [...prev, nextClass];
    });

    resetTrainingState();

    return nextClass;
  }, [resetTrainingState]);

  useEffect(() => {
    if (classes.length === probabilities.length) return;
    setProbabilities((prev) =>
      Array.from({ length: classes.length }, (_, index) => prev[index] ?? 0),
    );
  }, [classes.length, probabilities.length]);

  const drawVideoToCaptureCanvas = useCallback(
    (videoEl) => {
      if (!videoEl) return null;

      let canvas = captureCanvasRef.current;
      let ctx = captureCanvasContextRef.current;

      if (!canvas) {
        if (typeof document === 'undefined') return null;
        canvas = document.createElement('canvas');
        canvas.width = imageSize;
        canvas.height = imageSize;
        captureCanvasRef.current = canvas;
      }

      if (!ctx && canvas) {
        ctx = canvas.getContext('2d', { willReadFrequently: true });
        captureCanvasContextRef.current = ctx;
      }

      if (!ctx) return null;
      if (!drawSquareFrame(videoEl, ctx, imageSize)) return null;

      return { canvas, ctx };
    },
    [imageSize],
  );

  const captureExampleFrame = useCallback(
    (videoEl) => {
      const capture = drawVideoToCaptureCanvas(videoEl);
      if (!capture) return null;

      try {
        return capture.ctx.getImageData(0, 0, imageSize, imageSize);
      } catch (error) {
        console.error(error);
        return null;
      }
    },
    [drawVideoToCaptureCanvas, imageSize],
  );

  const flushPendingExamples = useCallback(async () => {
    if (!featureExtractor) return;

    const pending = pendingExamplesRef.current;
    if (!pending.length) return;

    pendingExamplesRef.current = [];
    setPendingExampleCount(pending.length);

    for (const { classIndex, imageData } of pending) {
      const features = tf.tidy(() => {
        const image = tf.browser.fromPixels(imageData);
        const normalized = image.toFloat().div(255).expandDims(0);

        const activation = featureExtractor.predict(normalized);
        const activationTensor = Array.isArray(activation) ? activation[0] : activation;

        return activationTensor.squeeze();
      });

      trainingInputsRef.current.push(features);
      trainingLabelsRef.current.push(classIndex);

      await new Promise((resolve) => window.requestAnimationFrame(resolve));
    }

    setPendingExampleCount(0);
  }, [featureExtractor]);

  useEffect(() => {
    if (modelStatus !== 'ready') return;
    void flushPendingExamples();
  }, [modelStatus, flushPendingExamples]);

  const collectExample = useCallback(
    (classIndex) => {
      const videoEl = videoRef?.current;

      if (!videoEl) return;
      if (videoEl.readyState < 2) return;

      const triggerCaptureHaptic = () => {
        if (typeof navigator === 'undefined') return;
        if (typeof navigator.vibrate !== 'function') return;
        navigator.vibrate(CAPTURE_HAPTIC_MS);
      };

      if (!featureExtractor || modelStatus !== 'ready') {
        const imageData = captureExampleFrame(videoEl);
        if (!imageData) return;
        pendingExamplesRef.current.push({ classIndex, imageData });
        setPendingExampleCount(pendingExamplesRef.current.length);
        triggerCaptureHaptic();
      } else {
        const capture = drawVideoToCaptureCanvas(videoEl);
        if (!capture) return;

        const features = tf.tidy(() => {
          const image = tf.browser.fromPixels(capture.canvas);
          const normalized = image.toFloat().div(255);
          const batched = normalized.expandDims(0);

          const activation = featureExtractor.predict(batched);
          const activationTensor = Array.isArray(activation) ? activation[0] : activation;

          return activationTensor.squeeze();
        });

        if (!features) return;
        trainingInputsRef.current.push(features);
        trainingLabelsRef.current.push(classIndex);
        triggerCaptureHaptic();
      }

      setClasses((prev) =>
        prev.map((cls, index) =>
          index === classIndex ? { ...cls, exampleCount: cls.exampleCount + 1 } : cls,
        ),
      );
    },
    [captureExampleFrame, drawVideoToCaptureCanvas, featureExtractor, modelStatus, videoRef],
  );

  const stopCollecting = useCallback(() => {
    if (captureIntervalRef.current) {
      window.clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    setCollectingClassIndex(null);
  }, []);

  const startCollecting = useCallback(
    (classIndex) => {
      if (!canCollect) return;

      stopCollecting();
      resetTrainingState();

      setCollectingClassIndex(classIndex);

      collectExample(classIndex);
      captureIntervalRef.current = window.setInterval(
        () => collectExample(classIndex),
        captureIntervalMs,
      );
    },
    [canCollect, captureIntervalMs, collectExample, resetTrainingState, stopCollecting],
  );

  const clearClassExamples = useCallback(
    (classIndex) => {
      if (collectingClassIndex === classIndex) {
        stopCollecting();
      }

      resetTrainingState();

      if (pendingExamplesRef.current.length) {
        const nextPending = pendingExamplesRef.current.filter(
          (example) => example.classIndex !== classIndex,
        );

        if (nextPending.length !== pendingExamplesRef.current.length) {
          pendingExamplesRef.current = nextPending;
          setPendingExampleCount(nextPending.length);
        }
      }

      if (trainingLabelsRef.current.length) {
        const nextInputs = [];
        const nextLabels = [];

        for (let index = 0; index < trainingLabelsRef.current.length; index += 1) {
          const label = trainingLabelsRef.current[index];
          const input = trainingInputsRef.current[index];

          if (label === classIndex) {
            if (input) input.dispose();
            continue;
          }

          nextLabels.push(label);
          nextInputs.push(input);
        }

        trainingLabelsRef.current = nextLabels;
        trainingInputsRef.current = nextInputs;
      }

      setClasses((prev) =>
        prev.map((cls, index) =>
          index === classIndex ? { ...cls, exampleCount: 0 } : cls,
        ),
      );
    },
    [collectingClassIndex, resetTrainingState, stopCollecting],
  );

  const train = useCallback(async () => {
    if (!canTrain) return false;

    stopCollecting();
    await flushPendingExamples();
    setIsTraining(true);
    setTrainingPercent(0);
    setIsTrained(false);

    if (transferModelRef.current) {
      transferModelRef.current.dispose();
      transferModelRef.current = null;
    }

    const inputs = trainingInputsRef.current;
    const labels = trainingLabelsRef.current;

    if (!inputs.length) {
      setIsTraining(false);
      console.warn('[ImageClassification] No training data available.');
      return false;
    }

    const numClasses = classes.length;
    const inputDim = featureSize;

    const model = createTransferModel({ inputDim, numClasses, learningRate });

    let xs;
    let ys;
    let labelTensor;

    try {
      xs = tf.stack(inputs);
      labelTensor = tf.tensor1d(labels, 'int32');
      ys = tf.oneHot(labelTensor, numClasses);

      await model.fit(xs, ys, {
        batchSize: Math.min(batchSize, inputs.length),
        epochs,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch) => {
            setTrainingPercent(Math.round(((epoch + 1) / epochs) * 100));
          },
        },
      });

      transferModelRef.current = model;
      setIsTrained(true);
      setTrainingPercent(100);
      if (typeof onTrainingComplete === 'function') onTrainingComplete();

      return true;
    } catch (error) {
      console.error(error);
      model.dispose();
      return false;
    } finally {
      setIsTraining(false);

      if (xs) xs.dispose();
      if (ys) ys.dispose();
      if (labelTensor) labelTensor.dispose();
    }
  }, [
    batchSize,
    canTrain,
    classes.length,
    epochs,
    featureSize,
    flushPendingExamples,
    learningRate,
    onTrainingComplete,
    stopCollecting,
  ]);

  useEffect(() => {
    if (!shouldPredict) return undefined;
    if (!isTrained) return undefined;
    if (!transferModelRef.current) return undefined;
    if (!featureExtractor) return undefined;
    if (!videoRef?.current) return undefined;
    if (isTraining) return undefined;

    let cancelled = false;

    const loop = () => {
      if (cancelled) return;

      const videoEl = videoRef?.current;
      const model = transferModelRef.current;

      if (videoEl?.readyState >= 2 && featureExtractor && model) {
        const now = performance.now();
        if (now - lastPredictionUpdateRef.current > predictionThrottleMs) {
          const capture = drawVideoToCaptureCanvas(videoEl);
          if (capture) {
            const next = tf.tidy(() => {
              const image = tf.browser.fromPixels(capture.canvas);
              const normalized = image.toFloat().div(255);
              const batched = normalized.expandDims(0);

              const activation = featureExtractor.predict(batched);
              const activationTensor = Array.isArray(activation) ? activation[0] : activation;
              const logits = model.predict(activationTensor);
              const logitsTensor = Array.isArray(logits) ? logits[0] : logits;

              const prediction = logitsTensor.squeeze();
              return Array.from(prediction.dataSync());
            });

            setProbabilities(next);
            lastPredictionUpdateRef.current = now;
          }
        }
      }

      predictLoopRafRef.current = window.requestAnimationFrame(loop);
    };

    predictLoopRafRef.current = window.requestAnimationFrame(loop);

    return () => {
      cancelled = true;
      if (predictLoopRafRef.current) {
        window.cancelAnimationFrame(predictLoopRafRef.current);
        predictLoopRafRef.current = null;
      }
    };
  }, [
    drawVideoToCaptureCanvas,
    featureExtractor,
    isTraining,
    isTrained,
    predictionThrottleMs,
    shouldPredict,
    videoRef,
  ]);

  useEffect(() => {
    return () => {
      if (transferModelRef.current) {
        transferModelRef.current.dispose();
        transferModelRef.current = null;
      }

      for (const tensor of trainingInputsRef.current) tensor.dispose();
      trainingInputsRef.current = [];
      trainingLabelsRef.current = [];
      pendingExamplesRef.current = [];
      captureCanvasRef.current = null;
      captureCanvasContextRef.current = null;
    };
  }, []);

  return {
    classes,
    initialClassId: initialClass.id,
    probabilities,
    collectingClassIndex,
    isTraining,
    trainingPercent,
    isTrained,
    pendingExampleCount,
    canCollect,
    canTrain,
    addClass,
    updateClassName,
    clearDefaultClassName,
    normalizeClassName,
    startCollecting,
    stopCollecting,
    clearClassExamples,
    train,
  };
}
