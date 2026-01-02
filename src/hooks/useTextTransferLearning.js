import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import * as tf from '@tensorflow/tfjs';
import { initTensorFlowBackend } from '../utils/tensorflow-init.js';

const DEFAULT_FEATURE_SIZE = 384;

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

function createClassId() {
  return `text-class-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function makeDefaultClass(index, id = createClassId()) {
  return {
    id,
    name: getDefaultClassName(index),
    exampleCount: 0,
  };
}

function normalizeFeatureOutput(output) {
  if (!output) return null;
  if (output instanceof tf.Tensor) return output;
  if (Array.isArray(output)) return tf.tensor1d(output);
  if (ArrayBuffer.isView(output)) return tf.tensor1d(Array.from(output));
  return null;
}

export function useTextTransferLearning({
  extractFeatures,
  modelStatus = 'idle',
  featureSize = DEFAULT_FEATURE_SIZE,
  epochs = 50,
  batchSize = 16,
  learningRate = 0.001,
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
  const transferModelRef = useRef(null);
  const collectInFlightRef = useRef(false);
  const predictionRequestRef = useRef(0);
  const tfReadyRef = useRef(null);

  const ensureTensorFlowReady = useCallback(async () => {
    if (!tfReadyRef.current) {
      tfReadyRef.current = initTensorFlowBackend();
    }
    return tfReadyRef.current;
  }, []);

  const hasFeatureExtractor = typeof extractFeatures === 'function';
  const canCollect = modelStatus === 'ready' && !isTraining && hasFeatureExtractor;

  useEffect(() => {
    void ensureTensorFlowReady();
  }, [ensureTensorFlowReady]);

  const canTrain = useMemo(() => {
    if (isTraining) return false;
    if (modelStatus !== 'ready') return false;
    if (pendingExampleCount > 0) return false;
    if (classes.length < 2) return false;

    return classes.every((cls) => cls.exampleCount > 0);
  }, [classes, isTraining, modelStatus, pendingExampleCount]);

  const trainBlockers = useMemo(() => {
    const blockers = [];

    if (isTraining) {
      blockers.push('Training läuft bereits.');
    }

    if (modelStatus !== 'ready') {
      const modelMessage =
        modelStatus === 'loading'
          ? 'Modell wird noch geladen.'
          : modelStatus === 'error'
            ? 'Modell konnte nicht geladen werden.'
            : 'Modell ist noch nicht bereit.';
      blockers.push(modelMessage);
    }

    if (pendingExampleCount > 0) {
      blockers.push(`Beispiele werden noch verarbeitet (${pendingExampleCount}).`);
    }

    if (classes.length < 2) {
      blockers.push('Mindestens zwei Klassen erforderlich.');
    }

    const emptyClasses = classes.reduce((acc, cls, index) => {
      if (cls.exampleCount > 0) return acc;
      const label = cls.name?.trim() || getDefaultClassName(index);
      acc.push(label);
      return acc;
    }, []);

    if (emptyClasses.length > 0) {
      blockers.push(`Leere Klassen: ${emptyClasses.join(', ')}.`);
    }

    return blockers;
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
    const nextId = createClassId();

    setClasses((prev) => [...prev, makeDefaultClass(prev.length, nextId)]);
    resetTrainingState();

    return { id: nextId };
  }, [resetTrainingState]);

  useEffect(() => {
    if (classes.length === probabilities.length) return;
    setProbabilities((prev) =>
      Array.from({ length: classes.length }, (_, index) => prev[index] ?? 0),
    );
  }, [classes.length, probabilities.length]);

  const flushPendingExamples = useCallback(async () => {
    if (!extractFeatures) return;

    const pending = pendingExamplesRef.current;
    if (!pending.length) return;

    await ensureTensorFlowReady();

    pendingExamplesRef.current = [];
    setPendingExampleCount(pending.length);

    for (const { classIndex, text } of pending) {
      let features = null;

      try {
        const output = await extractFeatures(text);
        features = tf.tidy(() => normalizeFeatureOutput(output));
      } catch (error) {
        console.error(error);
      }

      if (!features) continue;

      trainingInputsRef.current.push(features);
      trainingLabelsRef.current.push(classIndex);

      await new Promise((resolve) => window.requestAnimationFrame(resolve));
    }

    setPendingExampleCount(0);
  }, [ensureTensorFlowReady, extractFeatures]);

  useEffect(() => {
    if (modelStatus !== 'ready') return;
    void flushPendingExamples();
  }, [modelStatus, flushPendingExamples]);

  const collectExample = useCallback(
    async (classIndex, text) => {
      if (collectInFlightRef.current) return false;
      if (!hasFeatureExtractor) return false;
      if (modelStatus === 'error') return false;

      const normalizedText =
        typeof text === 'string' ? text.trim() : String(text ?? '').trim();
      if (!normalizedText) return false;
      if (isTraining) return false;

      collectInFlightRef.current = true;
      setCollectingClassIndex(classIndex);

      try {
        resetTrainingState();

        if (modelStatus !== 'ready') {
          pendingExamplesRef.current.push({ classIndex, text: normalizedText });
          setPendingExampleCount(pendingExamplesRef.current.length);
        } else {
          let features = null;
          await ensureTensorFlowReady();

          try {
            const output = await extractFeatures(normalizedText);
            features = tf.tidy(() => normalizeFeatureOutput(output));
          } catch (error) {
            console.error(error);
          }

          if (!features) return false;
          trainingInputsRef.current.push(features);
          trainingLabelsRef.current.push(classIndex);
        }

        setClasses((prev) =>
          prev.map((cls, index) =>
            index === classIndex ? { ...cls, exampleCount: cls.exampleCount + 1 } : cls,
          ),
        );

        return true;
      } finally {
        collectInFlightRef.current = false;
        setCollectingClassIndex(null);
      }
    },
    [
      ensureTensorFlowReady,
      extractFeatures,
      hasFeatureExtractor,
      isTraining,
      modelStatus,
      resetTrainingState,
    ],
  );

  const clearClassExamples = useCallback(
    (classIndex) => {
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
    [resetTrainingState],
  );

  const removeClass = useCallback(
    (classIndex) => {
      if (classes.length <= 1) return false;
      if (classIndex < 0 || classIndex >= classes.length) return false;

      resetTrainingState();

      if (pendingExamplesRef.current.length) {
        const nextPending = [];
        for (const example of pendingExamplesRef.current) {
          if (example.classIndex === classIndex) continue;
          const nextIndex =
            example.classIndex > classIndex ? example.classIndex - 1 : example.classIndex;
          nextPending.push({ ...example, classIndex: nextIndex });
        }
        pendingExamplesRef.current = nextPending;
        setPendingExampleCount(nextPending.length);
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

          nextLabels.push(label > classIndex ? label - 1 : label);
          nextInputs.push(input);
        }

        trainingLabelsRef.current = nextLabels;
        trainingInputsRef.current = nextInputs;
      }

      setClasses((prev) => prev.filter((_, index) => index !== classIndex));
      return true;
    },
    [classes.length, resetTrainingState],
  );

  const train = useCallback(async () => {
    if (!canTrain) return false;

    await ensureTensorFlowReady();
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
      console.warn('[TextClassification] No training data available.');
      return false;
    }

    const numClasses = classes.length;
    const inferredInputDim = inputs[0]?.shape?.[0] ?? inputs[0]?.shape?.at(-1);
    const inputDim = inferredInputDim ?? featureSize;

    if (!inputDim) {
      setIsTraining(false);
      console.warn('[TextClassification] Unable to infer input dimension.');
      return false;
    }

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
    ensureTensorFlowReady,
    epochs,
    featureSize,
    flushPendingExamples,
    learningRate,
    onTrainingComplete,
  ]);

  const predict = useCallback(
    async (text) => {
      if (!isTrained) return false;
      if (!transferModelRef.current) return false;
      if (!hasFeatureExtractor) return false;
      if (modelStatus !== 'ready') return false;
      if (isTraining) return false;

      const normalizedText =
        typeof text === 'string' ? text.trim() : String(text ?? '').trim();
      if (!normalizedText) {
        setProbabilities((prev) => prev.map(() => 0));
        return false;
      }

      const requestId = predictionRequestRef.current + 1;
      predictionRequestRef.current = requestId;

      let next = null;

      try {
        await ensureTensorFlowReady();

        let features = null;
        try {
          const output = await extractFeatures(normalizedText);
          features = tf.tidy(() => normalizeFeatureOutput(output));
        } catch (error) {
          console.error(error);
        }

        if (!features) return false;

        next = tf.tidy(() => {
          const batched = features.expandDims(0);
          const logits = transferModelRef.current.predict(batched);
          const logitsTensor = Array.isArray(logits) ? logits[0] : logits;

          const prediction = logitsTensor.squeeze();
          return Array.from(prediction.dataSync());
        });

        features.dispose();
      } catch (error) {
        console.error(error);
        return false;
      }

      if (!next) return false;
      if (requestId !== predictionRequestRef.current) return false;

      setProbabilities(next);
      return true;
    },
    [
      ensureTensorFlowReady,
      extractFeatures,
      hasFeatureExtractor,
      isTraining,
      isTrained,
      modelStatus,
    ],
  );

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
    trainBlockers,
    addClass,
    updateClassName,
    clearDefaultClassName,
    normalizeClassName,
    collectExample,
    clearClassExamples,
    removeClass,
    train,
    predict,
  };
}
