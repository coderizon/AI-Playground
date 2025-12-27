import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import * as tf from '@tensorflow/tfjs';
import * as speechCommands from '@tensorflow-models/speech-commands';

const DEFAULT_SAMPLE_DURATION_SEC = 1;
const DEFAULT_SNIPPET_DURATION_SEC = 0.25;
const DEFAULT_RECORDING_DURATION_SEC = 10;
const AUDIO_TRACK_CONSTRAINTS = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
};

function getDefaultClassName(index) {
  return `Klasse ${index + 1}`;
}

function createClassId() {
  return `audio-class-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function makeDefaultClass(index, id = createClassId()) {
  return {
    id,
    name: getDefaultClassName(index),
  };
}

export function useAudioTransferLearning({
  enabled = true,
  epochs = 30,
  batchSize = 16,
  learningRate = 0.001,
  sampleDurationSec = DEFAULT_SAMPLE_DURATION_SEC,
  snippetDurationSec = DEFAULT_SNIPPET_DURATION_SEC,
  audioTrackConstraints = AUDIO_TRACK_CONSTRAINTS,
  onTrainingComplete,
} = {}) {
  const initialClass = useMemo(() => makeDefaultClass(0), []);
  const [classes, setClasses] = useState(() => [initialClass]);
  const [counts, setCounts] = useState(() => [0]);
  const [probabilities, setProbabilities] = useState(() => [0]);
  const [collectingClassId, setCollectingClassId] = useState(null);
  const [recordingProgress, setRecordingProgress] = useState(0);
  const [recordingSecondsLeft, setRecordingSecondsLeft] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingPercent, setTrainingPercent] = useState(0);
  const [lossHistory, setLossHistory] = useState([]);
  const [isTrained, setIsTrained] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [status, setStatus] = useState(enabled ? 'loading' : 'idle');
  const [error, setError] = useState(null);

  const recognizerRef = useRef(null);
  const transferRecognizerRef = useRef(null);
  const spectrogramRef = useRef(null);
  const collectingSessionRef = useRef({ cancelled: false });
  const progressIntervalRef = useRef(null);

  useEffect(() => {
    if (!enabled) {
      setStatus('idle');
      return undefined;
    }

    let cancelled = false;
    setStatus('loading');

    (async () => {
      await tf.ready();
      const recognizer = speechCommands.create('BROWSER_FFT');
      await recognizer.ensureModelLoaded();
      if (cancelled) return;

      recognizerRef.current = recognizer;
      transferRecognizerRef.current = recognizer.createTransfer('audio-transfer');
      setStatus('ready');
    })().catch((loadError) => {
      if (cancelled) return;
      console.error(loadError);
      setError(loadError);
      setStatus('error');
    });

    return () => {
      cancelled = true;
    };
  }, [enabled]);

  useEffect(() => {
    setCounts((prev) => classes.map((_, index) => prev[index] ?? 0));
    setProbabilities((prev) => classes.map((_, index) => prev[index] ?? 0));
  }, [classes]);

  const resetTrainingState = useCallback(() => {
    if (isTrained) setIsTrained(false);
    setProbabilities(classes.map(() => 0));
  }, [classes, isTrained]);

  const updateCountsFromRecognizer = useCallback(() => {
    const recognizer = transferRecognizerRef.current;
    if (!recognizer) return;

    let exampleCounts = {};
    try {
      exampleCounts = recognizer.countExamples();
    } catch (countError) {
      exampleCounts = {};
    }

    setCounts(classes.map((cls) => exampleCounts[cls.id] ?? 0));
  }, [classes]);

  const updateSpectrogram = useCallback((spectrogram) => {
    if (!spectrogram) return;
    spectrogramRef.current = spectrogram;
  }, []);

  const stopListening = useCallback(async () => {
    const recognizer = transferRecognizerRef.current;
    if (!recognizer || !recognizer.isListening()) return;

    try {
      await recognizer.stopListening();
    } catch (stopError) {
      console.error(stopError);
    } finally {
      setIsListening(false);
    }
  }, []);

  const startListening = useCallback(async () => {
    if (!isTrained || isTraining) return false;
    const recognizer = transferRecognizerRef.current;
    if (!recognizer) return false;
    if (recognizer.isListening()) return true;

    setError(null);

    try {
      await recognizer.listen(
        async (result) => {
          if (result?.spectrogram) updateSpectrogram(result.spectrogram);
          if (result?.scores) {
            const labels = recognizer.wordLabels?.() ?? [];
            const scoreArray = Array.isArray(result.scores) ? result.scores[0] : result.scores;
            const scores = Array.from(scoreArray ?? []);
            const next = classes.map((cls) => {
              const labelIndex = labels.indexOf(cls.id);
              return labelIndex >= 0 ? scores[labelIndex] ?? 0 : 0;
            });
            setProbabilities(next);
          }
        },
        {
          includeSpectrogram: true,
          probabilityThreshold: 0,
          invokeCallbackOnNoiseAndUnknown: true,
          audioTrackConstraints,
        },
      );
      setIsListening(true);
      return true;
    } catch (listenError) {
      console.error(listenError);
      setError(listenError);
      setIsListening(false);
      return false;
    }
  }, [audioTrackConstraints, classes, isTraining, isTrained, updateSpectrogram]);

  const stopCollecting = useCallback(() => {
    collectingSessionRef.current.cancelled = true;
    if (progressIntervalRef.current) {
      window.clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
    setCollectingClassId(null);
    setRecordingProgress(0);
    setRecordingSecondsLeft(0);
  }, []);

  const startCollecting = useCallback(
    async (classId, durationSeconds = DEFAULT_RECORDING_DURATION_SEC) => {
      if (!classId) return false;
      if (status !== 'ready') return false;
      if (isTraining) return false;

      const recognizer = transferRecognizerRef.current;
      if (!recognizer) return false;

      await stopListening();
      resetTrainingState();
      setError(null);

      collectingSessionRef.current = { cancelled: false };
      const session = collectingSessionRef.current;

      setCollectingClassId(classId);
      setRecordingProgress(0);
      setRecordingSecondsLeft(durationSeconds);

      const startTimestamp = performance.now();
      if (progressIntervalRef.current) window.clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = window.setInterval(() => {
        const elapsed = (performance.now() - startTimestamp) / 1000;
        const percent = Math.min(100, Math.round((elapsed / durationSeconds) * 100));
        setRecordingProgress(percent);
        setRecordingSecondsLeft(Math.max(0, Math.ceil(durationSeconds - elapsed)));
      }, 120);

      try {
        while (!session.cancelled) {
          const elapsed = (performance.now() - startTimestamp) / 1000;
          if (elapsed >= durationSeconds) break;

          const spectrogram = await recognizer.collectExample(classId, {
            durationSec: sampleDurationSec,
            snippetDurationSec,
            onSnippet: async (snippet) => updateSpectrogram(snippet),
            audioTrackConstraints,
          });
          updateSpectrogram(spectrogram);
          updateCountsFromRecognizer();
        }
      } catch (collectError) {
        console.error(collectError);
        setError(collectError);
      } finally {
        stopCollecting();
      }

      return true;
    },
    [
      audioTrackConstraints,
      isTraining,
      resetTrainingState,
      sampleDurationSec,
      snippetDurationSec,
      status,
      stopCollecting,
      stopListening,
      updateCountsFromRecognizer,
      updateSpectrogram,
    ],
  );

  const addClass = useCallback(() => {
    const nextId = createClassId();
    setClasses((prev) => [...prev, makeDefaultClass(prev.length, nextId)]);
    resetTrainingState();
    return { id: nextId };
  }, [resetTrainingState]);

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

  const clearClassExamples = useCallback(
    (classId) => {
      if (!classId) return;
      const recognizer = transferRecognizerRef.current;
      if (!recognizer) return;

      stopCollecting();
      resetTrainingState();

      try {
        const examples = recognizer.getExamples(classId) || [];
        for (const example of examples) {
          recognizer.removeExample(example.uid);
        }
      } catch (clearError) {
        // Ignore if there were no examples to clear.
      }

      updateCountsFromRecognizer();
    },
    [resetTrainingState, stopCollecting, updateCountsFromRecognizer],
  );

  const removeClass = useCallback(
    (classIndex, classId) => {
      if (classes.length <= 1) return false;
      if (classIndex < 0 || classIndex >= classes.length) return false;

      clearClassExamples(classId);
      setClasses((prev) => prev.filter((_, index) => index !== classIndex));
      resetTrainingState();
      return true;
    },
    [classes.length, clearClassExamples, resetTrainingState],
  );

  const canCollect = useMemo(() => {
    if (status !== 'ready') return false;
    if (isTraining) return false;
    if (collectingClassId) return false;
    return true;
  }, [collectingClassId, isTraining, status]);

  const canTrain = useMemo(() => {
    if (status !== 'ready') return false;
    if (isTraining) return false;
    if (collectingClassId) return false;
    if (classes.length < 2) return false;
    return counts.every((count) => count > 0);
  }, [classes.length, collectingClassId, counts, isTraining, status]);

  const trainBlockers = useMemo(() => {
    const blockers = [];

    if (isTraining) blockers.push('Training läuft bereits.');
    if (status !== 'ready') {
      const modelMessage =
        status === 'loading'
          ? 'Modell wird noch geladen.'
          : status === 'error'
            ? 'Modell konnte nicht geladen werden.'
            : 'Modell ist noch nicht bereit.';
      blockers.push(modelMessage);
    }
    if (collectingClassId) blockers.push('Aufnahme läuft noch.');
    if (classes.length < 2) blockers.push('Mindestens zwei Klassen erforderlich.');

    const emptyClasses = classes.reduce((acc, cls, index) => {
      if ((counts[index] ?? 0) > 0) return acc;
      const label = cls.name?.trim() || getDefaultClassName(index);
      acc.push(label);
      return acc;
    }, []);

    if (emptyClasses.length > 0) {
      blockers.push(`Leere Klassen: ${emptyClasses.join(', ')}.`);
    }

    return blockers;
  }, [classes, collectingClassId, counts, isTraining, status]);

  const train = useCallback(async () => {
    if (!canTrain) return false;

    const recognizer = transferRecognizerRef.current;
    if (!recognizer) return false;

    await stopListening();
    stopCollecting();
    setIsTraining(true);
    setTrainingPercent(0);
    setLossHistory([]);
    setIsTrained(false);

    try {
      const history = await recognizer.train({
        epochs,
        batchSize,
        optimizer: tf.train.adam(learningRate),
        callback: {
          onEpochEnd: (epoch, logs) => {
            const percent = Math.round(((epoch + 1) / epochs) * 100);
            setTrainingPercent(percent);
            if (typeof logs?.loss === 'number') {
              setLossHistory((prev) => [...prev, logs.loss]);
            }
          },
        },
      });

      if (history) {
        const histories = Array.isArray(history) ? history : [history];
        const lossValues = histories.flatMap((item) => item?.history?.loss ?? []);
        if (lossValues.length) setLossHistory(lossValues);
      }

      setIsTrained(true);
      setTrainingPercent(100);
      if (typeof onTrainingComplete === 'function') onTrainingComplete();
      return true;
    } catch (trainError) {
      console.error(trainError);
      setError(trainError);
      return false;
    } finally {
      setIsTraining(false);
    }
  }, [
    batchSize,
    canTrain,
    epochs,
    learningRate,
    onTrainingComplete,
    stopCollecting,
    stopListening,
  ]);

  useEffect(() => {
    return () => {
      stopCollecting();
      void stopListening();
    };
  }, [stopCollecting, stopListening]);

  return {
    status,
    error,
    classes,
    counts,
    probabilities,
    collectingClassId,
    recordingProgress,
    recordingSecondsLeft,
    isTraining,
    trainingPercent,
    lossHistory,
    isTrained,
    isListening,
    spectrogramRef,
    canCollect,
    canTrain,
    trainBlockers,
    addClass,
    updateClassName,
    clearDefaultClassName,
    normalizeClassName,
    clearClassExamples,
    removeClass,
    train,
    startCollecting,
    stopCollecting,
    startListening,
    stopListening,
  };
}
