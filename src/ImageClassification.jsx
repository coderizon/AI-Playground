import { forwardRef, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import * as tf from '@tensorflow/tfjs';
import { SwitchCamera, Video, X } from 'lucide-react';

import NavigationDrawer from './NavigationDrawer.jsx';
import './ImageClassification.css';

const MOBILENET_URL =
  'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
const MOBILENET_IMAGE_SIZE = 224;
const CAPTURE_INTERVAL_MS = 200;

const CLASS_COLORS = [
  '#3f73ff', // blue
  '#22c55e', // green
  '#f59e0b', // amber
  '#a855f7', // violet
  '#ef4444', // red
  '#14b8a6', // teal
  '#e11d48', // rose
  '#6366f1', // indigo
  '#84cc16', // lime
  '#0ea5e9', // sky
  '#f97316', // orange
  '#8b5cf6', // purple
];

function getClassColor(index) {
  if (index < CLASS_COLORS.length) return CLASS_COLORS[index];

  const hue = Math.round((index * 137.508) % 360);
  return `hsl(${hue} 82% 55%)`;
}

let mobilenetPromise = null;

async function loadMobileNetOnce() {
  if (mobilenetPromise) return mobilenetPromise;

  mobilenetPromise = (async () => {
    await tf.ready();

    try {
      await tf.setBackend('webgl');
      await tf.ready();
    } catch {
      // Fallback to default backend
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

const StreamVideo = forwardRef(function StreamVideo(
  { stream, className, ...props },
  forwardedRef,
) {
  const internalRef = useRef(null);

  const setRef = useCallback(
    (node) => {
      internalRef.current = node;

      if (typeof forwardedRef === 'function') {
        forwardedRef(node);
      } else if (forwardedRef) {
        forwardedRef.current = node;
      }
    },
    [forwardedRef],
  );

  useEffect(() => {
    if (!internalRef.current) return;
    internalRef.current.srcObject = stream ?? null;
  }, [stream]);

  return <video ref={setRef} className={className} autoPlay muted playsInline {...props} />;
});

function ClassCard({
  classNameValue,
  exampleCount,
  isCollecting,
  stream,
  showCameraSwitch,
  isMirrored,
  onToggleCamera,
  onClassNameChange,
  onClassNameFocus,
  onClassNameBlur,
  onCollectStart,
  onCollectStop,
  canCollect,
  isWebcamEnabled,
  onToggleWebcam,
}) {
  const [particles, setParticles] = useState([]);
  const [bumpAnimation, setBumpAnimation] = useState(false);
  const bumpTimeoutRef = useRef(null);
  const particleTimeoutsRef = useRef(new Set());
  const previousExampleCountRef = useRef(exampleCount);

  const addParticle = useCallback(() => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const scale = 0.8 + Math.random() * 0.4;
    const duration = 0.8 + Math.random() * 0.4;
    setParticles((prev) => [...prev, { id, scale, duration }]);

    const timeoutId = window.setTimeout(() => {
      setParticles((prev) => prev.filter((particle) => particle.id !== id));
      particleTimeoutsRef.current.delete(timeoutId);
    }, duration * 1000);

    particleTimeoutsRef.current.add(timeoutId);
  }, []);

  const triggerIncrementEffect = useCallback(() => {
    setBumpAnimation(true);
    if (bumpTimeoutRef.current) window.clearTimeout(bumpTimeoutRef.current);
    bumpTimeoutRef.current = window.setTimeout(() => setBumpAnimation(false), 100);
    addParticle();
  }, [addParticle]);

  useEffect(() => {
    if (exampleCount > previousExampleCountRef.current) {
      triggerIncrementEffect();
    }
    previousExampleCountRef.current = exampleCount;
  }, [exampleCount, triggerIncrementEffect]);

  useEffect(() => {
    return () => {
      if (bumpTimeoutRef.current) window.clearTimeout(bumpTimeoutRef.current);
      for (const timeoutId of particleTimeoutsRef.current) {
        window.clearTimeout(timeoutId);
      }
      particleTimeoutsRef.current.clear();
    };
  }, []);

  return (
    <div className="card class-card">
      <div className="card-header">
        <input
          className="class-name-input"
          value={classNameValue}
          onChange={onClassNameChange}
          onFocus={onClassNameFocus}
          onBlur={onClassNameBlur}
        />
        <div className="class-card-actions">
          <button
            className="ic-webcam-toggle"
            type="button"
            aria-label={isWebcamEnabled ? 'Webcam schließen' : 'Webcam öffnen'}
            aria-pressed={isWebcamEnabled}
            onClick={onToggleWebcam}
          >
            <Video className="ic-webcam-icon" aria-hidden="true" />
            {isWebcamEnabled ? <X className="ic-webcam-x" aria-hidden="true" /> : null}
          </button>
          <span className="dots" aria-hidden="true">
            ⋮
          </span>
        </div>
      </div>

      {isWebcamEnabled ? (
        <div className="webcam-panel visible">
          <div className={`capture-slot${isMirrored ? ' mirrored' : ''}`}>
            <StreamVideo stream={stream} />
            {showCameraSwitch ? (
              <button
                className="ic-camera-switch"
                type="button"
                onClick={onToggleCamera}
                aria-label="Kamera wechseln"
              >
                <SwitchCamera aria-hidden="true" />
              </button>
            ) : null}
          </div>

          <button
            className="dataCollector primary block"
            type="button"
            disabled={!canCollect}
            aria-pressed={isCollecting}
            onPointerDown={(event) => {
              if (!canCollect) return;
              event.currentTarget.setPointerCapture(event.pointerId);
              onCollectStart();
            }}
            onPointerUp={onCollectStop}
            onPointerCancel={onCollectStop}
            onLostPointerCapture={onCollectStop}
          >
            Zum Aufnehmen halten
          </button>
        </div>
      ) : null}

      <div className="count-row">
        <div className={`count-box${isCollecting ? ' recording' : ''}`}>
          <div className="count-bubbles" aria-hidden="true">
            {particles.map((particle) => (
              <div
                key={particle.id}
                className="count-bubble"
                style={{
                  animationDuration: `${particle.duration}s`,
                  '--scale': `${particle.scale}`,
                }}
              >
                +PNG
              </div>
            ))}
          </div>

          <div className="count-meta">
            <span className="count-label">Anzahl Beispiele</span>
          </div>
          <div className={`count-number${bumpAnimation ? ' bump' : ''}`}>{exampleCount}</div>
        </div>
      </div>
    </div>
  );
}

function TrainingPanel({
  epochs,
  batchSize,
  learningRate,
  onEpochsChange,
  onBatchSizeChange,
  onLearningRateChange,
  onTrain,
  canTrain,
  isTraining,
  trainingPercent,
}) {
  return (
    <div className="card training-card">
      <div className="card-header spaced">
        <h3>Training</h3>
        <button className="primary" type="button" onClick={onTrain} disabled={!canTrain}>
          Modell trainieren
        </button>
      </div>

      <div className={`training-progress${isTraining ? '' : ' hidden'}`}>
        <div className="training-progress-meta">
          <span>Training läuft...</span>
          <span>{trainingPercent}%</span>
        </div>
        <div className="training-progress-bar">
          <div className="training-progress-fill" style={{ width: `${trainingPercent}%` }} />
        </div>
      </div>

      <details open className="accordion">
        <summary>Erweitert</summary>
        <div className="form-grid">
          <label>
            <span>Epochen</span>
            <input type="number" min={1} step={1} value={epochs} onChange={onEpochsChange} />
          </label>
          <label>
            <span>Batchgröße</span>
            <input type="number" min={1} step={1} value={batchSize} onChange={onBatchSizeChange} />
          </label>
          <label>
            <span>Lernrate</span>
            <input
              type="number"
              min={0.000001}
              step={0.0001}
              value={learningRate}
              onChange={onLearningRateChange}
            />
          </label>
        </div>
      </details>
    </div>
  );
}

function PreviewPanel({ stream, classes, probabilities, showCameraSwitch, isMirrored, onToggleCamera }) {
  return (
    <div className="card preview-card">
      <div className="preview-body">
        <div className={`video-shell${isMirrored ? ' mirrored' : ''}`}>
          <StreamVideo stream={stream} />
          {showCameraSwitch ? (
            <button
              className="ic-camera-switch"
              type="button"
              onClick={onToggleCamera}
              aria-label="Kamera wechseln"
            >
              <SwitchCamera aria-hidden="true" />
            </button>
          ) : null}
        </div>

        <div className="preview-output">
          <div className="preview-output-header">
            <span>Ausgabe</span>
          </div>

          <div id="probabilityList">
            {classes.map((cls, index) => {
              const probability = probabilities[index] ?? 0;
              const percent = Math.round(probability * 100);
              const color = getClassColor(index);

              return (
                <div key={cls.id} className="probability-row">
                  <div className="probability-row-header">
                    <span className="probability-label">{cls.name}</span>
                    <span className="probability-value">{percent}%</span>
                  </div>
                  <div
                    className="probability-bar"
                    role="meter"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={percent}
                  >
                    <div
                      className="probability-bar-fill"
                      style={{ width: `${percent}%`, backgroundColor: color }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function makeDefaultClass(index) {
  return {
    id: `class-${Date.now()}-${Math.random().toString(16).slice(2)}-${index}`,
    name: getDefaultClassName(index),
    exampleCount: 0,
  };
}

export default function ImageClassification() {
  const initialClass = useMemo(() => makeDefaultClass(0), []);
  const [classes, setClasses] = useState(() => [initialClass]);

  const [isNavOpen, setIsNavOpen] = useState(false);
  const [activeStep, setActiveStep] = useState('data');
  const [activeWebcamClassId, setActiveWebcamClassId] = useState(() => initialClass.id);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(15);
  const [learningRate, setLearningRate] = useState(0.001);

  const [mobilenetStatus, setMobilenetStatus] = useState('loading');
  const [webcamStatus, setWebcamStatus] = useState('idle');
  const [cameraFacingMode, setCameraFacingMode] = useState('user');
  const [canSwitchCamera, setCanSwitchCamera] = useState(false);
  const [pendingExampleCount, setPendingExampleCount] = useState(0);

  const [isTraining, setIsTraining] = useState(false);
  const [trainingPercent, setTrainingPercent] = useState(0);
  const [isTrained, setIsTrained] = useState(false);
  const [probabilities, setProbabilities] = useState(() => [0]);
  const [collectingClassIndex, setCollectingClassIndex] = useState(null);

  const mobilenetRef = useRef(null);
  const mobilenetOutputDimRef = useRef(1024);
  const trainingInputsRef = useRef([]);
  const trainingLabelsRef = useRef([]);
  const pendingExamplesRef = useRef([]);
  const captureCanvasRef = useRef(null);
  const captureCanvasContextRef = useRef(null);
  const transferModelRef = useRef(null);
  const captureVideoRef = useRef(null);
  const streamRef = useRef(null);
  const captureIntervalRef = useRef(null);
  const predictLoopRafRef = useRef(null);
  const lastPredictionUpdateRef = useRef(0);

  const stream = streamRef.current;
  const isMirrored = cameraFacingMode === 'user';
  const showCameraSwitch = webcamStatus === 'ready' && canSwitchCamera;

  const toggleCameraFacingMode = useCallback(() => {
    setCameraFacingMode((prev) => (prev === 'user' ? 'environment' : 'user'));
  }, []);

  const shouldEnableWebcam = useMemo(() => {
    if (activeStep === 'train') return false;
    if (activeStep === 'test') return true;
    return activeWebcamClassId !== null;
  }, [activeStep, activeWebcamClassId]);

  const toggleWebcamForClass = useCallback((classId) => {
    setActiveWebcamClassId((prev) => (prev === classId ? null : classId));
  }, []);

  const canCollect = mobilenetStatus !== 'error' && webcamStatus === 'ready' && !isTraining;

  const canTrain = useMemo(() => {
    if (isTraining) return false;
    if (mobilenetStatus !== 'ready') return false;
    if (pendingExampleCount > 0) return false;
    if (classes.length < 2) return false;

    const hasExamples = classes.every((cls) => cls.exampleCount > 0);
    return hasExamples;
  }, [classes, isTraining, mobilenetStatus, pendingExampleCount]);

  useEffect(() => {
    let cancelled = false;

    setMobilenetStatus('loading');
    loadMobileNetOnce()
      .then(({ model, outputDim }) => {
        if (cancelled) return;
        mobilenetRef.current = model;
        mobilenetOutputDimRef.current = outputDim;
        setMobilenetStatus('ready');
      })
      .catch((error) => {
        if (cancelled) return;
        console.error(error);
        setMobilenetStatus('error');
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!captureCanvasRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = MOBILENET_IMAGE_SIZE;
      canvas.height = MOBILENET_IMAGE_SIZE;
      captureCanvasRef.current = canvas;
      captureCanvasContextRef.current = canvas.getContext('2d', { willReadFrequently: true });
    }

    return () => {
      captureCanvasRef.current = null;
      captureCanvasContextRef.current = null;
      pendingExamplesRef.current = [];
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function startWebcam() {
      if (!navigator?.mediaDevices?.getUserMedia) {
        setWebcamStatus('error');
        return;
      }

      setWebcamStatus('loading');

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: cameraFacingMode },
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        setWebcamStatus('ready');
      } catch (error) {
        console.error(error);
        if (!cancelled && cameraFacingMode === 'environment') {
          setCameraFacingMode('user');
          return;
        }
        setWebcamStatus('error');
      }
    }

    if (!shouldEnableWebcam) {
      if (captureIntervalRef.current) {
        window.clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }

      setCollectingClassIndex(null);
      setWebcamStatus('disabled');
      return () => {
        cancelled = true;
      };
    }

    startWebcam();

    return () => {
      cancelled = true;

      if (captureIntervalRef.current) {
        window.clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }

      if (predictLoopRafRef.current) {
        window.cancelAnimationFrame(predictLoopRafRef.current);
        predictLoopRafRef.current = null;
      }

      const stream = streamRef.current;
      streamRef.current = null;
      if (stream) stream.getTracks().forEach((track) => track.stop());
    };
  }, [cameraFacingMode, shouldEnableWebcam]);

  useEffect(() => {
    if (webcamStatus !== 'ready') {
      setCanSwitchCamera(false);
      return;
    }

    let cancelled = false;

    const update = async () => {
      if (!navigator?.mediaDevices?.enumerateDevices) {
        setCanSwitchCamera(false);
        return;
      }

      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (cancelled) return;

        const videoInputs = devices.filter((device) => device.kind === 'videoinput');
        const uniqueIds = new Set(videoInputs.map((device) => device.deviceId).filter(Boolean));
        const count = uniqueIds.size || videoInputs.length;
        setCanSwitchCamera(count > 1);
      } catch (error) {
        console.error(error);
        if (!cancelled) setCanSwitchCamera(false);
      }
    };

    update();

    navigator.mediaDevices?.addEventListener?.('devicechange', update);

    return () => {
      cancelled = true;
      navigator.mediaDevices?.removeEventListener?.('devicechange', update);
    };
  }, [webcamStatus]);

  useEffect(() => {
    return () => {
      if (transferModelRef.current) {
        transferModelRef.current.dispose();
        transferModelRef.current = null;
      }

      for (const tensor of trainingInputsRef.current) tensor.dispose();
      trainingInputsRef.current = [];
      trainingLabelsRef.current = [];
    };
  }, []);

  useEffect(() => {
    if (classes.length === probabilities.length) return;
    setProbabilities((prev) => {
      const next = Array.from({ length: classes.length }, (_, index) => prev[index] ?? 0);
      return next;
    });
  }, [classes.length, probabilities.length]);

  const captureExampleFrame = useCallback((videoEl) => {
    let ctx = captureCanvasContextRef.current;
    if (!ctx) {
      if (typeof document === 'undefined') return null;
      const canvas = document.createElement('canvas');
      canvas.width = MOBILENET_IMAGE_SIZE;
      canvas.height = MOBILENET_IMAGE_SIZE;
      captureCanvasRef.current = canvas;
      ctx = canvas.getContext('2d', { willReadFrequently: true });
      captureCanvasContextRef.current = ctx;
    }
    if (!ctx) return null;

    try {
      ctx.drawImage(videoEl, 0, 0, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE);
      return ctx.getImageData(0, 0, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE);
    } catch (error) {
      console.error(error);
      return null;
    }
  }, []);

  const flushPendingExamples = useCallback(async () => {
    const mobilenet = mobilenetRef.current;
    if (!mobilenet) return;

    const pending = pendingExamplesRef.current;
    if (!pending.length) return;

    pendingExamplesRef.current = [];
    setPendingExampleCount(pending.length);

    for (const { classIndex, imageData } of pending) {
      const features = tf.tidy(() => {
        const image = tf.browser.fromPixels(imageData);
        const normalized = image.toFloat().div(255).expandDims(0);

        const activation = mobilenet.predict(normalized);
        const activationTensor = Array.isArray(activation) ? activation[0] : activation;

        return activationTensor.squeeze();
      });

      trainingInputsRef.current.push(features);
      trainingLabelsRef.current.push(classIndex);

      await new Promise((resolve) => window.requestAnimationFrame(resolve));
    }

    setPendingExampleCount(0);
  }, []);

  useEffect(() => {
    if (mobilenetStatus !== 'ready') return;
    void flushPendingExamples();
  }, [mobilenetStatus, flushPendingExamples]);

  const collectExample = useCallback(
    (classIndex) => {
      const videoEl = captureVideoRef.current;

      if (!videoEl) return;
      if (videoEl.readyState < 2) return;

      const mobilenet = mobilenetRef.current;

      if (!mobilenet) {
        const imageData = captureExampleFrame(videoEl);
        if (!imageData) return;
        pendingExamplesRef.current.push({ classIndex, imageData });
        setPendingExampleCount(pendingExamplesRef.current.length);
      } else {
        const features = tf.tidy(() => {
          const image = tf.browser.fromPixels(videoEl);
          const resized = tf.image.resizeBilinear(
            image,
            [MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE],
            true,
          );
          const normalized = resized.toFloat().div(255);
          const batched = normalized.expandDims(0);

          const activation = mobilenet.predict(batched);
          const activationTensor = Array.isArray(activation) ? activation[0] : activation;

          return activationTensor.squeeze();
        });

        if (!features) return;
        trainingInputsRef.current.push(features);
        trainingLabelsRef.current.push(classIndex);
      }

      setClasses((prev) =>
        prev.map((cls, index) =>
          index === classIndex ? { ...cls, exampleCount: cls.exampleCount + 1 } : cls,
        ),
      );
    },
    [captureExampleFrame],
  );

  const stopCollecting = useCallback(() => {
    if (captureIntervalRef.current) {
      window.clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    setCollectingClassIndex(null);
  }, []);

  useEffect(() => {
    stopCollecting();
  }, [activeStep, stopCollecting]);

  useEffect(() => {
    stopCollecting();
  }, [activeWebcamClassId, stopCollecting]);

  const startCollecting = useCallback(
    (classIndex) => {
      if (!canCollect) return;

      stopCollecting();

      if (transferModelRef.current) {
        transferModelRef.current.dispose();
        transferModelRef.current = null;
      }
      if (isTrained) {
        setIsTrained(false);
        setProbabilities((prev) => prev.map(() => 0));
      }

      setCollectingClassIndex(classIndex);

      collectExample(classIndex);
      captureIntervalRef.current = window.setInterval(() => collectExample(classIndex), CAPTURE_INTERVAL_MS);
    },
    [canCollect, collectExample, isTrained, stopCollecting],
  );

  const addClass = useCallback(() => {
    const nextClass = makeDefaultClass(classes.length);
    setClasses((prev) => [...prev, nextClass]);
    setActiveWebcamClassId(nextClass.id);
    setIsTrained(false);

    if (transferModelRef.current) {
      transferModelRef.current.dispose();
      transferModelRef.current = null;
    }
  }, [classes.length]);

  const handleTrain = useCallback(async () => {
    if (!canTrain) return;

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
      return;
    }

    const numClasses = classes.length;
    const inputDim = mobilenetOutputDimRef.current;

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
      setActiveStep('test');
    } catch (error) {
      console.error(error);
      model.dispose();
    } finally {
      setIsTraining(false);

      if (xs) xs.dispose();
      if (ys) ys.dispose();
      if (labelTensor) labelTensor.dispose();
    }
  }, [batchSize, canTrain, classes.length, epochs, flushPendingExamples, learningRate, stopCollecting]);

  useEffect(() => {
    if (activeStep !== 'test') return;
    if (!isTrained) return;
    if (!transferModelRef.current) return;
    if (!mobilenetRef.current) return;
    if (!captureVideoRef.current) return;
    if (isTraining) return;

    let cancelled = false;

    const loop = () => {
      if (cancelled) return;

      const videoEl = captureVideoRef.current;
      const mobilenet = mobilenetRef.current;
      const model = transferModelRef.current;

      if (videoEl?.readyState >= 2 && mobilenet && model) {
        const now = performance.now();
        if (now - lastPredictionUpdateRef.current > 100) {
          const next = tf.tidy(() => {
            const image = tf.browser.fromPixels(videoEl);
            const resized = tf.image.resizeBilinear(
              image,
              [MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE],
              true,
            );
            const normalized = resized.toFloat().div(255);
            const batched = normalized.expandDims(0);

            const activation = mobilenet.predict(batched);
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
  }, [activeStep, isTrained, isTraining, classes.length]);

  return (
    <div className="image-classification">
      <NavigationDrawer
        open={isNavOpen}
        onClose={() => setIsNavOpen(false)}
        drawerId="navigation-drawer"
      />
      <StreamVideo ref={captureVideoRef} stream={stream} className="ic-hidden-video" />

      <div className="ic-shell">
        <header className="ic-topbar">
          <button
            className="ic-menu"
            type="button"
            aria-label={isNavOpen ? 'Menü schließen' : 'Menü öffnen'}
            aria-controls="navigation-drawer"
            aria-expanded={isNavOpen}
            onClick={() => setIsNavOpen((prev) => !prev)}
          >
            <span className="ic-menu-lines" />
          </button>
          <div className="ic-title" aria-label="Bildklassifikation">
            Bildklassifikation
          </div>
        </header>

        <nav className="ic-steps" aria-label="Bildklassifikation Schritte">
          <button
            className={`ic-step${activeStep === 'data' ? ' active' : ''}`}
            type="button"
            onClick={() => setActiveStep('data')}
            disabled={isTraining}
          >
            <span className="ic-step-number">1</span>
            Daten
          </button>
          <button
            className={`ic-step${activeStep === 'train' ? ' active' : ''}`}
            type="button"
            onClick={() => setActiveStep('train')}
            disabled={isTraining}
          >
            <span className="ic-step-number">2</span>
            Trainieren
          </button>
          <button
            className={`ic-step${activeStep === 'test' ? ' active' : ''}`}
            type="button"
            onClick={() => setActiveStep('test')}
            disabled={!isTrained || isTraining}
          >
            <span className="ic-step-number">3</span>
            Testen
          </button>
        </nav>

        <main className="ic-stage" data-step={activeStep}>
          {activeStep === 'data' ? (
            <section className="classes-column">
              {classes.map((cls, index) => (
                <ClassCard
                  key={cls.id}
                  classNameValue={cls.name}
                  exampleCount={cls.exampleCount}
                  isCollecting={collectingClassIndex === index}
                  stream={stream}
                  showCameraSwitch={showCameraSwitch}
                  isMirrored={isMirrored}
                  onToggleCamera={toggleCameraFacingMode}
                  canCollect={canCollect}
                  isWebcamEnabled={activeWebcamClassId === cls.id}
                  onToggleWebcam={() => toggleWebcamForClass(cls.id)}
                  onClassNameChange={(event) => {
                    const nextName = event.target.value;
                    setClasses((prev) =>
                      prev.map((item, itemIndex) =>
                        itemIndex === index ? { ...item, name: nextName } : item,
                      ),
                    );
                  }}
                  onClassNameFocus={() => {
                    const defaultName = getDefaultClassName(index);
                    const legacyDefaultName = `Class ${index + 1}`;

                    setClasses((prev) =>
                      prev.map((item, itemIndex) => {
                        if (itemIndex !== index) return item;
                        if (item.name !== defaultName && item.name !== legacyDefaultName) return item;
                        return { ...item, name: '' };
                      }),
                    );
                  }}
                  onClassNameBlur={() => {
                    const defaultName = getDefaultClassName(index);

                    setClasses((prev) =>
                      prev.map((item, itemIndex) => {
                        if (itemIndex !== index) return item;

                        const nextName = item.name.trim();
                        return { ...item, name: nextName.length ? nextName : defaultName };
                      }),
                    );
                  }}
                  onCollectStart={() => startCollecting(index)}
                  onCollectStop={stopCollecting}
                />
              ))}

              <div
                className="card dashed"
                role="button"
                tabIndex={0}
                onClick={addClass}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return;
                  event.preventDefault();
                  addClass();
                }}
              >
                <span className="add-placeholder">＋ Klasse hinzufügen</span>
              </div>
            </section>
          ) : null}

          {activeStep === 'train' ? (
            <section className="training-column">
              <TrainingPanel
                epochs={epochs}
                batchSize={batchSize}
                learningRate={learningRate}
                canTrain={canTrain}
                isTraining={isTraining}
                trainingPercent={trainingPercent}
                onTrain={handleTrain}
                onEpochsChange={(event) => {
                  const next = Number(event.target.value);
                  setEpochs(Number.isFinite(next) && next >= 1 ? Math.floor(next) : 1);
                }}
                onBatchSizeChange={(event) => {
                  const next = Number(event.target.value);
                  setBatchSize(Number.isFinite(next) && next >= 1 ? Math.floor(next) : 1);
                }}
                onLearningRateChange={(event) => {
                  const next = Number(event.target.value);
                  setLearningRate(Number.isFinite(next) && next > 0 ? next : 0.000001);
                }}
              />
            </section>
          ) : null}

          {activeStep === 'test' ? (
            <section className="preview-column">
              <PreviewPanel
                stream={stream}
                classes={classes}
                probabilities={probabilities}
                showCameraSwitch={showCameraSwitch}
                isMirrored={isMirrored}
                onToggleCamera={toggleCameraFacingMode}
              />
            </section>
          ) : null}
        </main>
      </div>
    </div>
  );
}
