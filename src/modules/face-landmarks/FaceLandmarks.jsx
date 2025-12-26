import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import { useBluetooth } from '../../hooks/useBluetooth.js';
import { useFaceLandmarker } from '../../hooks/useFaceLandmarker.js';
import { useTransferLearning } from '../../hooks/useTransferLearning.js';
import { useWebcam } from '../../hooks/useWebcam.js';
import BluetoothModal from '../image-classification/components/BluetoothModal.jsx';
import ClassCard from '../image-classification/components/ClassCard.jsx';
import PreviewPanel from '../image-classification/components/PreviewPanel.jsx';
import TrainingPanel from '../image-classification/components/TrainingPanel.jsx';
import styles from '../image-classification/ImageClassification.module.css';

import { drawFaceLandmarks } from './drawFaceLandmarks.js';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

const STEP_ORDER = ['data', 'train', 'test'];
const PREVIEW_THROTTLE_MS = 35;

export default function FaceLandmarks() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [activeStep, setActiveStep] = useState('data');
  const [activeWebcamClassId, setActiveWebcamClassId] = useState(null);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(15);
  const [learningRate, setLearningRate] = useState(0.001);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeFace, setActiveFace] = useState(null);

  const captureVideoRef = useRef(null);
  const hasInitializedWebcamRef = useRef(false);
  const lastSentLabelRef = useRef(null);
  const faceQueueRef = useRef(Promise.resolve());
  const pendingDetectionsRef = useRef(0);
  const lastFaceRef = useRef({ face: null, features: null });

  const { connect, disconnect, send, isConnected, device } = useBluetooth();
  const {
    status: faceStatus,
    outputDim,
    connections: faceConnections,
    getFaceFeatures,
  } = useFaceLandmarker();

  const enqueueFaceDetection = useCallback(
    (input, { allowSkip = false, updatePreview = true } = {}) => {
      if (!input) return Promise.resolve({ face: null, features: null });

      if (allowSkip && pendingDetectionsRef.current > 0) {
        return Promise.resolve(lastFaceRef.current ?? { face: null, features: null });
      }

      pendingDetectionsRef.current += 1;

      const task = async () => {
        try {
          const result = await getFaceFeatures(input);
          lastFaceRef.current = result ?? { face: null, features: null };
          if (updatePreview) {
            setActiveFace(result?.face ?? null);
          }
          return result;
        } finally {
          pendingDetectionsRef.current -= 1;
        }
      };

      const next = faceQueueRef.current.then(task, task);
      faceQueueRef.current = next.catch(() => {});
      return next;
    },
    [getFaceFeatures],
  );

  const extractFaceFeatures = useCallback(
    async (input) => {
      const isLiveInput = Boolean(input && typeof input.videoWidth === 'number');
      const { features } = await enqueueFaceDetection(input, {
        allowSkip: false,
        updatePreview: isLiveInput,
      });
      return features;
    },
    [enqueueFaceDetection],
  );

  const drawFaceOverlay = useCallback(
    ({ ctx, canvas, video, width, height }) => {
      if (!activeFace?.landmarks?.length) return;
      drawFaceLandmarks({
        ctx,
        canvas,
        video,
        width,
        height,
        landmarks: activeFace.landmarks,
        connections: faceConnections,
      });
    },
    [activeFace, faceConnections],
  );

  const shouldEnableWebcam = useMemo(() => {
    if (activeStep === 'train') return false;
    if (activeStep === 'test') return true;
    return activeWebcamClassId !== null;
  }, [activeStep, activeWebcamClassId]);

  const modelStatusMessage = useMemo(() => {
    if (faceStatus === 'loading') {
      return 'Modell wird geladen. Gesichtsmerkmale sind gleich verfügbar.';
    }
    if (faceStatus === 'error') {
      return 'Modell konnte nicht geladen werden. Bitte Seite neu laden.';
    }
    if (faceStatus === 'idle') {
      return 'Modell ist noch nicht bereit.';
    }
    return null;
  }, [faceStatus]);

  const {
    status: webcamStatus,
    stream,
    isMirrored,
    canSwitchCamera,
    toggleFacingMode,
  } = useWebcam({ enabled: shouldEnableWebcam });

  useEffect(() => {
    if (!shouldEnableWebcam) return undefined;
    if (faceStatus !== 'ready') return undefined;

    let cancelled = false;
    let rafId = null;
    let lastTimestamp = 0;

    const loop = (timestamp) => {
      if (cancelled) return;

      const videoEl = captureVideoRef.current;
      if (videoEl?.readyState >= 2 && timestamp - lastTimestamp >= PREVIEW_THROTTLE_MS) {
        void enqueueFaceDetection(videoEl, { allowSkip: true, updatePreview: true });
        lastTimestamp = timestamp;
      }

      rafId = window.requestAnimationFrame(loop);
    };

    rafId = window.requestAnimationFrame(loop);

    return () => {
      cancelled = true;
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [enqueueFaceDetection, faceStatus, shouldEnableWebcam]);

  const {
    classes,
    probabilities,
    collectingClassIndex,
    isTraining,
    trainingPercent,
    isTrained,
    canCollect,
    canTrain,
    trainBlockers,
    addClass,
    updateClassName,
    clearDefaultClassName,
    normalizeClassName,
    startCollecting,
    stopCollecting,
    clearClassExamples,
    removeClass,
    train,
  } = useTransferLearning({
    extractFeatures: extractFaceFeatures,
    featureSize: outputDim,
    videoRef: captureVideoRef,
    modelStatus: faceStatus,
    isWebcamReady: webcamStatus === 'ready',
    epochs,
    batchSize,
    learningRate,
    predictionThrottleMs: 35,
    shouldPredict: activeStep === 'test',
    onTrainingComplete: () => setActiveStep('test'),
  });

  useEffect(() => {
    if (hasInitializedWebcamRef.current) return;
    if (!classes.length) return;
    setActiveWebcamClassId(classes[0].id);
    hasInitializedWebcamRef.current = true;
  }, [classes]);

  useEffect(() => {
    if (!shouldEnableWebcam) {
      setActiveFace(null);
    }
  }, [shouldEnableWebcam]);

  useEffect(() => {
    if (!isConnected || activeStep !== 'test' || !isTrained) {
      lastSentLabelRef.current = null;
      return;
    }
    if (!classes.length || !probabilities.length) return;

    const bestIndex = probabilities.reduce(
      (bestIdx, value, index) => (value > probabilities[bestIdx] ? index : bestIdx),
      0,
    );

    const bestLabel = classes[bestIndex]?.name?.trim();
    if (!bestLabel) return;
    if (lastSentLabelRef.current === bestLabel) return;

    lastSentLabelRef.current = bestLabel;
    send(bestLabel);
  }, [activeStep, classes, isConnected, isTrained, probabilities, send]);

  const showCameraSwitch = webcamStatus === 'ready' && canSwitchCamera;

  const activeStepIndex = Math.max(0, STEP_ORDER.indexOf(activeStep));

  const toggleWebcamForClass = useCallback((classId) => {
    setActiveWebcamClassId((prev) => (prev === classId ? null : classId));
  }, []);

  useEffect(() => {
    stopCollecting();
  }, [activeStep, activeWebcamClassId, stopCollecting]);

  const handleBleClick = useCallback(() => {
    if (isConnected) {
      disconnect();
      return;
    }
    setIsModalOpen(true);
  }, [disconnect, isConnected]);

  const handleSelectDevice = useCallback(
    (selectedDevice) => {
      connect(selectedDevice);
      setIsModalOpen(false);
    },
    [connect],
  );

  const handleAddClass = useCallback(() => {
    const nextClass = addClass();
    if (nextClass?.id) setActiveWebcamClassId(nextClass.id);
  }, [addClass]);

  const handleRemoveClass = useCallback(
    (classIndex, classId) => {
      if (classes.length <= 1) return;

      const nextClasses = classes.filter((_, index) => index !== classIndex);
      if (activeWebcamClassId === classId) {
        const nextActive =
          nextClasses[classIndex] ?? nextClasses[classIndex - 1] ?? nextClasses[0];
        setActiveWebcamClassId(nextActive?.id ?? null);
      }

      removeClass(classIndex);
    },
    [activeWebcamClassId, classes, removeClass],
  );

  return (
    <div className={styles['image-classification']}>
      <NavigationDrawer
        open={isNavOpen}
        onClose={() => setIsNavOpen(false)}
        drawerId="navigation-drawer"
      />

      <div className={styles['ic-shell']}>
        <header className={styles['ic-topbar']}>
          <button
            className={styles['ic-menu']}
            type="button"
            aria-label={isNavOpen ? 'Menü schließen' : 'Menü öffnen'}
            aria-controls="navigation-drawer"
            aria-expanded={isNavOpen}
            onClick={() => setIsNavOpen((prev) => !prev)}
          >
            <span className={styles['ic-menu-lines']} />
          </button>
          <div className={styles['ic-title']} aria-label="Gesichtsmerkmale">
            Gesichtsmerkmale
          </div>
        </header>

        <nav
          className={styles['ic-steps']}
          aria-label="Gesichtsmerkmale Schritte"
          style={{
            '--active-step': activeStepIndex,
            '--step-count': STEP_ORDER.length,
          }}
        >
          <span className={styles['ic-step-indicator']} aria-hidden="true" />
          <button
            className={cx(styles['ic-step'], activeStep === 'data' && styles.active)}
            type="button"
            onClick={() => setActiveStep('data')}
            disabled={isTraining}
          >
            <span className={styles['ic-step-number']}>1</span>
            Daten
          </button>
          <button
            className={cx(styles['ic-step'], activeStep === 'train' && styles.active)}
            type="button"
            onClick={() => setActiveStep('train')}
            disabled={isTraining}
          >
            <span className={styles['ic-step-number']}>2</span>
            Trainieren
          </button>
          <button
            className={cx(styles['ic-step'], activeStep === 'test' && styles.active)}
            type="button"
            onClick={() => setActiveStep('test')}
            disabled={!isTrained || isTraining}
          >
            <span className={styles['ic-step-number']}>3</span>
            Testen
          </button>
        </nav>

        <main className={styles['ic-stage']} data-step={activeStep}>
          {activeStep === 'data' ? (
            <section className={styles['classes-column']}>
              {modelStatusMessage ? (
                <div
                  className={cx(
                    styles['status-banner'],
                    faceStatus === 'error' && styles.error,
                  )}
                  role="status"
                  aria-live="polite"
                >
                  {modelStatusMessage}
                </div>
              ) : null}
              {classes.map((cls, index) => (
                <ClassCard
                  key={cls.id}
                  classNameValue={cls.name}
                  exampleCount={cls.exampleCount}
                  isCollecting={collectingClassIndex === index}
                  stream={stream}
                  showCameraSwitch={showCameraSwitch}
                  isMirrored={isMirrored}
                  onToggleCamera={toggleFacingMode}
                  canCollect={canCollect}
                  isWebcamEnabled={activeWebcamClassId === cls.id}
                  captureRef={activeWebcamClassId === cls.id ? captureVideoRef : null}
                  onToggleWebcam={() => toggleWebcamForClass(cls.id)}
                  overlayRenderer={drawFaceOverlay}
                  onClassNameChange={(event) => updateClassName(index, event.target.value)}
                  onClassNameFocus={() => clearDefaultClassName(index)}
                  onClassNameBlur={() => normalizeClassName(index)}
                  onCollectStart={() => startCollecting(index)}
                  onCollectStop={stopCollecting}
                  onClearExamples={() => clearClassExamples(index)}
                  canRemoveClass={classes.length > 1}
                  onRemoveClass={() => handleRemoveClass(index, cls.id)}
                />
              ))}

              <div
                className={cx(styles.card, styles.dashed)}
                role="button"
                tabIndex={0}
                onClick={handleAddClass}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return;
                  event.preventDefault();
                  handleAddClass();
                }}
              >
                <span className={styles['add-placeholder']}>＋ Klasse hinzufügen</span>
              </div>
            </section>
          ) : null}

          {activeStep === 'train' ? (
            <section className={styles['training-column']}>
              <TrainingPanel
                epochs={epochs}
                batchSize={batchSize}
                learningRate={learningRate}
                canTrain={canTrain}
                trainBlockers={trainBlockers}
                isTraining={isTraining}
                trainingPercent={trainingPercent}
                onTrain={train}
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
            <section className={styles['preview-column']}>
              <PreviewPanel
                stream={stream}
                classes={classes}
                probabilities={probabilities}
                showCameraSwitch={showCameraSwitch}
                isMirrored={isMirrored}
                onToggleCamera={toggleFacingMode}
                captureRef={captureVideoRef}
                overlayRenderer={drawFaceOverlay}
                onConnect={handleBleClick}
                isConnected={isConnected}
                deviceName={device?.name}
              />
            </section>
          ) : null}
        </main>
      </div>
      <BluetoothModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSelectDevice={handleSelectDevice}
      />
    </div>
  );
}
