import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import { useMobileNet } from '../../hooks/useMobileNet.js';
import { useTransferLearning } from '../../hooks/useTransferLearning.js';
import { useWebcam } from '../../hooks/useWebcam.js';
import ClassCard from './components/ClassCard.jsx';
import PreviewPanel from './components/PreviewPanel.jsx';
import TrainingPanel from './components/TrainingPanel.jsx';
import styles from './ImageClassification.module.css';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function ImageClassification() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [activeStep, setActiveStep] = useState('data');
  const [activeWebcamClassId, setActiveWebcamClassId] = useState(null);
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(15);
  const [learningRate, setLearningRate] = useState(0.001);

  const captureVideoRef = useRef(null);
  const hasInitializedWebcamRef = useRef(false);

  const { status: mobilenetStatus, model: mobilenet, outputDim, imageSize } = useMobileNet();

  const shouldEnableWebcam = useMemo(() => {
    if (activeStep === 'train') return false;
    if (activeStep === 'test') return true;
    return activeWebcamClassId !== null;
  }, [activeStep, activeWebcamClassId]);

  const {
    status: webcamStatus,
    stream,
    isMirrored,
    canSwitchCamera,
    toggleFacingMode,
  } = useWebcam({ enabled: shouldEnableWebcam });

  const {
    classes,
    probabilities,
    collectingClassIndex,
    isTraining,
    trainingPercent,
    isTrained,
    canCollect,
    canTrain,
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
    featureExtractor: mobilenet,
    featureSize: outputDim,
    imageSize,
    videoRef: captureVideoRef,
    modelStatus: mobilenetStatus,
    isWebcamReady: webcamStatus === 'ready',
    epochs,
    batchSize,
    learningRate,
    shouldPredict: activeStep === 'test',
    onTrainingComplete: () => setActiveStep('test'),
  });

  useEffect(() => {
    if (hasInitializedWebcamRef.current) return;
    if (!classes.length) return;
    setActiveWebcamClassId(classes[0].id);
    hasInitializedWebcamRef.current = true;
  }, [classes]);

  const showCameraSwitch = webcamStatus === 'ready' && canSwitchCamera;

  const toggleWebcamForClass = useCallback((classId) => {
    setActiveWebcamClassId((prev) => (prev === classId ? null : classId));
  }, []);

  useEffect(() => {
    stopCollecting();
  }, [activeStep, activeWebcamClassId, stopCollecting]);

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
          <div className={styles['ic-title']} aria-label="Bildklassifikation">
            Bildklassifikation
          </div>
        </header>

        <nav className={styles['ic-steps']} aria-label="Bildklassifikation Schritte">
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
              />
            </section>
          ) : null}
        </main>
      </div>
    </div>
  );
}
