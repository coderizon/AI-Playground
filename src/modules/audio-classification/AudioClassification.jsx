import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import { useBluetooth } from '../../hooks/useBluetooth.js';
import { useAudioTransferLearning } from '../../hooks/useAudioTransferLearning.js';
import BluetoothModal from '../image-classification/components/BluetoothModal.jsx';
import BluetoothButton from '../image-classification/components/BluetoothButton.jsx';
import TrainingPanel from '../image-classification/components/TrainingPanel.jsx';
import AudioClassCard from './components/AudioClassCard.jsx';
import LossSparkline from './components/LossSparkline.jsx';
import SpectrogramCanvas from './components/SpectrogramCanvas.jsx';
import styles from '../image-classification/ImageClassification.module.css';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

const STEP_ORDER = ['data', 'train', 'test'];

const CLASS_COLORS = [
  '#3f73ff',
  '#22c55e',
  '#f59e0b',
  '#a855f7',
  '#ef4444',
  '#14b8a6',
  '#e11d48',
  '#6366f1',
  '#84cc16',
  '#0ea5e9',
  '#f97316',
  '#8b5cf6',
];

function getClassColor(index) {
  if (index < CLASS_COLORS.length) return CLASS_COLORS[index];
  const hue = Math.round((index * 137.508) % 360);
  return `hsl(${hue} 82% 55%)`;
}

export default function AudioClassification() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [activeStep, setActiveStep] = useState('data');
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const hasStartedListeningRef = useRef(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const lastSentLabelRef = useRef(null);

  const { connect, disconnect, send, isConnected, device } = useBluetooth();

  const {
    status,
    error,
    classes,
    counts,
    probabilities,
    collectingClassId,
    recordingProgress,
    recordingSecondsLeft,
    recordingDurationSeconds,
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
  } = useAudioTransferLearning({
    epochs,
    batchSize,
    learningRate,
    onTrainingComplete: () => setActiveStep('test'),
  });

  useEffect(() => {
    if (activeStep !== 'test') {
      void stopListening();
      hasStartedListeningRef.current = false;
      return;
    }
    if (!isTrained || isTraining || isListening || hasStartedListeningRef.current) return;
    hasStartedListeningRef.current = true;
    void startListening();
  }, [activeStep, isListening, isTraining, isTrained, startListening, stopListening]);

  useEffect(() => {
    if (activeStep !== 'data') {
      stopCollecting();
    }
  }, [activeStep, stopCollecting]);

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

  const modelStatusMessage = useMemo(() => {
    if (status === 'loading') {
      return 'Modell wird geladen. Audioaufnahme ist gleich verfügbar.';
    }
    if (status === 'error') {
      return 'Modell konnte nicht geladen werden. Bitte Seite neu laden.';
    }
    if (status === 'idle') {
      return 'Modell ist noch nicht bereit.';
    }
    return null;
  }, [status]);

  const errorMessage = useMemo(() => {
    if (!error) return null;
    if (typeof error === 'string') return error;
    if (error?.message) return error.message;
    return 'Mikrofonzugriff fehlgeschlagen.';
  }, [error]);

  const activeStepIndex = Math.max(0, STEP_ORDER.indexOf(activeStep));

  const handleAddClass = useCallback(() => {
    addClass();
  }, [addClass]);

  const handleRemoveClass = useCallback(
    (classIndex, classId) => {
      if (classes.length <= 1) return;
      removeClass(classIndex, classId);
    },
    [classes.length, removeClass],
  );

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

  const statusBanner =
    modelStatusMessage || errorMessage ? (
      <div
        className={cx(
          styles['status-banner'],
          (status === 'error' || errorMessage) && styles.error,
        )}
        role="status"
        aria-live="polite"
      >
        {errorMessage ?? modelStatusMessage}
      </div>
    ) : null;

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
          <div className={styles['ic-title']} aria-label="Audioerkennung">
            Audioerkennung
          </div>
        </header>

        <nav
          className={styles['ic-steps']}
          aria-label="Audioerkennung Schritte"
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
              {statusBanner}

              {classes.map((cls, index) => (
                <AudioClassCard
                  key={cls.id}
                  classNameValue={cls.name}
                  exampleCount={counts[index] ?? 0}
                  isCollecting={collectingClassId === cls.id}
                  spectrogramRef={spectrogramRef}
                  recordingProgress={recordingProgress}
                  recordingSecondsLeft={recordingSecondsLeft}
                  recordingDurationSeconds={recordingDurationSeconds}
                  canCollect={canCollect}
                  onClassNameChange={(event) => updateClassName(index, event.target.value)}
                  onClassNameFocus={() => clearDefaultClassName(index)}
                  onClassNameBlur={() => normalizeClassName(index)}
                  onCollect={() => startCollecting(cls.id)}
                  onCollectStop={stopCollecting}
                  onClearExamples={() => clearClassExamples(cls.id)}
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
              {statusBanner}
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

              <div className={cx(styles.card, styles['loss-card'])}>
                <div className={styles['card-header']}>
                  <h3>Loss-Verlauf</h3>
                  {lossHistory.length ? (
                    <span className={styles['loss-value']}>
                      {lossHistory.at(-1).toFixed(4)}
                    </span>
                  ) : (
                    <span className={styles['loss-empty']}>Noch keine Daten</span>
                  )}
                </div>
                <div className={styles['loss-chart-shell']}>
                  <LossSparkline values={lossHistory} />
                </div>
              </div>
            </section>
          ) : null}

          {activeStep === 'test' ? (
            <section className={styles['preview-column']}>
              {statusBanner}
              <div className={cx(styles.card, styles['preview-card'])}>
                <div className={styles['preview-body']}>
                  <div className={styles['spectrogram-frame']}>
                    <SpectrogramCanvas spectrogramRef={spectrogramRef} isActive={isListening} />
                    {!isListening ? (
                      <div className={styles['spectrogram-overlay']}>
                        Live-Audio wird gestartet. Mikrofonzugriff erlauben.
                      </div>
                    ) : null}
                  </div>

                  <div className={styles['audio-output']}>
                    <div className={styles['preview-output-header']}>
                      <span>Ausgabe</span>
                      <BluetoothButton
                        label={isConnected ? 'Trennen' : 'Verbinden'}
                        onClick={handleBleClick}
                        isConnected={isConnected}
                        deviceName={device?.name}
                      />
                    </div>
                    <span className={styles['spectrogram-meta']}>
                      {isListening ? 'Live' : 'Wird gestartet…'}
                    </span>

                    <div className={styles.probabilityList}>
                      {classes.map((cls, index) => {
                        const probability = probabilities[index] ?? 0;
                        const percent = Math.round(probability * 100);
                        const color = getClassColor(index);

                        return (
                          <div key={cls.id} className={styles['probability-row']}>
                            <div className={styles['probability-row-header']}>
                              <span>{cls.name}</span>
                              <span>{percent}%</span>
                            </div>
                            <div
                              className={styles['probability-bar']}
                              role="meter"
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-valuenow={percent}
                            >
                              <div
                                className={styles['probability-bar-fill']}
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
