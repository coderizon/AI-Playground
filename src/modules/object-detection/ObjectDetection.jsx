import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import ModelSwitcher from '../../components/common/ModelSwitcher.jsx';
import { useBluetooth } from '../../hooks/useBluetooth.js';
import { useObjectDetector } from '../../hooks/useObjectDetector.js';
import { useWebcam } from '../../hooks/useWebcam.js';
import BluetoothModal from '../image-classification/components/BluetoothModal.jsx';
import PreviewPanel from '../image-classification/components/PreviewPanel.jsx';
import styles from '../image-classification/ImageClassification.module.css';

import { drawObjectDetections } from './drawObjectDetections.js';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

const DETECTION_THROTTLE_MS = 35;

function getTopCategory(detection) {
  const categories = detection?.categories;
  if (!Array.isArray(categories) || categories.length === 0) return null;

  return categories.reduce((best, current) => {
    const bestScore = typeof best?.score === 'number' ? best.score : 0;
    const currentScore = typeof current?.score === 'number' ? current.score : 0;
    return currentScore > bestScore ? current : best;
  }, categories[0]);
}

function buildDetectionSummary(detections) {
  if (!Array.isArray(detections) || detections.length === 0) {
    return { classes: [], probabilities: [], bestLabel: null };
  }

  const byLabel = new Map();
  let bestLabel = null;
  let bestScore = -1;

  detections.forEach((detection, index) => {
    const category = getTopCategory(detection);
    const label =
      category?.categoryName ??
      category?.displayName ??
      `Objekt ${index + 1}`;
    const score = typeof category?.score === 'number' ? category.score : 0;

    if (score > bestScore) {
      bestScore = score;
      bestLabel = label;
    }

    if (!label) return;
    const existing = byLabel.get(label);
    if (!existing || score > existing.score) {
      byLabel.set(label, { id: label, name: label, score });
    }
  });

  const items = Array.from(byLabel.values()).sort((a, b) => b.score - a.score);

  return {
    classes: items.map((item) => ({ id: item.id, name: item.name })),
    probabilities: items.map((item) => item.score),
    bestLabel,
  };
}

export default function ObjectDetection() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [detections, setDetections] = useState([]);

  const captureVideoRef = useRef(null);
  const lastSentLabelRef = useRef(null);

  const { connect, disconnect, send, isConnected, device } = useBluetooth();
  const { status: modelStatus, detect } = useObjectDetector({ enabled: true });
  const {
    status: webcamStatus,
    stream,
    isMirrored,
    canSwitchCamera,
    toggleFacingMode,
  } = useWebcam({ enabled: true });

  useEffect(() => {
    if (modelStatus !== 'ready' || webcamStatus !== 'ready') {
      setDetections([]);
      return undefined;
    }

    let cancelled = false;
    let rafId = null;
    let lastTimestamp = 0;

    const loop = (timestamp) => {
      if (cancelled) return;

      const videoEl = captureVideoRef.current;
      if (videoEl?.readyState >= 2 && timestamp - lastTimestamp >= DETECTION_THROTTLE_MS) {
        const nextDetections = detect(videoEl, timestamp);
        setDetections(nextDetections ?? []);
        lastTimestamp = timestamp;
      }

      rafId = window.requestAnimationFrame(loop);
    };

    rafId = window.requestAnimationFrame(loop);

    return () => {
      cancelled = true;
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [detect, modelStatus, webcamStatus]);

  const detectionSummary = useMemo(
    () => buildDetectionSummary(detections),
    [detections],
  );

  const overlayRenderer = useCallback(
    ({ ctx, canvas, video, width, height }) => {
      drawObjectDetections({
        ctx,
        canvas,
        video,
        width,
        height,
        detections,
      });
    },
    [detections],
  );

  useEffect(() => {
    if (!isConnected) {
      lastSentLabelRef.current = null;
      return;
    }

    const bestLabel = detectionSummary.bestLabel?.trim();
    if (!bestLabel) {
      lastSentLabelRef.current = null;
      return;
    }
    if (lastSentLabelRef.current === bestLabel) return;

    lastSentLabelRef.current = bestLabel;
    send(bestLabel);
  }, [detectionSummary.bestLabel, isConnected, send]);

  const showCameraSwitch = webcamStatus === 'ready' && canSwitchCamera;

  const statusMessage = useMemo(() => {
    if (modelStatus === 'loading') {
      return 'Modell wird geladen. Objekterkennung ist gleich verfügbar.';
    }
    if (modelStatus === 'error') {
      return 'Modell konnte nicht geladen werden. Bitte Seite neu laden.';
    }
    if (webcamStatus === 'loading') {
      return 'Kamera wird gestartet.';
    }
    if (webcamStatus === 'error') {
      return 'Kamerazugriff fehlgeschlagen.';
    }
    if (modelStatus === 'idle') {
      return 'Modell ist noch nicht bereit.';
    }
    return null;
  }, [modelStatus, webcamStatus]);

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
          <ModelSwitcher />
        </header>

        <nav
          className={styles['ic-steps']}
          aria-label="Objekterkennung Schritte"
          style={{
            '--active-step': 0,
            '--step-count': 1,
          }}
        >
          <span className={styles['ic-step-indicator']} aria-hidden="true" />
          <button className={cx(styles['ic-step'], styles.active)} type="button" disabled>
            <span className={styles['ic-step-number']}>1</span>
            Testen
          </button>
        </nav>

        <main className={styles['ic-stage']} data-step="test">
          {statusMessage ? (
            <div
              className={cx(
                styles['status-banner'],
                (modelStatus === 'error' || webcamStatus === 'error') && styles.error,
              )}
              role="status"
              aria-live="polite"
            >
              {statusMessage}
            </div>
          ) : null}

          <section className={styles['preview-column']}>
            <PreviewPanel
              stream={stream}
              classes={detectionSummary.classes}
              probabilities={detectionSummary.probabilities}
              showCameraSwitch={showCameraSwitch}
              isMirrored={isMirrored}
              onToggleCamera={toggleFacingMode}
              captureRef={captureVideoRef}
              overlayRenderer={overlayRenderer}
              onConnect={handleBleClick}
              isConnected={isConnected}
              deviceName={device?.name}
            />
          </section>
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
