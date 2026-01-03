import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import ModelSwitcher from '../../components/common/ModelSwitcher.jsx';
import { useBluetooth } from '../../hooks/useBluetooth.js';
import { useFaceLandmarker } from '../../hooks/useFaceLandmarker.js';
import { useWebcam } from '../../hooks/useWebcam.js';
import BluetoothModal from '../image-classification/components/BluetoothModal.jsx';
import PreviewPanel from '../image-classification/components/PreviewPanel.jsx';
import styles from '../image-classification/ImageClassification.module.css';

import { drawHeadTracking } from './drawHeadTracking.js';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

// Iris landmark indices (left eye)
const LEFT_IRIS_LEFT = 469;
const LEFT_IRIS_RIGHT = 471;

// Nose tip landmark index
const NOSE_TIP = 1;

// Initial focal length for distance calculation (calibrated value)
const INITIAL_FOCAL_LENGTH = 550;

// Average iris diameter in cm
const IRIS_DIAMETER_CM = 1.17;

// Bluetooth send throttle in ms
const BLUETOOTH_SEND_INTERVAL = 150;

// Detection throttle in ms
const DETECTION_THROTTLE_MS = 35;

export default function HeadTracking() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeFace, setActiveFace] = useState(null);
  const [distanceCM, setDistanceCM] = useState(null);
  const [positionPercent, setPositionPercent] = useState(null);

  const captureVideoRef = useRef(null);
  const lastBluetoothSendRef = useRef(0);
  const faceQueueRef = useRef(Promise.resolve());
  const pendingDetectionsRef = useRef(0);
  const lastFaceRef = useRef({ face: null, features: null });

  const { connect, disconnect, send, isConnected, device } = useBluetooth();
  const { status: modelStatus, getFaceFeatures } = useFaceLandmarker({ enabled: true });
  const {
    status: webcamStatus,
    stream,
    isMirrored,
    canSwitchCamera,
    toggleFacingMode,
  } = useWebcam({ enabled: true });

  // Enqueue face detection to prevent overlapping calls
  const enqueueFaceDetection = useCallback(
    (input) => {
      if (!input) return Promise.resolve({ face: null, features: null });

      // Skip if already processing
      if (pendingDetectionsRef.current > 0) {
        return Promise.resolve(lastFaceRef.current ?? { face: null, features: null });
      }

      pendingDetectionsRef.current += 1;

      const task = async () => {
        try {
          const result = await getFaceFeatures(input);
          lastFaceRef.current = result ?? { face: null, features: null };
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

  // Calculate distance from iris width
  const calculateDistance = useCallback((landmarks) => {
    if (!landmarks || landmarks.length < 472) return null;

    const irisLeft = landmarks[LEFT_IRIS_LEFT];
    const irisRight = landmarks[LEFT_IRIS_RIGHT];

    if (!irisLeft || !irisRight) return null;

    // Calculate iris width in pixels (normalized 0-1, so we need video dimensions)
    const irisWidthNormalized = Math.abs(irisRight.x - irisLeft.x);

    // Assuming video width of 640px as reference (will be scaled appropriately)
    const referenceWidth = 640;
    const irisWidthPixels = irisWidthNormalized * referenceWidth;

    if (irisWidthPixels <= 0) return null;

    // Distance formula: D = (F * realWidth) / pixelWidth
    const distance = (INITIAL_FOCAL_LENGTH * IRIS_DIAMETER_CM) / irisWidthPixels;

    return distance;
  }, []);

  // Calculate normalized head position (0-100%)
  const calculatePosition = useCallback((landmarks) => {
    if (!landmarks || landmarks.length <= NOSE_TIP) return null;

    const noseTip = landmarks[NOSE_TIP];
    if (!noseTip) return null;

    // Normalize x and y to 0-100%
    const xPercent = Math.round(noseTip.x * 100);
    const yPercent = Math.round(noseTip.y * 100);

    return { x: xPercent, y: yPercent };
  }, []);

  // Send data via Bluetooth
  const sendBluetoothData = useCallback(
    (distance, position) => {
      if (!isConnected) return;

      const now = performance.now();
      if (now - lastBluetoothSendRef.current < BLUETOOTH_SEND_INTERVAL) return;

      lastBluetoothSendRef.current = now;

      const distanceStr = distance != null ? Math.round(distance) : 0;
      const xStr = position?.x ?? 50;
      const yStr = position?.y ?? 50;

      const message = `D:${distanceStr};X:${xStr};Y:${yStr}`;
      send(message);
    },
    [isConnected, send],
  );

  // Main tracking loop
  useEffect(() => {
    if (modelStatus !== 'ready' || webcamStatus !== 'ready') {
      setActiveFace(null);
      setDistanceCM(null);
      setPositionPercent(null);
      return undefined;
    }

    let cancelled = false;
    let rafId = null;
    let lastTimestamp = 0;

    const loop = async (timestamp) => {
      if (cancelled) return;

      const videoEl = captureVideoRef.current;
      if (videoEl?.readyState >= 2 && timestamp - lastTimestamp >= DETECTION_THROTTLE_MS) {
        const { face } = await enqueueFaceDetection(videoEl);

        if (face?.landmarks) {
          setActiveFace(face);

          // Calculate distance and position
          const distance = calculateDistance(face.landmarks);
          const position = calculatePosition(face.landmarks);

          setDistanceCM(distance);
          setPositionPercent(position);

          // Send via Bluetooth if connected
          sendBluetoothData(distance, position);
        } else {
          setActiveFace(null);
          setDistanceCM(null);
          setPositionPercent(null);
        }

        lastTimestamp = timestamp;
      }

      rafId = window.requestAnimationFrame(loop);
    };

    rafId = window.requestAnimationFrame(loop);

    return () => {
      cancelled = true;
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [
    modelStatus,
    webcamStatus,
    enqueueFaceDetection,
    calculateDistance,
    calculatePosition,
    sendBluetoothData,
  ]);

  // Build tracking summary for display (similar to detection summary)
  const trackingSummary = useMemo(() => {
    const classes = [];
    const probabilities = [];

    if (distanceCM != null && Number.isFinite(distanceCM)) {
      classes.push({ id: 'distance', name: `Distanz: ${Math.round(distanceCM)} cm` });
      probabilities.push(1);
    }

    if (positionPercent?.x != null && positionPercent?.y != null) {
      classes.push({
        id: 'position',
        name: `Kopfmitte: X=${positionPercent.x}% Y=${positionPercent.y}%`,
      });
      probabilities.push(1);
    }

    return { classes, probabilities };
  }, [distanceCM, positionPercent]);

  const overlayRenderer = useCallback(
    ({ ctx, canvas, video, width, height }) => {
      if (!activeFace?.landmarks) return;
      drawHeadTracking({
        ctx,
        canvas,
        video,
        width,
        height,
        landmarks: activeFace.landmarks,
        distanceCM,
      });
    },
    [activeFace, distanceCM],
  );

  const showCameraSwitch = webcamStatus === 'ready' && canSwitchCamera;

  const statusMessage = useMemo(() => {
    if (modelStatus === 'loading') {
      return 'Modell wird geladen. Kopftracking ist gleich verfügbar.';
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
          aria-label="Kopftracking Schritte"
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
              classes={trackingSummary.classes}
              probabilities={trackingSummary.probabilities}
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
