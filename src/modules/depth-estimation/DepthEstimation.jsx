import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import MaterialSlider from '../../components/common/MaterialSlider.jsx';
import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import ModelSwitcher from '../../components/common/ModelSwitcher.jsx';
import { useDepthEstimation } from '../../hooks/useDepthEstimation.js';
import { useWebcam } from '../../hooks/useWebcam.js';
import PreviewPanel from '../image-classification/components/PreviewPanel.jsx';
import styles from '../image-classification/ImageClassification.module.css';

import depthStyles from './DepthEstimation.module.css';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

const DETECTION_THROTTLE_MS = 35;

function applyHeatmapColorsToDepthMap(depthMask, width, height) {
  if (!depthMask) return null;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const imageData = ctx.createImageData(width, height);
  const pixels = imageData.data;

  const maskData = depthMask.getAsFloat32Array();
  const totalPixels = width * height;

  for (let i = 0; i < totalPixels; i++) {
    const depth = maskData[i];
    const pixelOffset = i * 4;

    if (depth > 0.5) {
      const t = (depth - 0.5) * 2;
      pixels[pixelOffset] = Math.round(255 * (1 - t) + 255 * t);
      pixels[pixelOffset + 1] = Math.round(220 * (1 - t) + 215 * t);
      pixels[pixelOffset + 2] = Math.round(0 * (1 - t) + 0 * t);
    } else {
      const t = depth * 2;
      pixels[pixelOffset] = Math.round(138 * (1 - t) + 255 * t);
      pixels[pixelOffset + 1] = Math.round(43 * (1 - t) + 220 * t);
      pixels[pixelOffset + 2] = Math.round(226 * (1 - t) + 0 * t);
    }

    pixels[pixelOffset + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function applyDepthThresholdToDepthMap(depthMask, width, height, threshold) {
  if (!depthMask) return null;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const imageData = ctx.createImageData(width, height);
  const pixels = imageData.data;

  const maskData = depthMask.getAsFloat32Array();
  const totalPixels = width * height;

  const normalizedThreshold = threshold / 100;

  for (let i = 0; i < totalPixels; i++) {
    const depth = maskData[i];
    const pixelOffset = i * 4;

    const distance = Math.abs(depth - normalizedThreshold);
    const inFocus = distance < 0.15;

    if (inFocus) {
      const t = (depth - 0.5) * 2;
      pixels[pixelOffset] = Math.round(255 * (1 - t) + 255 * t);
      pixels[pixelOffset + 1] = Math.round(220 * (1 - t) + 215 * t);
      pixels[pixelOffset + 2] = Math.round(0 * (1 - t) + 0 * t);
      pixels[pixelOffset + 3] = 255;
    } else {
      const t = depth * 2;
      pixels[pixelOffset] = Math.round(138 * (1 - t) + 255 * t);
      pixels[pixelOffset + 1] = Math.round(43 * (1 - t) + 220 * t);
      pixels[pixelOffset + 2] = Math.round(226 * (1 - t) + 0 * t);
      pixels[pixelOffset + 3] = 100;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

function applyBlurEffect(videoElement, depthMask, width, height, threshold) {
  if (!depthMask || !videoElement) return null;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.drawImage(videoElement, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);
  const pixels = imageData.data;

  const maskData = depthMask.getAsFloat32Array();
  const normalizedThreshold = threshold / 100;

  for (let i = 0; i < width * height; i++) {
    const depth = maskData[i];
    const distance = Math.abs(depth - normalizedThreshold);
    const inFocus = distance < 0.15;

    if (!inFocus) {
      const pixelOffset = i * 4;
      const r = pixels[pixelOffset];
      const g = pixels[pixelOffset + 1];
      const b = pixels[pixelOffset + 2];

      const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      const blurAmount = Math.min(1, distance * 3);

      pixels[pixelOffset] = Math.round(r * (1 - blurAmount) + gray * blurAmount);
      pixels[pixelOffset + 1] = Math.round(g * (1 - blurAmount) + gray * blurAmount);
      pixels[pixelOffset + 2] = Math.round(b * (1 - blurAmount) + gray * blurAmount);
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

export default function DepthEstimation() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [depthMask, setDepthMask] = useState(null);
  const [focusThreshold, setFocusThreshold] = useState(70);

  const videoRef = useRef(null);
  const depthCanvasRef = useRef(null);
  const effectCanvasRef = useRef(null);

  const { status: modelStatus, predict } = useDepthEstimation({ enabled: true });
  const {
    status: webcamStatus,
    stream,
    isMirrored,
    canSwitchCamera,
    toggleFacingMode,
  } = useWebcam({ enabled: true });

  useEffect(() => {
    if (modelStatus !== 'ready' || webcamStatus !== 'ready') {
      setDepthMask(null);
      return undefined;
    }

    let cancelled = false;
    let rafId = null;
    let lastTimestamp = 0;

    const loop = (timestamp) => {
      if (cancelled) return;

      const videoEl = videoRef.current;
      if (videoEl?.readyState >= 2 && timestamp - lastTimestamp >= DETECTION_THROTTLE_MS) {
        const mask = predict(videoEl, timestamp);
        setDepthMask(mask);
        lastTimestamp = timestamp;
      }

      rafId = window.requestAnimationFrame(loop);
    };

    rafId = window.requestAnimationFrame(loop);

    return () => {
      cancelled = true;
      if (rafId) window.cancelAnimationFrame(rafId);
    };
  }, [predict, modelStatus, webcamStatus]);

  const renderDepthVisualization = useCallback(() => {
    const canvas = depthCanvasRef.current;
    const videoEl = videoRef.current;

    if (!canvas || !videoEl || !depthMask) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;

    if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
      canvas.width = videoWidth;
      canvas.height = videoHeight;
    }

    const heatmapCanvas = applyDepthThresholdToDepthMap(
      depthMask,
      depthMask.width,
      depthMask.height,
      focusThreshold,
    );

    if (heatmapCanvas) {
      ctx.clearRect(0, 0, videoWidth, videoHeight);
      ctx.drawImage(heatmapCanvas, 0, 0, videoWidth, videoHeight);
    }
  }, [depthMask, focusThreshold]);

  const renderEffectVisualization = useCallback(() => {
    const canvas = effectCanvasRef.current;
    const videoEl = videoRef.current;

    if (!canvas || !videoEl || !depthMask) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;

    if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
      canvas.width = videoWidth;
      canvas.height = videoHeight;
    }

    const effectCanvas = applyBlurEffect(
      videoEl,
      depthMask,
      depthMask.width,
      depthMask.height,
      focusThreshold,
    );

    if (effectCanvas) {
      ctx.clearRect(0, 0, videoWidth, videoHeight);
      ctx.drawImage(effectCanvas, 0, 0, videoWidth, videoHeight);
    }
  }, [depthMask, focusThreshold]);

  useEffect(() => {
    if (!depthMask) return;
    renderDepthVisualization();
    renderEffectVisualization();
  }, [depthMask, focusThreshold, renderDepthVisualization, renderEffectVisualization]);

  const showCameraSwitch = webcamStatus === 'ready' && canSwitchCamera;

  const statusMessage = useMemo(() => {
    if (modelStatus === 'loading') {
      return 'Modell wird geladen. Tiefenschätzung ist gleich verfügbar.';
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

  const handleSliderChange = useCallback((value) => {
    setFocusThreshold(value);
  }, []);

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
          aria-label="Tiefenschätzung Schritte"
          style={{
            '--active-step': 0,
            '--step-count': 1,
          }}
        >
          <span className={styles['ic-step-indicator']} aria-hidden="true" />
          <button className={cx(styles['ic-step'], styles.active)} type="button" disabled>
            <span className={styles['ic-step-number']}>1</span>
            Live-Analyse
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

          <div className={depthStyles['depth-grid']}>
            <section className={depthStyles['depth-section']}>
              <div className={styles.card}>
                <div className={cx(styles['card-header'], styles.spaced)}>
                  <h3>A. Eingabe (Webcam)</h3>
                </div>
                <div className={depthStyles['depth-view']}>
                  <PreviewPanel
                    stream={stream}
                    classes={[]}
                    probabilities={[]}
                    showCameraSwitch={showCameraSwitch}
                    isMirrored={isMirrored}
                    onToggleCamera={toggleFacingMode}
                    captureRef={videoRef}
                    overlayRenderer={null}
                  />
                </div>
              </div>
            </section>

            <section className={depthStyles['depth-section']}>
              <div className={styles.card}>
                <div className={cx(styles['card-header'], styles.spaced)}>
                  <h3>B. Das "Gehirn" (Tiefenkarte)</h3>
                </div>
                <div className={depthStyles['depth-view']}>
                  <div className={depthStyles['canvas-container']}>
                    <canvas
                      ref={depthCanvasRef}
                      className={depthStyles['depth-canvas']}
                      style={{ transform: isMirrored ? 'scaleX(-1)' : 'none' }}
                    />
                  </div>
                  <div className={depthStyles['depth-legend']}>
                    <div className={depthStyles['legend-item']}>
                      <div
                        className={depthStyles['legend-color']}
                        style={{ background: 'linear-gradient(90deg, #8A2BE2, #FF6347, #FFD700)' }}
                      />
                      <span className={depthStyles['legend-label']}>
                        Violett = Fern • Rot = Mittel • Gelb = Nah
                      </span>
                    </div>
                    <p className={depthStyles['legend-hint']}>
                      Aufgehellte Bereiche sind im aktuellen Fokus-Bereich
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <section className={depthStyles['depth-section']}>
              <div className={styles.card}>
                <div className={cx(styles['card-header'], styles.spaced)}>
                  <h3>C. Ausgabe (Effekt)</h3>
                </div>
                <div className={depthStyles['depth-view']}>
                  <div className={depthStyles['canvas-container']}>
                    <canvas
                      ref={effectCanvasRef}
                      className={depthStyles['depth-canvas']}
                      style={{ transform: isMirrored ? 'scaleX(-1)' : 'none' }}
                    />
                  </div>
                  <p className={depthStyles['effect-hint']}>
                    Bereiche außerhalb des Fokus werden entsättigt
                  </p>
                </div>
              </div>
            </section>
          </div>

          <section className={depthStyles['control-section']}>
            <div className={styles.card}>
              <div className={cx(styles['card-header'], styles.spaced)}>
                <h3>Steuerung</h3>
              </div>
              <div className={depthStyles['control-body']}>
                <div className={depthStyles['slider-group']}>
                  <label htmlFor="focus-slider" className={depthStyles['slider-label']}>
                    Fokus-Distanz: {focusThreshold}%
                  </label>
                  <MaterialSlider
                    id="focus-slider"
                    value={focusThreshold}
                    min={0}
                    max={100}
                    step={1}
                    onChange={handleSliderChange}
                    aria-label="Fokus-Distanz einstellen"
                  />
                  <p className={depthStyles['slider-hint']}>
                    Bewege den Slider, um den Fokus-Bereich zu ändern und sieh, wie die
                    Tiefenkarte (B) und der Effekt (C) sich live anpassen.
                  </p>
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
