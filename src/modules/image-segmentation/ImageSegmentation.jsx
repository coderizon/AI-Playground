import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import * as deeplab from '@tensorflow-models/deeplab';
import { initTensorFlowBackend } from '../../utils/tensorflow-init.js';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import ModelSwitcher from '../../components/common/ModelSwitcher.jsx';
import { useWebcam } from '../../hooks/useWebcam.js';
import WebcamCapture from '../image-classification/components/WebcamCapture.jsx';
import styles from '../image-classification/ImageClassification.module.css';

import segStyles from './ImageSegmentation.module.css';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

const MODEL_CONFIG = { base: 'pascal', quantizationBytes: 2 };
const OVERLAY_ALPHA = 0.85;
const LABELS = deeplab.getLabels(MODEL_CONFIG.base);
const COLOR_MAP = deeplab.getColormap(MODEL_CONFIG.base);

function buildLegendItems(counts, total, labels, colormap) {
  const items = [];

  counts.forEach((count, rawId) => {
    const id = Math.round(Number(rawId));
    if (!Number.isFinite(id) || id === 0) return;

    const name = labels[id] ?? `Klasse ${id}`;
    const baseColor = colormap[id] ?? [0, 0, 0];
    const color = [baseColor[0], baseColor[1], baseColor[2], OVERLAY_ALPHA];
    const percent = total ? Math.round((count / total) * 100) : null;

    items.push({ id, name, color, percent, count });
  });

  return items.sort((a, b) => b.count - a.count);
}

function createMaskCanvas(map, colormap) {
  const width = map?.width ?? 0;
  const height = map?.height ?? 0;
  if (!width || !height) return null;

  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = width;
  maskCanvas.height = height;

  const ctx = maskCanvas.getContext('2d');
  if (!ctx) return null;

  const imageData = ctx.createImageData(width, height);
  const output = imageData.data;
  const data = map.data ?? [];
  const total = Math.min(data.length, width * height);

  for (let index = 0; index < total; index += 1) {
    const rawValue = data[index];
    const id = Number.isFinite(rawValue) ? Math.max(0, Math.round(rawValue)) : 0;
    const baseColor = colormap[id] ?? [0, 0, 0];
    const alpha = id === 0 ? 0 : Math.round(OVERLAY_ALPHA * 255);
    const offset = index * 4;
    output[offset] = baseColor[0];
    output[offset + 1] = baseColor[1];
    output[offset + 2] = baseColor[2];
    output[offset + 3] = alpha;
  }

  ctx.putImageData(imageData, 0, 0);
  return maskCanvas;
}

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Bild konnte nicht geladen werden.'));
    img.src = src;
  });
}

async function resolveImageElement(imageElement, imageURL) {
  if (imageElement?.complete && imageElement.naturalWidth) return imageElement;
  if (!imageURL) return null;

  if (imageElement?.src === imageURL && typeof imageElement.decode === 'function') {
    try {
      await imageElement.decode();
      if (imageElement.naturalWidth) return imageElement;
    } catch {
      // Fall back to creating a new Image element.
    }
  }

  return loadImageElement(imageURL);
}

export default function ImageSegmentation() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [imageURL, setImageURL] = useState(null);
  const [legendItems, setLegendItems] = useState([]);
  const [segmentationMeta, setSegmentationMeta] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [modelStatus, setModelStatus] = useState('idle');
  const [isSegmenting, setIsSegmenting] = useState(false);

  const videoRef = useRef(null);
  const imageRef = useRef(null);
  const overlayRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const modelRef = useRef(null);
  const modelPromiseRef = useRef(null);

  const {
    status: webcamStatus,
    stream,
    isMirrored,
    canSwitchCamera,
    toggleFacingMode,
  } = useWebcam({ enabled: !imageURL });

  const isReview = Boolean(imageURL);
  const showCameraSwitch = !isReview && webcamStatus === 'ready' && canSwitchCamera;
  const activeStepIndex = isReview ? 1 : 0;

  const clearOverlay = useCallback(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const resetSegmentation = useCallback(() => {
    maskCanvasRef.current = null;
    setSegmentationMeta(null);
    setLegendItems([]);
    setAnalysisError(null);
    clearOverlay();
  }, [clearOverlay]);

  const loadModel = useCallback(async () => {
    if (modelRef.current) return modelRef.current;

    if (!modelPromiseRef.current) {
      setModelStatus('loading');

      modelPromiseRef.current = (async () => {
        await initTensorFlowBackend();

        const model = await deeplab.load(MODEL_CONFIG);
        modelRef.current = model;
        setModelStatus('ready');
        return model;
      })().catch((error) => {
        modelPromiseRef.current = null;
        setModelStatus('error');
        throw error;
      });
    }

    return modelPromiseRef.current;
  }, []);

  const capture = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.readyState < 2) return;

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (isMirrored) {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }

    ctx.drawImage(video, 0, 0, width, height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

    resetSegmentation();
    setImageURL(dataUrl);
  }, [isMirrored, resetSegmentation]);

  const handleRetake = useCallback(() => {
    if (isSegmenting) return;
    resetSegmentation();
    setImageURL(null);
  }, [isSegmenting, resetSegmentation]);

  const handleSegment = useCallback(async () => {
    if (!imageURL || isSegmenting) return;

    resetSegmentation();
    setIsSegmenting(true);

    let rawSegmentationMap = null;

    try {
      const model = await loadModel();
      const imageElement = await resolveImageElement(imageRef.current, imageURL);
      if (!imageElement) {
        throw new Error('Kein Bild verfügbar.');
      }

      rawSegmentationMap = model.predict(imageElement);
      const [height, width] = rawSegmentationMap?.shape ?? [];
      const data = await rawSegmentationMap.data();

      if (!data || !width || !height) {
        throw new Error('Segmentierungsdaten fehlen.');
      }

      const map = { data, width, height };
      const counts = new Map();
      const totalPixels = Math.min(data.length, width * height);
      for (let index = 0; index < totalPixels; index += 1) {
        const value = data[index];
        if (!Number.isFinite(value)) continue;
        const id = Math.max(0, Math.round(value));
        counts.set(id, (counts.get(id) ?? 0) + 1);
      }

      maskCanvasRef.current = createMaskCanvas(map, COLOR_MAP);
      setLegendItems(buildLegendItems(counts, totalPixels, LABELS, COLOR_MAP));
      setSegmentationMeta({
        width,
        height,
        labels: LABELS.length,
      });
    } catch (error) {
      console.error(error);
      const fallbackMessage = 'Segmentierung fehlgeschlagen. Bitte erneut versuchen.';
      setAnalysisError(fallbackMessage);
    } finally {
      if (rawSegmentationMap?.dispose) rawSegmentationMap.dispose();
      setIsSegmenting(false);
    }
  }, [imageURL, isSegmenting, loadModel, resetSegmentation]);

  const drawOverlay = useCallback(() => {
    const canvas = overlayRef.current;
    const maskCanvas = maskCanvasRef.current;
    const imageElement = imageRef.current;

    if (!canvas || !maskCanvas || !imageElement) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;

    const devicePixelRatio =
      typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;
    const nextWidth = Math.round(bounds.width * devicePixelRatio);
    const nextHeight = Math.round(bounds.height * devicePixelRatio);

    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }

    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, bounds.width, bounds.height);

    const sourceWidth =
      imageElement.naturalWidth || imageElement.width || maskCanvas.width;
    const sourceHeight =
      imageElement.naturalHeight || imageElement.height || maskCanvas.height;

    if (!sourceWidth || !sourceHeight) return;

    const scale = Math.max(bounds.width / sourceWidth, bounds.height / sourceHeight);
    const drawWidth = sourceWidth * scale;
    const drawHeight = sourceHeight * scale;
    const offsetX = (bounds.width - drawWidth) / 2;
    const offsetY = (bounds.height - drawHeight) / 2;

    ctx.drawImage(maskCanvas, offsetX, offsetY, drawWidth, drawHeight);
  }, []);

  useEffect(() => {
    if (!segmentationMeta) {
      clearOverlay();
      return;
    }

    drawOverlay();
  }, [clearOverlay, drawOverlay, segmentationMeta]);

  useEffect(() => {
    if (!segmentationMeta) return;

    const handleResize = () => drawOverlay();
    window.addEventListener('resize', handleResize);

    return () => window.removeEventListener('resize', handleResize);
  }, [drawOverlay, segmentationMeta]);

  const statusMessage = useMemo(() => {
    if (analysisError) return analysisError;
    if (!isReview) {
      if (webcamStatus === 'loading') return 'Kamera wird gestartet.';
      if (webcamStatus === 'error') return 'Kamerazugriff fehlgeschlagen.';
      return null;
    }
    if (modelStatus === 'loading') return 'Segmentierungsmodell wird geladen.';
    if (modelStatus === 'error') {
      return 'Segmentierungsmodell konnte nicht geladen werden. Bitte Seite neu laden.';
    }
    return null;
  }, [analysisError, isReview, modelStatus, webcamStatus]);

  const statusIsError = useMemo(() => {
    if (analysisError) return true;
    if (!isReview) return webcamStatus === 'error';
    return modelStatus === 'error';
  }, [analysisError, isReview, modelStatus, webcamStatus]);

  const segmentationStatus = useMemo(() => {
    if (!isReview) return null;
    if (isSegmenting) return 'Segmentierung laeuft...';
    if (segmentationMeta) {
      if (!legendItems.length) return 'Keine Klassen erkannt.';
      return `${legendItems.length} Klassen erkannt.`;
    }
    return 'Bereit für Segmentierung.';
  }, [isReview, isSegmenting, legendItems.length, segmentationMeta]);

  const outputMeta = useMemo(() => {
    if (!isReview) return null;
    if (isSegmenting) return 'Segmentiere...';
    if (segmentationMeta) return 'Fertig';
    return 'Bereit';
  }, [isReview, isSegmenting, segmentationMeta]);

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
          aria-label="Bildsegmentierung Schritte"
          style={{
            '--active-step': activeStepIndex,
            '--step-count': 2,
          }}
        >
          <span className={styles['ic-step-indicator']} aria-hidden="true" />
          <button
            className={cx(styles['ic-step'], !isReview && styles.active)}
            type="button"
            onClick={handleRetake}
            disabled={isSegmenting}
          >
            <span className={styles['ic-step-number']}>1</span>
            Kamera
          </button>
          <button
            className={cx(styles['ic-step'], isReview && styles.active)}
            type="button"
            disabled={!isReview}
          >
            <span className={styles['ic-step-number']}>2</span>
            Review
          </button>
        </nav>

        <main className={styles['ic-stage']} data-step={isReview ? 'review' : 'live'}>
          {statusMessage ? (
            <div
              className={cx(styles['status-banner'], statusIsError && styles.error)}
              role="status"
              aria-live="polite"
            >
              {statusMessage}
            </div>
          ) : null}

          <div className={segStyles['seg-grid']}>
            <section className={styles['preview-column']}>
              <div className={styles.card}>
                <div className={cx(styles['card-header'], styles.spaced)}>
                  <h3>{isReview ? 'Aufnahme' : 'Live-View'}</h3>
                </div>

                <div className={segStyles['capture-body']}>
                  {isReview ? (
                    <div className={styles['capture-slot']}>
                      <img
                        ref={imageRef}
                        className={segStyles['capture-image']}
                        src={imageURL}
                        alt="Aufnahme"
                        onLoad={() => segmentationMeta && drawOverlay()}
                      />
                      {segmentationMeta ? (
                        <canvas
                          ref={overlayRef}
                          className={segStyles['segmentation-overlay']}
                          aria-hidden="true"
                        />
                      ) : null}
                    </div>
                  ) : (
                    <WebcamCapture
                      ref={videoRef}
                      stream={stream}
                      isMirrored={isMirrored}
                      showCameraSwitch={showCameraSwitch}
                      onToggleCamera={toggleFacingMode}
                      variant="preview"
                    />
                  )}

                  {!isReview ? (
                    <div className={segStyles['capture-actions']}>
                      <button
                        className={cx(
                          styles.dataCollector,
                          styles.primary,
                          segStyles['capture-button'],
                        )}
                        type="button"
                        onClick={capture}
                        disabled={webcamStatus !== 'ready'}
                      >
                        Foto aufnehmen
                      </button>
                    </div>
                  ) : null}
                </div>
              </div>
            </section>

            <section className={styles['training-column']}>
              <div className={styles.card}>
                <div className={cx(styles['card-header'], styles.spaced)}>
                  <h3>Steuerung</h3>
                  {isReview ? (
                    <button
                      className={segStyles['secondary-button']}
                      type="button"
                      onClick={handleRetake}
                      disabled={isSegmenting}
                    >
                      Neu aufnehmen
                    </button>
                  ) : null}
                </div>

                {isReview ? (
                  <div className={segStyles['control-body']}>
                    <p className={segStyles.placeholder}>Automatische Bildsegmentierung.</p>
                    <button
                      className={cx(styles.primary, styles.block)}
                      type="button"
                      onClick={handleSegment}
                      disabled={isSegmenting}
                    >
                      {isSegmenting ? 'Segmentiere...' : 'Bild segmentieren'}
                    </button>
                    {segmentationStatus ? (
                      <span className={segStyles['status-hint']}>{segmentationStatus}</span>
                    ) : null}
                  </div>
                ) : (
                  <p className={segStyles.placeholder}>
                    Nimm ein Foto auf, um eine Segmentierung zu erhalten.
                  </p>
                )}
              </div>

              <div className={styles.card}>
                <div className={cx(styles['card-header'], styles.spaced)}>
                  <h3>Segmentierung</h3>
                  {outputMeta ? (
                    <span className={segStyles['output-meta']}>{outputMeta}</span>
                  ) : null}
                </div>

                <div className={segStyles['segmentation-output']} role="log" aria-live="polite">
                  {legendItems.length ? (
                    <ul className={segStyles['legend-list']} role="list">
                      {legendItems.map((item) => (
                        <li key={item.id} className={segStyles['legend-item']}>
                          <span
                            className={segStyles['legend-swatch']}
                            style={{
                              backgroundColor: `rgba(${item.color[0]}, ${item.color[1]}, ${item.color[2]}, ${item.color[3]})`,
                            }}
                          />
                          <span className={segStyles['legend-name']}>{item.name}</span>
                          {typeof item.percent === 'number' ? (
                            <span className={segStyles['legend-percent']}>
                              {item.percent}%
                            </span>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <span className={styles['loss-empty']}>
                      {isSegmenting ? 'Segmentierung laeuft...' : 'Noch keine Segmentierung.'}
                    </span>
                  )}
                </div>
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
