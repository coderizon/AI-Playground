import { useCallback, useMemo, useRef, useState } from 'react';

import { env, pipeline } from '@xenova/transformers';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import ModelSwitcher from '../../components/common/ModelSwitcher.jsx';
import { useWebcam } from '../../hooks/useWebcam.js';
import WebcamCapture from '../image-classification/components/WebcamCapture.jsx';
import styles from '../image-classification/ImageClassification.module.css';

import qaStyles from './VisualQA.module.css';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

const DEFAULT_PROMPT = 'Beschreibe dieses Bild';
const MODEL_ID = 'Xenova/vit-gpt2-image-captioning';
const MODEL_OPTIONS = { quantized: true };
const ASSISTANT_LABEL = 'Modell';

// Avoid Vite SPA fallback returning HTML for missing local model files.
env.allowLocalModels = false;
env.allowRemoteModels = true;

export default function VisualQA() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [imageURL, setImageURL] = useState(null);
  const [messages, setMessages] = useState([]);
  const [analysisError, setAnalysisError] = useState(null);
  const [modelStatus, setModelStatus] = useState('idle');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const videoRef = useRef(null);
  const pipelineRef = useRef(null);
  const pipelinePromiseRef = useRef(null);

  const { status: webcamStatus, stream, isMirrored, canSwitchCamera, toggleFacingMode } =
    useWebcam({ enabled: !imageURL });

  const isReview = Boolean(imageURL);
  const showCameraSwitch = !isReview && webcamStatus === 'ready' && canSwitchCamera;
  const activeStepIndex = isReview ? 1 : 0;

  const loadPipeline = useCallback(async () => {
    if (pipelineRef.current) return pipelineRef.current;
    if (!pipelinePromiseRef.current) {
      setModelStatus('loading');
      pipelinePromiseRef.current = pipeline(
        'image-to-text',
        MODEL_ID,
        MODEL_OPTIONS,
      )
        .then((loaded) => {
          pipelineRef.current = loaded;
          setModelStatus('ready');
          return loaded;
        })
        .catch((error) => {
          pipelinePromiseRef.current = null;
          setModelStatus('error');
          throw error;
        });
    }
    return pipelinePromiseRef.current;
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

    setImageURL(dataUrl);
    setMessages([]);
    setAnalysisError(null);
  }, [isMirrored]);

  const handleRetake = useCallback(() => {
    if (isAnalyzing) return;
    setImageURL(null);
    setMessages([]);
    setAnalysisError(null);
  }, [isAnalyzing]);

  const handleAnalyze = useCallback(async () => {
    if (!imageURL || isAnalyzing) return;

    const nextPrompt = DEFAULT_PROMPT;
    setAnalysisError(null);
    setIsAnalyzing(true);
    setMessages((prev) => [...prev, { role: 'user', content: nextPrompt }]);

    try {
      const model = await loadPipeline();
      const result = await model(imageURL);
      const generatedText =
        (Array.isArray(result) ? result[0]?.generated_text : result?.generated_text) ?? '';
      const answer =
        typeof generatedText === 'string' && generatedText.trim().length
          ? generatedText.trim()
          : 'Keine Antwort erhalten.';
      setMessages((prev) => [...prev, { role: 'assistant', content: answer }]);
    } catch (error) {
      console.error(error);
      const fallbackMessage = 'Analyse fehlgeschlagen. Bitte erneut versuchen.';
      setAnalysisError(fallbackMessage);
      setMessages((prev) => [...prev, { role: 'assistant', content: fallbackMessage }]);
    } finally {
      setIsAnalyzing(false);
    }
  }, [imageURL, isAnalyzing, loadPipeline]);

  const statusMessage = useMemo(() => {
    if (analysisError) return analysisError;
    if (!isReview) {
      if (webcamStatus === 'loading') return 'Kamera wird gestartet.';
      if (webcamStatus === 'error') return 'Kamerazugriff fehlgeschlagen.';
      return null;
    }
    if (modelStatus === 'loading') return 'Bildbeschreibung-Modell wird geladen.';
    if (modelStatus === 'error') {
      return 'Bildbeschreibung-Modell konnte nicht geladen werden. Bitte Seite neu laden.';
    }
    return null;
  }, [analysisError, isReview, modelStatus, webcamStatus]);

  const statusIsError = useMemo(() => {
    if (analysisError) return true;
    if (!isReview) return webcamStatus === 'error';
    return modelStatus === 'error';
  }, [analysisError, isReview, modelStatus, webcamStatus]);

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
          aria-label="Bildbeschreibung Schritte"
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
            disabled={isAnalyzing}
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

          <div className={qaStyles['qa-grid']}>
            <section className={styles['preview-column']}>
              <div className={styles.card}>
                <div className={cx(styles['card-header'], styles.spaced)}>
                  <h3>{isReview ? 'Aufnahme' : 'Live-View'}</h3>
                </div>

                <div className={qaStyles['capture-body']}>
                  {isReview ? (
                    <div className={styles['capture-slot']}>
                      <img className={qaStyles['capture-image']} src={imageURL} alt="Aufnahme" />
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
                    <div className={qaStyles['capture-actions']}>
                      <button
                        className={qaStyles['shutter-button']}
                        type="button"
                        onClick={capture}
                        disabled={webcamStatus !== 'ready'}
                      >
                        <span className={qaStyles['shutter-ring']} aria-hidden="true" />
                        <span className={qaStyles['shutter-label']}>Foto aufnehmen</span>
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
                      className={qaStyles['secondary-button']}
                      type="button"
                      onClick={handleRetake}
                      disabled={isAnalyzing}
                    >
                      Neu aufnehmen
                    </button>
                  ) : null}
                </div>

                {isReview ? (
                  <div className={qaStyles['control-body']}>
                    <p className={qaStyles.placeholder}>Automatische Bildbeschreibung.</p>
                    <button
                      className={cx(styles.primary, styles.block)}
                      type="button"
                      onClick={handleAnalyze}
                      disabled={isAnalyzing}
                    >
                      {isAnalyzing ? 'Analysiere...' : 'Bild beschreiben'}
                    </button>
                  </div>
                ) : (
                  <p className={qaStyles.placeholder}>
                    Nimm ein Foto auf, um eine Beschreibung zu erhalten.
                  </p>
                )}
              </div>

              <div className={styles.card}>
                <div className={cx(styles['card-header'], styles.spaced)}>
                  <h3>Chat-Output</h3>
                  {isReview ? (
                    <span className={qaStyles['output-meta']}>
                      {isAnalyzing ? 'Analysiere...' : 'Bereit'}
                    </span>
                  ) : null}
                </div>

                <div className={qaStyles['chat-log']} role="log" aria-live="polite">
                  {messages.length ? (
                    messages.map((message, index) => (
                      <div
                        key={`${message.role}-${index}`}
                        className={cx(
                          qaStyles['chat-message'],
                          message.role === 'user' && qaStyles.user,
                        )}
                      >
                        <div className={qaStyles['chat-role']}>
                          {message.role === 'user' ? 'Du' : ASSISTANT_LABEL}
                        </div>
                        <p className={qaStyles['chat-text']}>{message.content}</p>
                      </div>
                    ))
                  ) : (
                    <span className={styles['loss-empty']}>Noch keine Analyse.</span>
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
