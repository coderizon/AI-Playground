import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-wasm';
import { MessageCirclePlus, SendHorizontal } from 'lucide-react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import chatStyles from '../llm-chat/LLMChat.module.css';
import {
  createCharTokenizer,
  createNanoTransformer,
  generateNanoText,
  setEncoderTrainable,
  trainNanoTransformer,
  warmupModel,
} from './nanoTransformer.js';
import './LLMTeachable.css';

const DEFAULT_TEXT = `Das ist ein winziger Demo-Datensatz.
Das Modell soll kurze Muster lernen.
Du kannst beliebigen Text einfuegen.`;

const HEADS = 4;
const GENERATION_CONFIG = {
  maxNewTokens: 120,
  temperature: 0.8,
  topK: 40,
};

const ROLE_LABELS = {
  user: 'Du',
  assistant: 'Modell',
};

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

function LossChart({ values }) {
  if (!values.length) {
    return <div className="loss-empty">Noch keine Loss-Daten.</div>;
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const span = values.length > 1 ? values.length - 1 : 1;
  const points = values
    .map((value, index) => {
      const x = (index / span) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(' ');

  const lastValue = values[values.length - 1];
  const lastX = ((values.length - 1) / span) * 100;
  const lastY = 100 - ((lastValue - min) / range) * 100;

  return (
    <svg
      className="loss-chart"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      role="img"
      aria-label="Loss-Verlauf"
    >
      <polyline points={points} fill="none" stroke="#2563eb" strokeWidth="2" />
      <circle cx={lastX} cy={lastY} r="2.6" fill="#1d4ed8" />
    </svg>
  );
}

export default function LLMTeachable() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [backendLabel, setBackendLabel] = useState('');
  const [status, setStatus] = useState('Backend wird initialisiert...');
  const [trainingText, setTrainingText] = useState(DEFAULT_TEXT);

  const [maxVocabSize, setMaxVocabSize] = useState(500);
  const [contextWindow, setContextWindow] = useState(64);
  const [embedDim, setEmbedDim] = useState(64);
  const [numLayers, setNumLayers] = useState(2);
  const [batchSize, setBatchSize] = useState(4);
  const [epochs, setEpochs] = useState(12);
  const [stepsPerEpoch, setStepsPerEpoch] = useState(20);
  const [learningRate, setLearningRate] = useState(0.001);
  const [freezeEncoder, setFreezeEncoder] = useState(false);

  const [lossHistory, setLossHistory] = useState([]);
  const [currentLoss, setCurrentLoss] = useState(null);
  const [progress, setProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [activeVocabSize, setActiveVocabSize] = useState(null);
  const [tokenCount, setTokenCount] = useState(0);
  const [backendReady, setBackendReady] = useState(false);
  const [hasModel, setHasModel] = useState(false);
  const [promptInput, setPromptInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const modelRef = useRef(null);
  const tokenizerRef = useRef(null);
  const optimizerRef = useRef(null);
  const stopRef = useRef(false);
  const generateStopRef = useRef(false);
  const trainedConfigRef = useRef(null);
  const chatLogRef = useRef(null);
  const lastRenderRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    const initBackend = async () => {
      setStatus('WebGPU wird initialisiert...');
      try {
        await tf.setBackend('webgpu');
        await tf.ready();
        if (cancelled) return;
        setBackendReady(true);
        setBackendLabel('WebGPU');
        setStatus('Backend: WebGPU bereit.');
      } catch (error) {
        console.warn('WebGPU nicht verfuegbar, versuche WASM.', error);
        try {
          await tf.setBackend('wasm');
          await tf.ready();
          if (cancelled) return;
          setBackendReady(true);
          setBackendLabel('WASM');
          setStatus('Backend: WASM bereit.');
        } catch (fallbackError) {
          console.warn('WASM nicht verfuegbar, nutze CPU.', fallbackError);
          await tf.setBackend('cpu');
          await tf.ready();
          if (cancelled) return;
          setBackendReady(true);
          setBackendLabel('CPU');
          setStatus('Backend: CPU bereit (langsam).');
        }
      }
    };

    initBackend();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      generateStopRef.current = true;
      if (modelRef.current) {
        modelRef.current.dispose();
        modelRef.current = null;
      }
      tokenizerRef.current = null;
      trainedConfigRef.current = null;
      if (optimizerRef.current) {
        optimizerRef.current.dispose();
        optimizerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const container = chatLogRef.current;
    if (!container) return undefined;
    const raf = requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
    return () => cancelAnimationFrame(raf);
  }, [messages, isGenerating]);

  const estimatedStats = useMemo(() => {
    const uniqueChars = new Set(trainingText).size;
    const estimatedVocab = Math.min(maxVocabSize, Math.max(2, uniqueChars + 1));
    return {
      uniqueChars,
      estimatedVocab,
      textLength: trainingText.length,
    };
  }, [maxVocabSize, trainingText]);

  const canTrain = backendReady && trainingText.length >= contextWindow + 1;
  const canGenerate = hasModel && !isTraining;
  const canSend = canGenerate && !isGenerating && promptInput.trim().length > 0;

  const handleStop = useCallback(() => {
    if (!isTraining) return;
    stopRef.current = true;
    setStatus('Training wird gestoppt...');
  }, [isTraining]);

  const handleTrain = useCallback(async () => {
    if (isTraining) return;
    if (isGenerating) {
      setStatus('Bitte Generierung abschliessen oder abbrechen.');
      return;
    }
    if (!backendReady) {
      setStatus('Backend ist noch nicht bereit.');
      return;
    }

    if (trainingText.length < contextWindow + 1) {
      setStatus('Text ist zu kurz fuer das Kontextfenster.');
      return;
    }

    setIsTraining(true);
    stopRef.current = false;
    setStatus('Tokenisierung und Modellaufbau...');
    setProgress(0);
    setLossHistory([]);
    setCurrentLoss(null);
    setMessages([]);
    setPromptInput('');
    setHasModel(false);

    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }
    if (optimizerRef.current) {
      optimizerRef.current.dispose();
      optimizerRef.current = null;
    }

    try {
      const tokenizer = createCharTokenizer(trainingText, maxVocabSize);
      const tokens = tokenizer.encode(trainingText);
      tokenizerRef.current = tokenizer;
      setActiveVocabSize(tokenizer.vocabSize);
      setTokenCount(tokens.length);
      trainedConfigRef.current = {
        contextWindow,
        vocabSize: tokenizer.vocabSize,
      };

      const model = createNanoTransformer({
        vocabSize: tokenizer.vocabSize,
        contextWindow,
        embedDim,
        numHeads: HEADS,
        numLayers,
        ffDim: embedDim * 2,
      });

      setEncoderTrainable(model, !freezeEncoder);
      modelRef.current = model;
      setHasModel(true);

      setStatus('Warm-up fuer Backend...');
      await warmupModel(model, contextWindow);

      const optimizer = tf.train.adam(learningRate);
      optimizerRef.current = optimizer;

      setStatus('Training laeuft...');

      await trainNanoTransformer({
        model,
        tokens,
        vocabSize: tokenizer.vocabSize,
        contextWindow,
        batchSize,
        epochs,
        stepsPerEpoch,
        optimizer,
        onEpochEnd: ({ epoch, loss }) => {
          setLossHistory((prev) => [...prev, loss]);
          setCurrentLoss(loss);
          setProgress(Math.round(((epoch + 1) / epochs) * 100));
          setStatus(`Epoche ${epoch + 1}/${epochs}`);
        },
        shouldStop: () => stopRef.current,
      });

      if (stopRef.current) {
        setStatus('Training abgebrochen.');
      } else {
        setStatus('Training abgeschlossen.');
      }
    } catch (error) {
      console.error(error);
      setStatus(`Fehler: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  }, [
    backendReady,
    batchSize,
    contextWindow,
    embedDim,
    epochs,
    freezeEncoder,
    isGenerating,
    isTraining,
    learningRate,
    maxVocabSize,
    numLayers,
    stepsPerEpoch,
    trainingText,
  ]);

  const handleClearChat = useCallback(() => {
    if (isGenerating) {
      generateStopRef.current = true;
    }
    setMessages([]);
    setPromptInput('');
  }, [isGenerating]);

  const handleGenerate = useCallback(
    async (event) => {
      if (event) event.preventDefault();
      if (isGenerating) return;

      const seed = promptInput.trim();
      if (!seed) return;

      if (!hasModel || !modelRef.current || !tokenizerRef.current) {
        setStatus('Bitte zuerst trainieren.');
        return;
      }

      setIsGenerating(true);
      generateStopRef.current = false;
      setStatus('Generiere Text...');
      setPromptInput('');

      let assistantText = '';
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: seed },
        { role: 'assistant', content: '', pending: true },
      ]);

      const updateAssistant = (pending) => {
        setMessages((prev) => {
          if (!prev.length) return prev;
          const updated = [...prev];
          const lastIndex = updated.length - 1;
          const last = updated[lastIndex];
          if (!last || last.role !== 'assistant') return prev;
          updated[lastIndex] = { ...last, content: assistantText, pending };
          return updated;
        });
      };

      const contextForGeneration =
        trainedConfigRef.current?.contextWindow ?? contextWindow;

      try {
        await generateNanoText({
          model: modelRef.current,
          tokenizer: tokenizerRef.current,
          seed,
          contextWindow: contextForGeneration,
          maxNewTokens: GENERATION_CONFIG.maxNewTokens,
          temperature: GENERATION_CONFIG.temperature,
          topK: GENERATION_CONFIG.topK,
          shouldStop: () => generateStopRef.current,
          onToken: (token) => {
            assistantText += token;
            const now = Date.now();
            if (now - lastRenderRef.current > 60) {
              lastRenderRef.current = now;
              updateAssistant(true);
            }
          },
        });

        updateAssistant(false);
        setStatus('Generierung abgeschlossen.');
      } catch (error) {
        console.error(error);
        setStatus(`Fehler: ${error.message}`);
        updateAssistant(false);
      } finally {
        setIsGenerating(false);
      }
    },
    [contextWindow, hasModel, isGenerating, promptInput],
  );

  const handlePromptKeyDown = useCallback(
    (event) => {
      if (event.key !== 'Enter' || event.shiftKey) return;
      event.preventDefault();
      if (!canSend) return;
      handleGenerate();
    },
    [canSend, handleGenerate],
  );

  return (
    <div className="llm-teachable-page">
      <NavigationDrawer
        open={isNavOpen}
        onClose={() => setIsNavOpen(false)}
        drawerId="llm-teachable-drawer"
      />

      <header className="page-header">
        <button
          className="icon-button"
          onClick={() => setIsNavOpen(true)}
          aria-label="Menue oeffnen"
        >
          <span className="icon-lines" />
        </button>
        <div>
          <h1>Teachable LLM (Nano-Transformer)</h1>
          <p className="page-subtitle">
            Trainiere ein winziges Sprachmodell direkt im Browser (iPad/WebGPU).
          </p>
        </div>
      </header>

      <main className="teachable-grid">
        <div className="teachable-left">
          <section className="card control-panel">
            <h2>Datensatz & Tokenisierung</h2>
            <p>Fuege Trainings-Text ein. Das Modell lernt naechstes Zeichen.</p>
            <textarea
              value={trainingText}
              onChange={(event) => setTrainingText(event.target.value)}
              rows={10}
              className="text-input"
              disabled={isTraining}
            />

            <div className="stats-row">
              <div>
                <strong>Zeichen:</strong> {estimatedStats.textLength}
              </div>
              <div>
                <strong>Einzigartig:</strong> {estimatedStats.uniqueChars}
              </div>
              <div>
                <strong>Vokabular:</strong> {estimatedStats.estimatedVocab}
              </div>
            </div>

            <div className="field-grid">
              <label className="field">
                <span>Max. Vokabular</span>
                <input
                  type="number"
                  min={100}
                  max={2000}
                  step={100}
                  value={maxVocabSize}
                  onChange={(event) => setMaxVocabSize(Number(event.target.value))}
                  disabled={isTraining}
                />
              </label>
              <label className="field">
                <span>Kontextfenster</span>
                <select
                  value={contextWindow}
                  onChange={(event) => setContextWindow(Number(event.target.value))}
                  disabled={isTraining}
                >
                  <option value={64}>64 Tokens</option>
                  <option value={32}>32 Tokens</option>
                </select>
              </label>
              <label className="field">
                <span>Heads (fix)</span>
                <input type="text" value={HEADS} readOnly />
              </label>
            </div>
          </section>

          <section className="card control-panel">
            <h2>Modell & Training</h2>
            <div className="field-grid">
              <label className="field">
                <span>Embedding</span>
                <select
                  value={embedDim}
                  onChange={(event) => setEmbedDim(Number(event.target.value))}
                  disabled={isTraining}
                >
                  <option value={64}>64 Dim</option>
                  <option value={128}>128 Dim</option>
                </select>
              </label>
              <label className="field">
                <span>Transformer-Layer</span>
                <select
                  value={numLayers}
                  onChange={(event) => setNumLayers(Number(event.target.value))}
                  disabled={isTraining}
                >
                  <option value={2}>2 Layer</option>
                  <option value={3}>3 Layer</option>
                  <option value={4}>4 Layer</option>
                </select>
              </label>
              <label className="field">
                <span>Batch Size</span>
                <select
                  value={batchSize}
                  onChange={(event) => setBatchSize(Number(event.target.value))}
                  disabled={isTraining}
                >
                  <option value={4}>4</option>
                  <option value={8}>8</option>
                </select>
              </label>
              <label className="field">
                <span>Epochen</span>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={epochs}
                  onChange={(event) => setEpochs(Number(event.target.value))}
                  disabled={isTraining}
                />
              </label>
              <label className="field">
                <span>Steps/Epoche</span>
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={stepsPerEpoch}
                  onChange={(event) => setStepsPerEpoch(Number(event.target.value))}
                  disabled={isTraining}
                />
              </label>
              <label className="field">
                <span>Learning Rate</span>
                <input
                  type="number"
                  min={0.0001}
                  max={0.01}
                  step={0.0001}
                  value={learningRate}
                  onChange={(event) => setLearningRate(Number(event.target.value))}
                  disabled={isTraining}
                />
              </label>
            </div>

            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={freezeEncoder}
                onChange={(event) => setFreezeEncoder(event.target.checked)}
                disabled={isTraining}
              />
              Encoder einfrieren (nur Output-Layer trainieren)
            </label>

            <div className="actions">
              <button
                className="btn-primary"
                onClick={handleTrain}
                disabled={!canTrain || isTraining}
              >
                {isTraining ? 'Trainiere...' : 'Training starten'}
              </button>
              <button
                className="btn-secondary"
                type="button"
                onClick={handleStop}
                disabled={!isTraining}
              >
                Stoppen
              </button>
            </div>
            {!canTrain && (
              <p className="hint">
                Hinweis: Mindestens {contextWindow + 1} Zeichen fuer Training.
              </p>
            )}
          </section>
        </div>

        <div className="teachable-right">
          <section className="card visualization-panel">
            <h2>Status & Verlauf</h2>
            <div className="status-row">
              <div>
                <strong>Backend:</strong> {backendLabel || '...'}
              </div>
              <div>
                <strong>Status:</strong> {status}
              </div>
            </div>

            <div className="metric-grid">
              <div className="metric">
                <span>Aktueller Loss</span>
                <strong>{currentLoss ? currentLoss.toFixed(4) : '--'}</strong>
              </div>
              <div className="metric">
                <span>Vokabular aktiv</span>
                <strong>{activeVocabSize ?? '--'}</strong>
              </div>
              <div className="metric">
                <span>Tokens</span>
                <strong>{tokenCount || '--'}</strong>
              </div>
            </div>

            <div className="progress-container">
              <div className="progress-bar" style={{ width: `${progress}%` }} />
            </div>

            <div className="loss-card">
              <div className="loss-header">
                <span>Loss-Verlauf</span>
                <span>{lossHistory.length ? `${lossHistory.length} Epochen` : '---'}</span>
              </div>
              <LossChart values={lossHistory} />
            </div>

            <div className="note">
              <p>
                Speicherstrategie: Jede Trainings-Iteration laeuft in tf.tidy(),
                Inputs/Labels werden sofort freigegeben, Batch Size ist klein.
              </p>
            </div>
          </section>

          <section className="card llm-teachable-chat-wrapper">
            <div className={cx(chatStyles['chat-card'], 'llm-teachable-chat')}>
              <div className={chatStyles['chat-toolbar']}>
                <span className={chatStyles['model-chip']}>
                  Teachable Nano-Transformer
                </span>
                <div className={chatStyles['chat-toolbar-actions']}>
                  <button
                    className={chatStyles['chat-toolbar-action']}
                    type="button"
                    aria-label="Chat leeren"
                    onClick={handleClearChat}
                  >
                    <MessageCirclePlus size={18} strokeWidth={2} />
                  </button>
                </div>
              </div>

              <div
                className={chatStyles['chat-log']}
                ref={chatLogRef}
                role="log"
                aria-live="polite"
                aria-busy={isGenerating}
              >
                {messages.length ? (
                  messages.map((message, index) => {
                    const isAssistant = message.role === 'assistant';
                    const showLoader = message.pending && !message.content;
                    return (
                      <div
                        key={`${message.role}-${index}`}
                        className={cx(
                          chatStyles['chat-message-group'],
                          message.role === 'user' && chatStyles.user,
                        )}
                      >
                        {isAssistant ? (
                          <div className={chatStyles['chat-role']}>
                            <span className={chatStyles['chat-role-label']}>
                              {ROLE_LABELS[message.role] ?? message.role}
                            </span>
                            <span className={chatStyles['gpu-badge']}>
                              {backendLabel || 'Modell'}
                            </span>
                          </div>
                        ) : null}
                        <div
                          className={cx(
                            chatStyles['chat-message'],
                            message.role === 'user' && chatStyles.user,
                            message.pending && chatStyles.pending,
                            showLoader && chatStyles.loading,
                          )}
                        >
                          {showLoader ? (
                            <div
                              className={chatStyles['chat-loader']}
                              role="status"
                              aria-live="polite"
                              aria-label="Antwort wird erstellt"
                            >
                              <span className={chatStyles['hex-loader']}>
                                <span className={chatStyles['hex-core']} />
                              </span>
                            </div>
                          ) : (
                            <p className={chatStyles['chat-text']}>{message.content}</p>
                          )}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <p className={chatStyles.placeholder}>
                    Gib einen Seed-Text ein und lasse das Modell fortsetzen.
                  </p>
                )}
              </div>

              <form className={chatStyles['chat-form']} onSubmit={handleGenerate}>
                <div className={chatStyles['chat-input-shell']}>
                  <textarea
                    className={chatStyles['chat-input']}
                    name="seed"
                    placeholder="Seed eingeben..."
                    rows={1}
                    value={promptInput}
                    onChange={(event) => setPromptInput(event.target.value)}
                    onKeyDown={handlePromptKeyDown}
                    disabled={!canGenerate || isGenerating}
                  />
                  <button
                    className={chatStyles['send-icon-button']}
                    type="submit"
                    aria-label="Generieren"
                    disabled={!canSend}
                  >
                    <SendHorizontal size={18} strokeWidth={2.2} />
                  </button>
                </div>
              </form>

              <div className={chatStyles['chat-hint']}>
                {hasModel
                  ? `Max ${GENERATION_CONFIG.maxNewTokens} Tokens | Temp ${GENERATION_CONFIG.temperature} | Top-K ${GENERATION_CONFIG.topK}`
                  : 'Textgenerierung ist nach dem Training verfuegbar.'}
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
