import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MessageCirclePlus, SendHorizontal } from 'lucide-react';
import { TeachableLLM, selectBackend } from '@genai-fi/nanogpt';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import chatStyles from '../llm-chat/LLMChat.module.css';
import './LLMTeachable.css';

const DEFAULT_TEXT = `Heute ist Dienstag. Mia geht in die fuenfte Klasse.
Sie steht frueh auf, packt ihre Tasche und isst Muesli.
In die Tasche kommen: Heft, Stift, Brotbox, Flasche.
Vor der Schule trifft sie ihren Freund Ali und sagt: Guten Morgen.
Ali sagt: Guten Morgen, heute ist Sport!

Im Klassenzimmer sitzt die Klasse im Kreis.
Die Lehrerin fragt: Wer moechte vorlesen?
Mia liest langsam, Ali liest schnell, Lara liest laut.
Die Klasse hoert zu und klatscht am Ende.

In der Pause spielt die Klasse auf dem Hof.
Ein Ball rollt, ein Seil schwingt, ein Drachen steigt.
Zwei Kinder laufen um die Wette. Eins ruft: Los gehts!
Alle lachen, niemand schubst, und alle teilen fair.

Nach der Schule geht Mia nach Hause.
Sie macht Hausaufgaben, isst einen Apfel und trinkt Wasser.
Danach baut sie ein kleines Modell aus Papier.
Sie faltet, sie klebt, sie schreibt ihren Namen darauf.

Am Nachmittag trifft sie ihre Freunde im Park.
Sie spielen Fangen. Sie spielen Verstecken. Sie spielen Fussball.
Der Himmel ist blau, die Wolken ziehen langsam.
Ein Hund bellt, ein Vogel singt, die Sonne ist warm.

Am Abend erzaehlt Mia ihrer Familie vom Tag.
Sie sagt: Heute war spannend, heute war ruhig, heute war lustig.
Die Familie hoert zu und plant den naechsten Tag.

Mini-Geschichte:
Ein kleiner Drache findet einen glitzernden Stein.
Er hebt den Stein auf, laeuft nach Hause und zeigt ihn stolz.
Sein Freund, ein kleiner Fuchs, sagt: Lass uns ein Abenteuer starten!
Sie gehen den Weg entlang, ueber den Bach und durch den Wald.

Merksaetze:
Ein guter Freund hilft. Ein gutes Team teilt. Ein guter Plan spart Zeit.
Ein neuer Tag bringt neue Ideen. Ein langer Weg braucht Pausen.`;

const HEADS = 4;
const GENERATION_CONFIG = {
  maxNewTokens: 180,
  temperature: 0.6,
  topK: 20,
};

const ROLE_LABELS = {
  user: 'Du',
  assistant: 'Modell',
};

const BACKEND_LABELS = {
  webgpu: 'WebGPU',
  webgl: 'WebGL',
  cpu: 'CPU',
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

  const [vocabSize, setVocabSize] = useState(500);
  const [blockSize, setBlockSize] = useState(64);
  const [nEmbed, setNEmbed] = useState(128);
  const [nLayer, setNLayer] = useState(3);
  const [batchSize, setBatchSize] = useState(8);
  const [epochs, setEpochs] = useState(40);
  const [learningRate, setLearningRate] = useState(0.0008);

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
  const generateStopRef = useRef(false);
  const chatLogRef = useRef(null);
  const lastRenderRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    const initBackend = async () => {
      setStatus('WebGPU wird initialisiert...');
      try {
        const selected = await selectBackend('webgpu');
        if (cancelled) return;
        const label = BACKEND_LABELS[selected] ?? 'WebGPU';
        setBackendReady(true);
        setBackendLabel(label);
        setStatus(`Backend: ${label} bereit.`);
        return;
      } catch (error) {
        console.warn('WebGPU nicht verfuegbar, versuche WebGL.', error);
      }

      setStatus('WebGL wird initialisiert...');
      try {
        const selected = await selectBackend('webgl');
        if (cancelled) return;
        const label = BACKEND_LABELS[selected] ?? 'WebGL';
        setBackendReady(true);
        setBackendLabel(label);
        setStatus(`Backend: ${label} bereit.`);
        return;
      } catch (error) {
        console.warn('WebGL nicht verfuegbar, nutze CPU.', error);
      }

      setStatus('CPU wird initialisiert...');
      try {
        const selected = await selectBackend('cpu');
        if (cancelled) return;
        const label = BACKEND_LABELS[selected] ?? 'CPU';
        setBackendReady(true);
        setBackendLabel(label);
        setStatus(`Backend: ${label} bereit (langsam).`);
      } catch (error) {
        console.warn('Backend-Initialisierung fehlgeschlagen.', error);
        if (cancelled) return;
        setBackendReady(false);
        setBackendLabel('');
        setStatus('Backend konnte nicht initialisiert werden.');
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
      modelRef.current = null;
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
    const estimatedVocab = Math.min(vocabSize, Math.max(2, uniqueChars + 1));
    return {
      uniqueChars,
      estimatedVocab,
      textLength: trainingText.length,
    };
  }, [vocabSize, trainingText]);

  const canTrain = backendReady && trainingText.length >= blockSize + 1;
  const canGenerate = hasModel && !isTraining;
  const canSend = canGenerate && !isGenerating && promptInput.trim().length > 0;

  const handleStop = useCallback(() => {
    if (!isTraining) return;
    setStatus('Training kann nicht abgebrochen werden.');
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

    if (trainingText.length < blockSize + 1) {
      setStatus('Text ist zu kurz fuer das Kontextfenster.');
      return;
    }

    setIsTraining(true);
    generateStopRef.current = false;
    setStatus('Modell wird geladen...');
    setProgress(0);
    setLossHistory([]);
    setCurrentLoss(null);
    setActiveVocabSize(null);
    setTokenCount(0);
    setMessages([]);
    setPromptInput('');
    setHasModel(false);

    modelRef.current = null;

    let handleTrainStep = null;

    try {
      const config = {
        vocabSize,
        blockSize,
        nLayer,
        nHead: HEADS,
        nEmbed,
        dropout: 0.1,
        useRope: true,
      };

      const llm = TeachableLLM.create('char', config);
      modelRef.current = llm;

      await new Promise((resolve) => {
        llm.on('loaded', resolve);
      });

      const stepsPerEpoch = Math.max(
        1,
        Math.floor((trainingText.length - blockSize - 1) / batchSize),
      );
      const maxSteps = Math.max(1, stepsPerEpoch * epochs);

      handleTrainStep = (log, trainProgress) => {
        setLossHistory((prev) => [...prev, log.loss]);
        setCurrentLoss(log.loss);

        const stepIndex = Number.isFinite(log?.step) ? log.step + 1 : 0;
        const safeStep = Math.min(stepIndex || 0, maxSteps);
        const percent = Math.round((safeStep / maxSteps) * 100);
        setProgress(percent);

        const remainingSeconds = Number.isFinite(trainProgress?.timeRemaining)
          ? Math.max(0, Math.round(trainProgress.timeRemaining))
          : Number.isFinite(trainProgress?.remaining)
            ? Math.max(0, Math.round(trainProgress.remaining / 1000))
            : null;

        if (Number.isFinite(remainingSeconds)) {
          setStatus(
            `Training laeuft... ${percent}% | Schritt ${safeStep}/${maxSteps} | ${remainingSeconds}s`,
          );
        } else {
          setStatus(
            `Training laeuft... ${percent}% | Schritt ${safeStep}/${maxSteps}`,
          );
        }
      };

      setStatus('Tokeniser wird trainiert...');

      llm.on('trainStep', handleTrainStep);

      const trainedVocabSize = await llm.trainTokeniser([trainingText]);
      setActiveVocabSize(trainedVocabSize);
      setTokenCount(trainingText.length);
      setHasModel(true);

      setStatus('Training laeuft...');

      await llm.train([trainingText], {
        batchSize,
        learningRate,
        epochs,
        maxSteps,
      });

      setProgress(100);
      setStatus('Training abgeschlossen.');
    } catch (error) {
      console.error(error);
      setStatus(`Fehler: ${error.message}`);
    } finally {
      if (modelRef.current && handleTrainStep) {
        modelRef.current.off('trainStep', handleTrainStep);
      }
      setIsTraining(false);
    }
  }, [
    backendReady,
    batchSize,
    blockSize,
    epochs,
    isGenerating,
    isTraining,
    learningRate,
    nEmbed,
    nLayer,
    trainingText,
    vocabSize,
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

      if (!hasModel || !modelRef.current) {
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
        if (generateStopRef.current) return;
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

      const generator = modelRef.current.generator();

      generator.on('tokens', (tokens) => {
        if (generateStopRef.current) return;
        const decoded = modelRef.current?.tokeniser.decode(tokens) ?? '';
        assistantText += decoded;
        const now = Date.now();
        if (now - lastRenderRef.current > 60) {
          lastRenderRef.current = now;
          updateAssistant(true);
        }
      });

      try {
        await generator.generate(seed, {
          maxNewTokens: GENERATION_CONFIG.maxNewTokens,
          temperature: GENERATION_CONFIG.temperature,
          topK: GENERATION_CONFIG.topK,
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
    [hasModel, isGenerating, promptInput],
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
          <h1>Teachable LLM (NanoGPT)</h1>
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
                  value={vocabSize}
                  onChange={(event) => setVocabSize(Number(event.target.value))}
                  disabled={isTraining}
                />
              </label>
              <label className="field">
                <span>Kontextfenster</span>
                <select
                  value={blockSize}
                  onChange={(event) => setBlockSize(Number(event.target.value))}
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
                  value={nEmbed}
                  onChange={(event) => setNEmbed(Number(event.target.value))}
                  disabled={isTraining}
                >
                  <option value={64}>64 Dim</option>
                  <option value={128}>128 Dim</option>
                </select>
              </label>
              <label className="field">
                <span>Transformer-Layer</span>
                <select
                  value={nLayer}
                  onChange={(event) => setNLayer(Number(event.target.value))}
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
                Hinweis: Mindestens {blockSize + 1} Zeichen fuer Training.
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
                <span>{lossHistory.length ? `${lossHistory.length} Schritte` : '---'}</span>
              </div>
              <LossChart values={lossHistory} />
            </div>

            <div className="note">
              <p>
                Training und Speicherverwaltung laufen in @genai-fi/nanogpt,
                kleine Modelle halten die Auslastung niedrig.
              </p>
            </div>
          </section>

          <section className="card llm-teachable-chat-wrapper">
            <div className={cx(chatStyles['chat-card'], 'llm-teachable-chat')}>
              <div className={chatStyles['chat-toolbar']}>
                <span className={chatStyles['model-chip']}>Teachable NanoGPT</span>
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
