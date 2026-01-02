import { useCallback, useEffect, useMemo, useState } from 'react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import ModelSwitcher from '../../components/common/ModelSwitcher.jsx';
import { useTextEmbedder } from '../../hooks/useTextEmbedder.js';
import { useTextTransferLearning } from '../../hooks/useTextTransferLearning.js';
import styles from './TextClassification.module.css';

const PREDICTION_DEBOUNCE_MS = 300;

const CLASS_COLORS = [
  '#3f73ff', // blue
  '#22c55e', // green
  '#f59e0b', // amber
  '#a855f7', // violet
  '#ef4444', // red
  '#14b8a6', // teal
  '#e11d48', // rose
  '#6366f1', // indigo
  '#84cc16', // lime
  '#0ea5e9', // sky
  '#f97316', // orange
  '#8b5cf6', // purple
];

function getClassColor(index) {
  if (index < CLASS_COLORS.length) return CLASS_COLORS[index];

  const hue = Math.round((index * 137.508) % 360);
  return `hsl(${hue} 82% 55%)`;
}

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function TextClassification() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [classExamples, setClassExamples] = useState(() => [[]]);
  const [isPredicting, setIsPredicting] = useState(false);

  const { status: embedderStatus, error: embedderError, extractFeatures } = useTextEmbedder();

  const {
    classes,
    probabilities,
    collectingClassIndex,
    isTraining,
    trainingPercent,
    isTrained,
    pendingExampleCount,
    canCollect,
    canTrain,
    trainBlockers,
    addClass,
    updateClassName,
    clearDefaultClassName,
    normalizeClassName,
    collectExample,
    clearClassExamples,
    removeClass,
    train,
    predict,
  } = useTextTransferLearning({
    extractFeatures,
    modelStatus: embedderStatus,
    epochs,
    batchSize,
    learningRate,
  });

  const trimmedInput = inputValue.trim();
  const canAddExample = canCollect && trimmedInput.length > 0;

  useEffect(() => {
    setClassExamples((prev) => {
      if (prev.length >= classes.length) return prev;
      const missing = classes.length - prev.length;
      return [...prev, ...Array.from({ length: missing }, () => [])];
    });
  }, [classes.length]);

  useEffect(() => {
    if (!isTrained) {
      setIsPredicting(false);
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(() => {
      setIsPredicting(true);
      Promise.resolve(predict(inputValue))
        .catch(() => {})
        .finally(() => {
          if (cancelled) return;
          setIsPredicting(false);
        });
    }, PREDICTION_DEBOUNCE_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [inputValue, isTrained, predict]);

  const handleAddClass = useCallback(() => {
    if (isTraining) return;
    addClass();
    setClassExamples((prev) => [...prev, []]);
  }, [addClass, isTraining]);

  const handleRemoveClass = useCallback(
    (classIndex) => {
      if (classes.length <= 1) return;
      if (isTraining) return;
      const removed = removeClass(classIndex);
      if (!removed) return;
      setClassExamples((prev) => prev.filter((_, index) => index !== classIndex));
    },
    [classes.length, isTraining, removeClass],
  );

  const handleClearExamples = useCallback(
    (classIndex) => {
      if (isTraining) return;
      clearClassExamples(classIndex);
      setClassExamples((prev) =>
        prev.map((items, index) => (index === classIndex ? [] : items)),
      );
    },
    [clearClassExamples, isTraining],
  );

  const handleCollectExample = useCallback(
    async (classIndex) => {
      if (!trimmedInput) return;
      const didCollect = await collectExample(classIndex, trimmedInput);
      if (!didCollect) return;
      setClassExamples((prev) =>
        prev.map((items, index) => (index === classIndex ? [...items, trimmedInput] : items)),
      );
    },
    [collectExample, trimmedInput],
  );

  const bestPrediction = useMemo(() => {
    if (!isTrained) return null;
    if (!classes.length || !probabilities.length) return null;
    const bestIndex = probabilities.reduce(
      (bestIdx, value, index) => (value > probabilities[bestIdx] ? index : bestIdx),
      0,
    );
    const label = classes[bestIndex]?.name ?? '';
    const value = probabilities[bestIndex] ?? 0;
    return { label, value };
  }, [classes, isTrained, probabilities]);

  const statusMessage = useMemo(() => {
    if (embedderError) {
      return embedderError.message ?? 'Embedding-Modell konnte nicht geladen werden.';
    }
    if (embedderStatus === 'loading') {
      return 'Embedding-Modell wird geladen.';
    }
    if (embedderStatus === 'error') {
      return 'Embedding-Modell konnte nicht geladen werden.';
    }
    if (pendingExampleCount > 0) {
      return `Beispiele werden verarbeitet (${pendingExampleCount}).`;
    }
    return null;
  }, [embedderError, embedderStatus, pendingExampleCount]);

  const statusIsError = Boolean(embedderError) || embedderStatus === 'error';

  return (
    <div className={styles['text-classification']}>
      <NavigationDrawer
        open={isNavOpen}
        onClose={() => setIsNavOpen(false)}
        drawerId="navigation-drawer"
      />

      <div className={styles['tc-shell']}>
        <header className={styles['tc-topbar']}>
          <button
            className={styles['tc-menu']}
            type="button"
            aria-label={isNavOpen ? 'Menü schließen' : 'Menü öffnen'}
            aria-controls="navigation-drawer"
            aria-expanded={isNavOpen}
            onClick={() => setIsNavOpen((prev) => !prev)}
          >
            <span className={styles['tc-menu-lines']} />
          </button>
          <ModelSwitcher />
        </header>

        <main className={styles['tc-stage']}>
          {statusMessage ? (
            <div
              className={cx(styles['status-banner'], statusIsError && styles.error)}
              role="status"
              aria-live="polite"
            >
              {statusMessage}
            </div>
          ) : null}

          <section className={cx(styles.card, styles['input-card'])}>
            <div className={styles['input-header']}>
              <h2>Trainingssatz</h2>
              {isTrained ? <span className={styles['input-pill']}>Testfeld aktiv</span> : null}
            </div>
            <textarea
              className={styles['text-input']}
              rows={3}
              placeholder="Gib einen Beispielsatz ein..."
              value={inputValue}
              onChange={(event) => setInputValue(event.target.value)}
            />
            <div className={styles['input-meta']}>
              <span>
                {isTrained
                  ? 'Nach dem Training dient dieses Feld zum Testen.'
                  : 'Beispiele eingeben und einer Klasse zuordnen.'}
              </span>
              <span className={styles['input-status']}>
                {embedderStatus === 'ready'
                  ? 'Modell bereit'
                  : embedderStatus === 'loading'
                    ? 'Modell lädt...'
                    : embedderStatus === 'error'
                      ? 'Modellfehler'
                      : 'Modell inaktiv'}
              </span>
            </div>
          </section>

          <div className={styles['tc-grid']}>
            <section className={styles['classes-column']}>
              {classes.map((cls, index) => {
                const examples = classExamples[index] ?? [];
                const isCollecting = collectingClassIndex === index;

                return (
                  <div key={cls.id} className={cx(styles.card, styles['class-card'])}>
                    <div className={styles['card-header']}>
                      <input
                        className={styles['class-name-input']}
                        value={cls.name}
                        onChange={(event) => updateClassName(index, event.target.value)}
                        onFocus={() => clearDefaultClassName(index)}
                        onBlur={() => normalizeClassName(index)}
                        aria-label={`Name für Klasse ${index + 1}`}
                      />
                      <div className={styles['class-meta']}>
                        <span className={styles['class-count']}>
                          {cls.exampleCount} Beispiele
                        </span>
                        <button
                          className={cx(styles.ghost, styles.danger)}
                          type="button"
                          onClick={() => handleRemoveClass(index)}
                          disabled={classes.length <= 1 || isTraining}
                        >
                          Klasse entfernen
                        </button>
                      </div>
                    </div>

                    <div className={styles['class-actions']}>
                      <button
                        className={cx(styles.primary, styles.block)}
                        type="button"
                        onClick={() => handleCollectExample(index)}
                        disabled={!canAddExample || isCollecting}
                      >
                        {isCollecting ? 'Wird hinzugefügt...' : 'Diesen Text hinzufügen'}
                      </button>
                      <button
                        className={styles.ghost}
                        type="button"
                        onClick={() => handleClearExamples(index)}
                        disabled={cls.exampleCount === 0 || isTraining}
                      >
                        Beispiele löschen
                      </button>
                    </div>

                    <div className={styles['example-list']} aria-live="polite">
                      {examples.length ? (
                        examples.map((text, textIndex) => (
                          <span
                            key={`${cls.id}-${textIndex}`}
                            className={styles['example-chip']}
                            title={text}
                          >
                            {text}
                          </span>
                        ))
                      ) : (
                        <span className={styles['example-empty']}>Noch keine Beispiele</span>
                      )}
                    </div>
                  </div>
                );
              })}

              <div
                className={cx(styles.card, styles.dashed)}
                role="button"
                tabIndex={0}
                aria-disabled={isTraining}
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

            <section className={styles['training-column']}>
              <div className={cx(styles.card, styles['training-card'])}>
                <div className={cx(styles['card-header'], styles.spaced)}>
                  <div>
                    <h3>Training</h3>
                    <p className={styles['card-subtitle']}>
                      Transfer-Learning auf Text-Embeddings
                    </p>
                  </div>
                  <button
                    className={styles.primary}
                    type="button"
                    onClick={train}
                    disabled={!canTrain}
                  >
                    Modell trainieren
                  </button>
                </div>

                {!isTraining && !canTrain && trainBlockers.length ? (
                  <div className={styles['status-banner']} role="status" aria-live="polite">
                    <div className={styles['status-title']}>Training deaktiviert</div>
                    <ul className={styles['status-list']}>
                      {trainBlockers.map((reason, index) => (
                        <li key={`${index}-${reason}`} className={styles['status-item']}>
                          {reason}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                <div className={cx(styles['training-progress'], !isTraining && styles.hidden)}>
                  <div className={styles['training-progress-meta']}>
                    <span>Training läuft...</span>
                    <span>{trainingPercent}%</span>
                  </div>
                  <div className={styles['training-progress-bar']}>
                    <div
                      className={styles['training-progress-fill']}
                      style={{ width: `${trainingPercent}%` }}
                    />
                  </div>
                </div>

                <details open className={styles.accordion}>
                  <summary>Erweitert</summary>
                  <div className={styles['form-grid']}>
                    <label>
                      <span>Epochen</span>
                      <input
                        type="number"
                        min={1}
                        step={1}
                        value={epochs}
                        onChange={(event) => {
                          const next = Number(event.target.value);
                          setEpochs(Number.isFinite(next) && next >= 1 ? Math.floor(next) : 1);
                        }}
                      />
                    </label>
                    <label>
                      <span>Batchgröße</span>
                      <input
                        type="number"
                        min={1}
                        step={1}
                        value={batchSize}
                        onChange={(event) => {
                          const next = Number(event.target.value);
                          setBatchSize(Number.isFinite(next) && next >= 1 ? Math.floor(next) : 1);
                        }}
                      />
                    </label>
                    <label>
                      <span>Lernrate</span>
                      <input
                        type="number"
                        min={0.000001}
                        step={0.0001}
                        value={learningRate}
                        onChange={(event) => {
                          const next = Number(event.target.value);
                          setLearningRate(Number.isFinite(next) && next > 0 ? next : 0.000001);
                        }}
                      />
                    </label>
                  </div>
                </details>
              </div>

              <div className={cx(styles.card, styles['preview-card'])}>
                <div className={styles['card-header']}>
                  <div>
                    <h3>Preview</h3>
                    <p className={styles['card-subtitle']}>
                      {isTrained
                        ? isPredicting
                          ? 'Vorhersage wird aktualisiert...'
                          : 'Vorhersage bereit.'
                        : 'Noch nicht trainiert.'}
                    </p>
                  </div>
                  {bestPrediction?.label ? (
                    <span className={styles['status-pill']}>
                      {bestPrediction.label} · {Math.round(bestPrediction.value * 100)}%
                    </span>
                  ) : null}
                </div>

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
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
