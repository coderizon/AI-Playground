import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import { useLLM } from '../../hooks/useLLM.js';

import styles from './LLMChat.module.css';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

const ROLE_LABELS = {
  system: 'System',
  user: 'Du',
  assistant: 'Modell',
};

export default function LLMChat() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState([]);
  const chatLogRef = useRef(null);

  const { status, progress, error, generateResponse, modelId } = useLLM();

  const isLoading = status === 'loading';
  const isGenerating = status === 'generating';
  const isReady = status === 'ready';

  const modelLabel = useMemo(() => {
    if (!modelId) return 'LLM';
    const parts = String(modelId).split('/');
    return parts[parts.length - 1];
  }, [modelId]);

  const statusMessage = useMemo(() => {
    if (error) return error?.message ?? 'Das LLM konnte nicht geladen werden.';
    if (isGenerating) return 'Antwort wird erstellt...';
    return null;
  }, [error, isGenerating]);

  const statusIsError = Boolean(error);
  const progressValue = Number.isFinite(progress) ? Math.round(progress) : 0;
  const canSend = isReady && inputValue.trim().length > 0;
  const inputDisabled = status === 'loading' || status === 'error';

  useEffect(() => {
    const container = chatLogRef.current;
    if (!container) return;
    container.scrollTop = container.scrollHeight;
  }, [messages, isGenerating]);

  const handleSend = useCallback(async () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    if (!isReady) return;

    const nextMessages = [...messages, { role: 'user', content: trimmed }];
    setMessages(nextMessages);
    setInputValue('');

    try {
      const responseText = await generateResponse(nextMessages);
      const reply = responseText?.trim() ? responseText.trim() : 'Keine Antwort erhalten.';
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }]);
    } catch (generateError) {
      console.error(generateError);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Antwort konnte nicht erstellt werden.' },
      ]);
    }
  }, [generateResponse, inputValue, isReady, messages]);

  const handleSubmit = useCallback(
    (event) => {
      event.preventDefault();
      handleSend();
    },
    [handleSend],
  );

  const handleKeyDown = useCallback(
    (event) => {
      if (event.key !== 'Enter' || event.shiftKey) return;
      event.preventDefault();
      handleSend();
    },
    [handleSend],
  );

  return (
    <div className={styles['llm-chat']}>
      <NavigationDrawer
        open={isNavOpen}
        onClose={() => setIsNavOpen(false)}
        drawerId="navigation-drawer"
      />

      <div className={styles['llm-shell']}>
        <header className={styles['llm-topbar']}>
          <button
            className={styles['llm-menu']}
            type="button"
            aria-label={isNavOpen ? 'Menü schließen' : 'Menü öffnen'}
            aria-controls="navigation-drawer"
            aria-expanded={isNavOpen}
            onClick={() => setIsNavOpen((prev) => !prev)}
          >
            <span className={styles['llm-menu-lines']} />
          </button>
          <div className={styles['llm-title']}>LLM Chat</div>
        </header>

        <main className={styles['llm-stage']}>
          {statusMessage ? (
            <div
              className={cx(styles['status-banner'], statusIsError && styles.error)}
              role="status"
              aria-live="polite"
            >
              {statusMessage}
            </div>
          ) : null}

          {isLoading ? (
            <section className={styles.card} aria-live="polite">
              <div className={styles['progress-meta']}>
                <span>Modell wird geladen</span>
                <span>{progressValue}%</span>
              </div>
              <div
                className={styles['progress-bar']}
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={progressValue}
              >
                <span
                  className={styles['progress-fill']}
                  style={{ width: `${progressValue}%` }}
                />
              </div>
            </section>
          ) : null}

          <section className={styles.card}>
            <div className={styles['card-header']}>
              <div>
                <h2 className={styles['card-title']}>Chat</h2>
                <p className={styles['card-subtitle']}>
                  Lokal im Browser · WebGPU
                </p>
              </div>
              <span className={styles['model-pill']} title={modelId || ''}>
                {modelLabel}
              </span>
            </div>

            <div
              className={styles['chat-log']}
              ref={chatLogRef}
              role="log"
              aria-live="polite"
              aria-busy={isGenerating}
            >
              {messages.length ? (
                messages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className={cx(
                      styles['chat-message'],
                      message.role === 'user' && styles.user,
                    )}
                  >
                    <div className={styles['chat-role']}>
                      {ROLE_LABELS[message.role] ?? message.role}
                    </div>
                    <p className={styles['chat-text']}>{message.content}</p>
                  </div>
                ))
              ) : (
                <p className={styles.placeholder}>
                  Stelle eine Frage, um zu starten.
                </p>
              )}

              {isGenerating ? (
                <div className={cx(styles['chat-message'], styles.pending)}>
                  <div className={styles['chat-role']}>Modell</div>
                  <p className={styles['chat-text']}>Denke...</p>
                </div>
              ) : null}
            </div>

            <form className={styles['chat-form']} onSubmit={handleSubmit}>
              <textarea
                className={styles['chat-input']}
                name="prompt"
                placeholder="Schreibe eine Nachricht…"
                rows={2}
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
                onKeyDown={handleKeyDown}
                disabled={inputDisabled}
              />
              <button
                className={styles['send-button']}
                type="submit"
                disabled={!canSend}
              >
                Senden
              </button>
            </form>
            <div className={styles['chat-hint']}>
              Shift + Enter für Zeilenumbruch
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
