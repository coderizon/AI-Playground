import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SendHorizontal } from 'lucide-react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import ModelSwitcher from '../../components/common/ModelSwitcher.jsx';
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
  const lastHapticRef = useRef(0);

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

  const triggerHaptic = useCallback(() => {
    if (typeof navigator === 'undefined' || typeof navigator.vibrate !== 'function') {
      return;
    }
    const now = Date.now();
    if (now - lastHapticRef.current < 50) return;
    lastHapticRef.current = now;
    navigator.vibrate(8);
  }, []);

  const handleSend = useCallback(async () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    if (!isReady) return;

    const nextMessages = [...messages, { role: 'user', content: trimmed }];
    const assistantIndex = nextMessages.length;
    setMessages([...nextMessages, { role: 'assistant', content: '' }]);
    setInputValue('');

    try {
      const responseText = await generateResponse(nextMessages, {
        onDelta: (_, fullText) => {
          triggerHaptic();
          setMessages((prev) => {
            if (!prev[assistantIndex]) return prev;
            const updated = [...prev];
            const current = updated[assistantIndex];
            if (!current || current.role !== 'assistant') return prev;
            updated[assistantIndex] = { ...current, content: fullText };
            return updated;
          });
        },
      });
      const reply = responseText?.trim() ? responseText.trim() : 'Keine Antwort erhalten.';
      setMessages((prev) => {
        if (!prev[assistantIndex]) return prev;
        const updated = [...prev];
        updated[assistantIndex] = { ...updated[assistantIndex], content: reply };
        return updated;
      });
    } catch (generateError) {
      console.error(generateError);
      setMessages((prev) => {
        if (!prev[assistantIndex]) return prev;
        const updated = [...prev];
        updated[assistantIndex] = {
          role: 'assistant',
          content: 'Antwort konnte nicht erstellt werden.',
        };
        return updated;
      });
    }
  }, [generateResponse, inputValue, isReady, messages, triggerHaptic]);

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
          <ModelSwitcher />
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
            <section className={cx(styles.card, styles['progress-card'])} aria-live="polite">
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

          <section className={cx(styles.card, styles['chat-card'])}>
            <div className={styles['chat-toolbar']}>
              <span className={styles['model-chip']} title={modelId || ''}>
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
                messages.map((message, index) => {
                  const isPendingMessage =
                    isGenerating &&
                    message.role === 'assistant' &&
                    index === messages.length - 1;
                  const content =
                    message.content || (isPendingMessage ? 'Denke...' : '');
                  return (
                    <div
                      key={`${message.role}-${index}`}
                      className={cx(
                        styles['chat-message'],
                        message.role === 'user' && styles.user,
                        isPendingMessage && styles.pending,
                      )}
                    >
                      <div className={styles['chat-role']}>
                        {ROLE_LABELS[message.role] ?? message.role}
                      </div>
                      <p className={styles['chat-text']}>{content}</p>
                    </div>
                  );
                })
              ) : (
                <p className={styles.placeholder}>
                  Stelle eine Frage, um zu starten.
                </p>
              )}
            </div>

            <form className={styles['chat-form']} onSubmit={handleSubmit}>
              <div className={styles['chat-input-shell']}>
                <textarea
                  className={styles['chat-input']}
                  name="prompt"
                  placeholder="Schreibe eine Nachricht…"
                  rows={1}
                  value={inputValue}
                  onChange={(event) => setInputValue(event.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={inputDisabled}
                />
                <button
                  className={styles['send-icon-button']}
                  type="submit"
                  aria-label="Senden"
                  disabled={!canSend}
                >
                  <SendHorizontal size={18} strokeWidth={2.2} />
                </button>
              </div>
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
