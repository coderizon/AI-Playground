import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { MessageCirclePlus, SendHorizontal, SlidersHorizontal } from 'lucide-react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import ModelSwitcher from '../../components/common/ModelSwitcher.jsx';
import ConfigDialog from '../../components/common/ConfigDialog.jsx';
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
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [modelConfig, setModelConfig] = useState({
    topK: 40,
    topP: 1,
    temperature: 0.8,
    maxTokens: 2048,
    repetitionPenalty: 1.1,
    presencePenalty: 0,
    frequencyPenalty: 0,
    seed: null,
    accelerator: 'GPU',
  });
  const [messages, setMessages] = useState([]);
  const [expandedStats, setExpandedStats] = useState({});
  const chatLogRef = useRef(null);
  const shouldAutoScrollRef = useRef(true);
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

    const updateAutoScroll = () => {
      const threshold = 24;
      const distanceFromBottom =
        container.scrollHeight - container.scrollTop - container.clientHeight;
      shouldAutoScrollRef.current = distanceFromBottom <= threshold;
    };

    updateAutoScroll();
    container.addEventListener('scroll', updateAutoScroll, { passive: true });

    return () => {
      container.removeEventListener('scroll', updateAutoScroll);
    };
  }, []);

  useEffect(() => {
    if (!shouldAutoScrollRef.current) return;
    const container = chatLogRef.current;
    if (!container) return;
    const raf = requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
    return () => cancelAnimationFrame(raf);
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

  const formatSeconds = useCallback((value) => {
    if (!Number.isFinite(value)) return '--';
    return (value / 1000).toFixed(2);
  }, []);

  const formatTps = useCallback((value) => {
    if (!Number.isFinite(value)) return '--';
    return value.toFixed(1);
  }, []);

  const toggleStats = useCallback((index) => {
    setExpandedStats((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  }, []);

  const handleConfigApply = useCallback((nextConfig) => {
    setModelConfig(nextConfig);
    setIsConfigOpen(false);
  }, []);

  const handleConfigClose = useCallback(() => {
    setIsConfigOpen(false);
  }, []);

  const handleSend = useCallback(async () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    if (!isReady) return;

    const nextMessages = [
      ...messages,
      { role: 'user', content: trimmed, meta: { stats: null, device: null } },
    ];
    const assistantIndex = nextMessages.length;
    setMessages([
      ...nextMessages,
      { role: 'assistant', content: '', meta: { stats: null, device: 'gpu' } },
    ]);
    setInputValue('');

    try {
      const responsePayload = await generateResponse(nextMessages, {
        config: modelConfig,
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
      const responseText =
        typeof responsePayload === 'string' ? responsePayload : responsePayload?.text;
      const responseStats =
        typeof responsePayload === 'string' ? null : responsePayload?.stats ?? null;
      const reply = responseText?.trim() ? responseText.trim() : 'Keine Antwort erhalten.';
      setMessages((prev) => {
        if (!prev[assistantIndex]) return prev;
        const updated = [...prev];
        const current = updated[assistantIndex];
        updated[assistantIndex] = {
          ...current,
          content: reply,
          meta: {
            ...(current?.meta ?? {}),
            stats: responseStats,
            device: current?.meta?.device ?? 'gpu',
          },
        };
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
          meta: { stats: null, device: 'gpu' },
        };
        return updated;
      });
    }
  }, [generateResponse, inputValue, isReady, messages, modelConfig, triggerHaptic]);

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
              <div className={styles['chat-toolbar-actions']}>
                <button
                  className={styles['chat-toolbar-action']}
                  type="button"
                  aria-label="Einstellungen"
                  onClick={() => setIsConfigOpen(true)}
                >
                  <SlidersHorizontal size={18} strokeWidth={2} />
                </button>
                <button
                  className={styles['chat-toolbar-action']}
                  type="button"
                  aria-label="Neue Unterhaltung"
                >
                  <MessageCirclePlus size={18} strokeWidth={2} />
                </button>
              </div>
            </div>

            <div
              className={styles['chat-log']}
              ref={chatLogRef}
              role="log"
              aria-live="polite"
              aria-busy={isGenerating}
            >
              {messages.length
                ? messages.map((message, index) => {
                    const isAssistant = message.role === 'assistant';
                    const isPendingMessage =
                      isGenerating && isAssistant && index === messages.length - 1;
                    const content = message.content || '';
                    const hasContent = Boolean(content && content.trim().length > 0);
                    const showLoader = isPendingMessage && !hasContent;
                    const stats = message.meta?.stats;
                    const hasStats =
                      stats &&
                      Number.isFinite(stats.latency) &&
                      Number.isFinite(stats.ttft) &&
                      Number.isFinite(stats.tps);
                    const isStatsOpen = Boolean(expandedStats[index]);
                    return (
                      <div
                        key={`${message.role}-${index}`}
                        className={cx(
                          styles['chat-message-group'],
                          message.role === 'user' && styles.user,
                        )}
                      >
                        {isAssistant ? (
                          <div className={styles['chat-role']}>
                            <span className={styles['chat-role-label']}>
                              {ROLE_LABELS[message.role] ?? message.role}
                            </span>
                            <span className={styles['gpu-badge']}>Modell auf GPU</span>
                          </div>
                        ) : null}
                        <div
                          className={cx(
                            styles['chat-message'],
                            message.role === 'user' && styles.user,
                            isPendingMessage && styles.pending,
                            showLoader && styles.loading,
                          )}
                        >
                          {showLoader ? (
                            <div
                              className={styles['chat-loader']}
                              role="status"
                              aria-live="polite"
                              aria-label="Antwort wird erstellt"
                            >
                              <span className={styles['hex-loader']}>
                                <span className={styles['hex-core']} />
                              </span>
                            </div>
                          ) : (
                            <p className={styles['chat-text']}>{content}</p>
                          )}
                        </div>
                        {isAssistant && !isPendingMessage && hasStats ? (
                          <div className={styles['chat-stats']}>
                            <button
                              className={styles['chat-stats-toggle']}
                              type="button"
                              onClick={() => toggleStats(index)}
                            >
                              {isStatsOpen ? 'Statistiken ausblenden' : 'Statistiken anzeigen'}
                            </button>
                            {isStatsOpen ? (
                              <div className={styles['chat-stats-panel']}>
                                <div className={styles['chat-stat']}>
                                  <span className={styles['chat-stat-label']}>
                                    1st Token
                                  </span>
                                  <span className={styles['chat-stat-value']}>
                                    {formatSeconds(stats.ttft)} s
                                  </span>
                                </div>
                                <div className={styles['chat-stat']}>
                                  <span className={styles['chat-stat-label']}>
                                    Prefill-Rate
                                  </span>
                                  <span className={styles['chat-stat-value']}>
                                    {formatTps(stats.prefillTps)} tok/s
                                  </span>
                                </div>
                                <div className={styles['chat-stat']}>
                                  <span className={styles['chat-stat-label']}>
                                    Latenz
                                  </span>
                                  <span className={styles['chat-stat-value']}>
                                    {formatSeconds(stats.latency)} s
                                  </span>
                                </div>
                                <div className={styles['chat-stat']}>
                                  <span className={styles['chat-stat-label']}>
                                    Decode-Rate
                                  </span>
                                  <span className={styles['chat-stat-value']}>
                                    {formatTps(stats.tps)} tok/s
                                  </span>
                                </div>
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    );
                  })
                : null}
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
          </section>
        </main>
      </div>

      <ConfigDialog
        isOpen={isConfigOpen}
        config={modelConfig}
        onApply={handleConfigApply}
        onClose={handleConfigClose}
      />
    </div>
  );
}
