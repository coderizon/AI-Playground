import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import * as webllm from '@mlc-ai/web-llm';

const PREFERRED_MODEL_ID = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';
const FALLBACK_MODEL_ID = PREFERRED_MODEL_ID;
const SMALL_MODEL_MATCHERS = [
  /[^0-9]1b[^0-9]/i,
  /[^0-9]2b[^0-9]/i,
  /gemma/i,
  /llama-3\.2/i,
];

function resolveModelId(preferredModelId) {
  const config = webllm.prebuiltAppConfig;
  const modelList = Array.isArray(config?.model_list) ? config.model_list : [];

  if (preferredModelId && modelList.length) {
    const match = modelList.find((model) => {
      const id = model?.model_id ?? model?.model ?? '';
      return id === preferredModelId;
    });
    if (match) return match.model_id ?? match.model ?? preferredModelId;
  }

  if (modelList.length) {
    if (!preferredModelId) {
      const preferred = modelList.find((model) => {
        const id = model?.model_id ?? model?.model ?? '';
        return id === PREFERRED_MODEL_ID;
      });
      if (preferred) return preferred.model_id ?? preferred.model ?? PREFERRED_MODEL_ID;
    }

    const candidate = modelList.find((model) => {
      const id = String(model?.model_id ?? model?.model ?? '');
      if (!id) return false;
      const normalized = ` ${id} `;
      return SMALL_MODEL_MATCHERS.some((matcher) => matcher.test(normalized));
    });

    if (candidate) return candidate.model_id ?? candidate.model;
    return modelList[0]?.model_id ?? modelList[0]?.model ?? preferredModelId ?? FALLBACK_MODEL_ID;
  }

  return preferredModelId ?? FALLBACK_MODEL_ID;
}

function clampProgress(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, Math.round(value)));
}

function getProgressValue(report) {
  if (!report) return 0;
  if (typeof report === 'number') return clampProgress(report * 100);
  if (typeof report.progress === 'number') return clampProgress(report.progress * 100);
  return 0;
}

function getCompletionTokens(usage) {
  const tokens = usage?.completion_tokens;
  return Number.isFinite(tokens) ? tokens : null;
}

function getPromptTokens(usage) {
  const tokens = usage?.prompt_tokens;
  return Number.isFinite(tokens) ? tokens : null;
}

function estimateTokenCount(text) {
  if (!text) return 0;
  const normalized = text.trim();
  if (!normalized) return 0;
  return normalized.split(/\s+/).length;
}

function estimatePromptTokens(messages) {
  if (!Array.isArray(messages)) return 0;
  const combined = messages
    .map((message) => message?.content ?? '')
    .filter(Boolean)
    .join(' ');
  return estimateTokenCount(combined);
}

function clampValue(value, min, max) {
  if (!Number.isFinite(value)) return null;
  return Math.min(max, Math.max(min, value));
}

function buildGenerationConfig(config) {
  if (!config) return {};
  const temperature = clampValue(Number(config.temperature), 0, 2);
  const topP = clampValue(Number(config.topP), 0, 1);
  const topK = clampValue(Number(config.topK), 1, 100);

  const payload = {};
  if (temperature !== null) payload.temperature = temperature;
  if (topP !== null) payload.top_p = topP;
  if (topK !== null) payload.top_k = Math.round(topK);
  return payload;
}

function buildChatCandidates(basePayload, useStream) {
  const candidates = [];
  const hasTopK = basePayload.top_k != null;

  if (useStream) {
    candidates.push({ ...basePayload, stream: true });
    if (hasTopK) {
      const { top_k, ...rest } = basePayload;
      candidates.push({ ...rest, stream: true });
    }
  }

  candidates.push(basePayload);
  if (hasTopK) {
    const { top_k, ...rest } = basePayload;
    candidates.push(rest);
  }

  return candidates;
}

function isWebGPUSupported() {
  if (typeof navigator === 'undefined') return false;
  return Boolean(navigator.gpu);
}

export function useLLM({ enabled = true, modelId } = {}) {
  const [status, setStatus] = useState(enabled ? 'loading' : 'idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const engineRef = useRef(null);
  const enginePromiseRef = useRef(null);
  const mountedRef = useRef(false);
  const perfRef = useRef({ start: 0, firstToken: null, end: 0 });

  const resolvedModelId = useMemo(() => resolveModelId(modelId), [modelId]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const loadEngine = useCallback(async () => {
    if (engineRef.current) return engineRef.current;

    if (!enginePromiseRef.current) {
      setStatus('loading');
      setProgress(0);
      setError(null);

      enginePromiseRef.current = webllm
        .CreateMLCEngine(resolvedModelId, {
          appConfig: webllm.prebuiltAppConfig,
          initProgressCallback: (report) => {
            if (!mountedRef.current) return;
            setProgress(getProgressValue(report));
          },
        })
        .then((engine) => {
          if (!mountedRef.current) {
            engine?.unload?.();
            return engine;
          }
          engineRef.current = engine;
          setProgress(100);
          setStatus('ready');
          return engine;
        })
        .catch((loadError) => {
          enginePromiseRef.current = null;
          if (mountedRef.current) {
            setError(loadError);
            setStatus('error');
          }
          throw loadError;
        });
    }

    return enginePromiseRef.current;
  }, [resolvedModelId]);

  useEffect(() => {
    if (!enabled) {
      setStatus('idle');
      setProgress(0);
      setError(null);
      return undefined;
    }

    if (!isWebGPUSupported()) {
      const gpuError = new Error(
        'WebGPU ist auf diesem Gerät nicht verfügbar. Bitte nutze einen aktuellen Desktop-Browser.',
      );
      setError(gpuError);
      setStatus('error');
      return undefined;
    }

    let cancelled = false;

    loadEngine().catch(() => {
      if (cancelled) return;
    });

    return () => {
      cancelled = true;
    };
  }, [enabled, loadEngine]);

  useEffect(() => {
    return () => {
      const engine = engineRef.current;
      engineRef.current = null;
      enginePromiseRef.current = null;
      if (engine?.unload) {
        engine.unload();
      }
    };
  }, []);

  const generateResponse = useCallback(
    async (messages, { onDelta, config } = {}) => {
      if (!isWebGPUSupported()) {
        const gpuError = new Error(
          'WebGPU ist auf diesem Gerät nicht verfügbar. Bitte nutze einen aktuellen Desktop-Browser.',
        );
        if (mountedRef.current) {
          setError(gpuError);
          setStatus('error');
        }
        throw gpuError;
      }

      const engine = await loadEngine();

      if (mountedRef.current) {
        setError(null);
        setStatus('generating');
      }

      try {
        const perf = perfRef.current;
        perf.start = performance.now();
        perf.firstToken = null;
        perf.end = 0;

        const sanitizedMessages = Array.isArray(messages)
          ? messages.map((message) => ({
              role: message?.role,
              content: message?.content,
            }))
          : messages;
        const promptTokenFallback = estimatePromptTokens(sanitizedMessages);
        const generationConfig = buildGenerationConfig(config);

        if (!engine?.chat?.completions?.create) {
          const apiError = new Error(
            'WebLLM Chat API ist nicht verfügbar. Bitte aktualisiere @mlc-ai/web-llm.',
          );
          if (mountedRef.current) {
            setError(apiError);
            setStatus('error');
          }
          throw apiError;
        }

        const useStream = typeof onDelta === 'function';
        let response = null;
        let usageTokens = null;
        let promptTokens = null;
        let chunkCount = 0;

        const basePayload = {
          messages: sanitizedMessages,
          ...generationConfig,
        };
        const candidates = buildChatCandidates(basePayload, useStream);
        let lastError = null;

        for (const payload of candidates) {
          try {
            response = await engine.chat.completions.create(payload);
            lastError = null;
            break;
          } catch (createError) {
            lastError = createError;
          }
        }

        if (!response && lastError) {
          throw lastError;
        }

        let content = '';
        const isAsyncIterable =
          response && typeof response[Symbol.asyncIterator] === 'function';

        if (useStream && isAsyncIterable) {
          for await (const chunk of response) {
            const delta = chunk?.choices?.[0]?.delta?.content ?? '';
            if (delta) {
              if (perf.firstToken === null) {
                perf.firstToken = performance.now();
              }
              chunkCount += 1;
              content += delta;
              onDelta?.(delta, content);
            }
            const chunkUsageTokens = getCompletionTokens(chunk?.usage);
            if (chunkUsageTokens !== null) {
              usageTokens = chunkUsageTokens;
            }
            const chunkPromptTokens = getPromptTokens(chunk?.usage);
            if (chunkPromptTokens !== null) {
              promptTokens = chunkPromptTokens;
            }
          }
        } else {
          content = response?.choices?.[0]?.message?.content ?? '';
          const responseUsageTokens = getCompletionTokens(response?.usage);
          if (responseUsageTokens !== null) {
            usageTokens = responseUsageTokens;
          }
          const responsePromptTokens = getPromptTokens(response?.usage);
          if (responsePromptTokens !== null) {
            promptTokens = responsePromptTokens;
          }
          if (useStream && content) {
            if (perf.firstToken === null) {
              perf.firstToken = performance.now();
            }
            onDelta?.(content, content);
          }
        }
        perf.end = performance.now();
        if (perf.firstToken === null) {
          perf.firstToken = perf.end;
        }
        const latency = Math.max(0, perf.end - perf.start);
        const ttft = Math.max(0, perf.firstToken - perf.start);
        const prefillDurationMs = Math.max(0, perf.firstToken - perf.start);
        const prefillTokens = promptTokens ?? promptTokenFallback;
        const prefillTps =
          prefillDurationMs > 0 ? prefillTokens / (prefillDurationMs / 1000) : 0;
        const decodeDurationMs = Math.max(0, perf.end - perf.firstToken);
        const fallbackTokens = estimateTokenCount(content) || chunkCount;
        const totalTokens = usageTokens ?? fallbackTokens;
        const tps =
          decodeDurationMs > 0 ? totalTokens / (decodeDurationMs / 1000) : 0;
        if (mountedRef.current) {
          setStatus('ready');
        }
        return {
          text: content,
          stats: {
            latency,
            ttft,
            prefillTps,
            tps,
          },
        };
      } catch (generateError) {
        if (mountedRef.current) {
          setError(generateError);
          setStatus('error');
        }
        throw generateError;
      }
    },
    [loadEngine],
  );

  return {
    status,
    progress,
    error,
    modelId: resolvedModelId,
    generateResponse,
  };
}
