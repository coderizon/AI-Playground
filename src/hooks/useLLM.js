import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import * as webllm from '@mlc-ai/web-llm';

const FALLBACK_MODEL_ID = 'Llama-3.2-1B-Instruct-q4f32_1-MLC';
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
    async (messages) => {
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

        const response = await engine.chat.completions.create({ messages });
        const content = response?.choices?.[0]?.message?.content ?? '';
        if (mountedRef.current) {
          setStatus('ready');
        }
        return content;
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
