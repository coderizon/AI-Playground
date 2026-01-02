import { useCallback, useEffect, useRef, useState } from 'react';

import { env, pipeline } from '@xenova/transformers';

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

// Avoid Vite SPA fallback returning HTML for missing local model files.
env.allowLocalModels = false;
env.allowRemoteModels = true;

let embedderPromise = null;
let embedderInstance = null;

export function useTextEmbedder({ enabled = true } = {}) {
  const [status, setStatus] = useState(enabled ? 'loading' : 'idle');
  const [error, setError] = useState(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const loadEmbedder = useCallback(async () => {
    if (embedderInstance) return embedderInstance;

    if (!embedderPromise) {
      if (mountedRef.current) {
        setStatus('loading');
        setError(null);
      }

      embedderPromise = pipeline('feature-extraction', MODEL_ID)
        .then((loaded) => {
          embedderInstance = loaded;
          return loaded;
        })
        .catch((loadError) => {
          embedderPromise = null;
          throw loadError;
        });
    }

    try {
      const embedder = await embedderPromise;
      if (mountedRef.current) {
        setStatus('ready');
        setError(null);
      }
      return embedder;
    } catch (loadError) {
      if (mountedRef.current) {
        setError(loadError);
        setStatus('error');
      }
      throw loadError;
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      setStatus('idle');
      setError(null);
      return undefined;
    }

    if (embedderInstance) {
      setStatus('ready');
      setError(null);
      return undefined;
    }

    let cancelled = false;

    loadEmbedder().catch(() => {
      if (cancelled) return;
    });

    return () => {
      cancelled = true;
    };
  }, [enabled, loadEmbedder]);

  const extractFeatures = useCallback(
    async (text) => {
      if (!enabled) return null;
      const normalizedText =
        typeof text === 'string' ? text.trim() : String(text ?? '').trim();
      if (!normalizedText) return null;

      try {
        const embedder = await loadEmbedder();
        const output = await embedder(normalizedText, { pooling: 'mean', normalize: true });
        const data = output?.data;

        if (data instanceof Float32Array) return data;
        if (ArrayBuffer.isView(data)) return Float32Array.from(data);
        if (Array.isArray(data)) return Float32Array.from(data);
        if (typeof output?.tolist === 'function') {
          const list = await output.tolist();
          if (Array.isArray(list)) {
            const flattened = Array.isArray(list[0]) ? list.flat() : list;
            return Float32Array.from(flattened);
          }
        }
      } catch (inferenceError) {
        console.error(inferenceError);
        if (mountedRef.current) {
          setError(inferenceError);
          setStatus('error');
        }
      }

      return null;
    },
    [enabled, loadEmbedder],
  );

  return {
    status,
    error,
    modelId: MODEL_ID,
    extractFeatures,
  };
}
