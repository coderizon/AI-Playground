import { useEffect, useRef } from 'react';

import styles from '../../image-classification/ImageClassification.module.css';

const MIN_DB = -100;
const MAX_DB = -30;

function normalizeDb(value) {
  if (!Number.isFinite(value)) return 0;
  const normalized = (value - MIN_DB) / (MAX_DB - MIN_DB);
  return Math.min(1, Math.max(0, normalized));
}

export default function SpectrogramCanvas({ spectrogramRef, isActive }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    };

    resize();
    ctx.fillStyle = '#0f1115';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    if (isActive) {
      const draw = () => {
        const width = canvas.width;
        const height = canvas.height;

        ctx.fillStyle = '#0f1115';
        ctx.drawImage(canvas, -1, 0);
        ctx.fillRect(width - 1, 0, 1, height);

        const spectrogram = spectrogramRef?.current;
        if (spectrogram?.data && spectrogram?.frameSize) {
          const { data, frameSize } = spectrogram;
          const frameOffset = Math.max(0, data.length - frameSize);
          const binHeight = height / frameSize;

          for (let i = 0; i < frameSize; i += 1) {
            const intensity = normalizeDb(data[frameOffset + i]);
            const hue = 220 - intensity * 160;
            const light = 18 + intensity * 55;
            ctx.fillStyle = `hsl(${hue} 90% ${light}%)`;
            const y = height - (i + 1) * binHeight;
            ctx.fillRect(width - 1, y, 1, binHeight + 0.6);
          }
        }

        rafRef.current = window.requestAnimationFrame(draw);
      };

      rafRef.current = window.requestAnimationFrame(draw);
    }

    return () => {
      if (rafRef.current) window.cancelAnimationFrame(rafRef.current);
      observer.disconnect();
    };
  }, [isActive, spectrogramRef]);

  return <canvas ref={canvasRef} className={styles['spectrogram-canvas']} />;
}
