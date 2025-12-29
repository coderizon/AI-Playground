import { useEffect, useRef } from 'react';
import styles from '../../image-classification/ImageClassification.module.css';

// "Teachable Machine"-aehnliche Farbpalette (dunkel -> lila -> rot -> gelb)
const COLOR_PALETTE = {
  0: [0, 0, 0], // Stille (schwarz)
  10: [75, 0, 159], // Tiefes lila
  20: [104, 0, 251], // Lila
  30: [131, 0, 255], // Helllila
  40: [155, 18, 157], // Magenta
  50: [175, 37, 0], // Rot
  60: [191, 59, 0], // Rot-orange
  70: [206, 88, 0], // Orange
  80: [223, 132, 0], // Hellorange
  90: [240, 188, 0], // Gold
  100: [255, 252, 0], // Grelles gelb
};

/**
 * Berechnet die Farbe basierend auf der Lautstaerke (0-255).
 * Nutzt lineare Interpolation fuer weiche Uebergaenge.
 */
function getHeatmapColor(value) {
  const percent = (value / 255) * 100;
  const floored = 10 * Math.floor(percent / 10);

  if (floored >= 100) {
    const [r, g, b] = COLOR_PALETTE[100];
    return `rgb(${r}, ${g}, ${b})`;
  }

  const distFromFloor = percent - floored;
  const factor = distFromFloor / 10;

  const startColor = COLOR_PALETTE[floored] || [0, 0, 0];
  const endColor = COLOR_PALETTE[floored + 10] || COLOR_PALETTE[100];

  const r = Math.round(startColor[0] + factor * (endColor[0] - startColor[0]));
  const g = Math.round(startColor[1] + factor * (endColor[1] - startColor[1]));
  const b = Math.round(startColor[2] + factor * (endColor[2] - startColor[2]));

  return `rgb(${r}, ${g}, ${b})`;
}

export default function SpectrogramCanvas({ isActive }) {
  const canvasRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(null);

  // Audio-Initialisierung: unabhaengig von der KI.
  useEffect(() => {
    if (!isActive) return undefined;

    const initAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        });

        streamRef.current = stream;

        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioContext();
        audioContextRef.current = ctx;

        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.0;
        analyserRef.current = analyser;

        const source = ctx.createMediaStreamSource(stream);
        source.connect(analyser);
      } catch (err) {
        console.error('Visualisierungs-Fehler:', err);
      }
    };

    initAudio();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, [isActive]);

  // Zeichen-Loop (60 FPS)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return undefined;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      ctx.fillStyle = 'rgb(0, 0, 0)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    const draw = () => {
      if (!isActive || !analyserRef.current) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const width = canvas.width;
      const height = canvas.height;
      const analyser = analyserRef.current;

      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteFrequencyData(dataArray);

      const scrollSpeed = 2;
      ctx.drawImage(canvas, -scrollSpeed, 0);

      const relevantFrequencies = Math.floor(bufferLength * 0.7);

      for (let i = 0; i < relevantFrequencies; i += 1) {
        const value = dataArray[i];

        if (value > 5) {
          const color = getHeatmapColor(value);
          ctx.fillStyle = color;

          const y = height - Math.round((i / relevantFrequencies) * height);
          const pointHeight = Math.ceil(height / relevantFrequencies) || 1;

          ctx.fillRect(
            width - scrollSpeed,
            y - pointHeight,
            scrollSpeed,
            pointHeight,
          );
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(rafRef.current);
      observer.disconnect();
    };
  }, [isActive]);

  return <canvas ref={canvasRef} className={styles['spectrogram-canvas']} />;
}
