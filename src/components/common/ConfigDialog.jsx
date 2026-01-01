import { useEffect, useState } from 'react';

import styles from './ConfigDialog.module.css';

const DEFAULT_CONFIG = {
  temperature: 0.8,
  topP: 1,
  topK: 40,
  accelerator: 'GPU',
};

const SLIDER_CONFIGS = [
  {
    key: 'topK',
    label: 'TopK',
    min: 1,
    max: 100,
    step: 1,
    integer: true,
  },
  {
    key: 'topP',
    label: 'TopP',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'temperature',
    label: 'Temperature',
    min: 0,
    max: 2,
    step: 0.1,
  },
];

const ACCELERATOR_OPTIONS = ['GPU', 'CPU'];

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function readNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeConfig(config) {
  const safeConfig = config ?? DEFAULT_CONFIG;
  const temperature = clampNumber(
    readNumber(safeConfig.temperature, DEFAULT_CONFIG.temperature),
    0,
    2,
  );
  const topP = clampNumber(readNumber(safeConfig.topP, DEFAULT_CONFIG.topP), 0, 1);
  const topK = clampNumber(readNumber(safeConfig.topK, DEFAULT_CONFIG.topK), 1, 100);

  return {
    temperature,
    topP,
    topK: Math.round(topK),
    accelerator: safeConfig.accelerator === 'CPU' ? 'CPU' : 'GPU',
  };
}

function getDecimals(step) {
  const text = String(step);
  const parts = text.split('.');
  return parts.length > 1 ? parts[1].length : 0;
}

function SliderRow({ label, value, min, max, step, onChange }) {
  const decimals = getDecimals(step);
  const displayValue = Number.isFinite(value) ? value.toFixed(decimals) : '';
  const handleChange = (event) => {
    const nextValue = Number(event.target.value);
    if (!Number.isFinite(nextValue)) return;
    onChange(nextValue);
  };

  return (
    <div className={styles.row}>
      <span className={styles.label}>{label}</span>
      <div className={styles.sliderRow}>
        <input
          className={styles.slider}
          type="range"
          min={min}
          max={max}
          step={step}
          value={Number.isFinite(value) ? value : min}
          onChange={handleChange}
          aria-label={`${label} slider`}
        />
        <input
          className={styles.numberInput}
          type="number"
          min={min}
          max={max}
          step={step}
          value={displayValue}
          onChange={handleChange}
          aria-label={`${label} value`}
        />
      </div>
    </div>
  );
}

export default function ConfigDialog({ isOpen, config, onApply, onClose }) {
  const [isRendered, setIsRendered] = useState(isOpen);
  const [isClosing, setIsClosing] = useState(false);
  const [localConfig, setLocalConfig] = useState(() => normalizeConfig(config));

  useEffect(() => {
    if (isOpen) {
      setIsRendered(true);
      setIsClosing(false);
      return undefined;
    }

    if (!isRendered) return undefined;

    setIsClosing(true);
    const timeoutId = window.setTimeout(() => {
      setIsRendered(false);
      setIsClosing(false);
    }, 160);

    return () => window.clearTimeout(timeoutId);
  }, [isOpen, isRendered]);

  useEffect(() => {
    if (!isOpen) return;
    setLocalConfig(normalizeConfig(config));
  }, [config, isOpen]);

  useEffect(() => {
    if (!isOpen) return undefined;

    const handleKeyDown = (event) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onClose?.();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isRendered) return null;

  const dialogState = isClosing ? 'closing' : 'open';

  const updateField = (key, { min, max, integer }) => (nextValue) => {
    const clamped = clampNumber(nextValue, min, max);
    const normalized = integer ? Math.round(clamped) : clamped;
    setLocalConfig((prev) => ({
      ...prev,
      [key]: normalized,
    }));
  };

  const handleApply = () => {
    onApply?.(normalizeConfig(localConfig));
  };

  return (
    <div
      className={styles.overlay}
      data-state={dialogState}
      role="dialog"
      aria-modal="true"
      aria-label="Model configs"
      onClick={() => onClose?.()}
    >
      <div
        className={styles.card}
        data-state={dialogState}
        role="document"
        onClick={(event) => event.stopPropagation()}
      >
        <header className={styles.header}>
          <h2 className={styles.title}>Model configs</h2>
        </header>

        <div className={styles.body}>
          <div className={styles.row}>
            <span className={styles.label}>Max tokens</span>
            <span className={styles.readonlyValue}>2048</span>
          </div>
          {SLIDER_CONFIGS.map((slider) => (
            <SliderRow
              key={slider.key}
              label={slider.label}
              value={localConfig[slider.key]}
              min={slider.min}
              max={slider.max}
              step={slider.step}
              onChange={updateField(slider.key, slider)}
            />
          ))}

          <div className={styles.row}>
            <span className={styles.label}>Accelerator</span>
            <div className={styles.segmented} role="group" aria-label="Accelerator">
              {ACCELERATOR_OPTIONS.map((option) => (
                <button
                  key={option}
                  className={cx(
                    styles.segmentButton,
                    localConfig.accelerator === option && styles.segmentButtonActive,
                  )}
                  type="button"
                  aria-pressed={localConfig.accelerator === option}
                  onClick={() =>
                    setLocalConfig((prev) => ({
                      ...prev,
                      accelerator: option,
                    }))
                  }
                >
                  {localConfig.accelerator === option ? (
                    <span className={styles.segmentCheck} aria-hidden="true">
                      ✓
                    </span>
                  ) : null}
                  {option}
                </button>
              ))}
            </div>
          </div>
        </div>

        <footer className={styles.footer}>
          <button className={styles.textButton} type="button" onClick={() => onClose?.()}>
            Cancel
          </button>
          <button className={styles.primaryButton} type="button" onClick={handleApply}>
            OK
          </button>
        </footer>
      </div>
    </div>
  );
}
