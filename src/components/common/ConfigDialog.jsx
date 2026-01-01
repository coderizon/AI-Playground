import { useEffect, useState } from 'react';

import styles from './ConfigDialog.module.css';

const DEFAULT_CONFIG = {
  maxTokens: 2048,
  temperature: 0.8,
  topP: 1,
  topK: 40,
  repetitionPenalty: 1.1,
  presencePenalty: 0,
  frequencyPenalty: 0,
  seed: null,
  accelerator: 'GPU',
};

const MAX_TOKENS_RANGE = {
  min: 32,
  max: 2048,
  step: 32,
};

const REPETITION_PENALTY_RANGE = {
  min: 1,
  max: 2,
  step: 0.1,
};

const PRESENCE_PENALTY_RANGE = {
  min: -2,
  max: 2,
  step: 0.1,
};

const FREQUENCY_PENALTY_RANGE = {
  min: -2,
  max: 2,
  step: 0.1,
};

const SEED_RANGE = {
  min: 0,
  max: 10000,
  step: 1,
};

const SLIDER_CONFIGS = [
  {
    key: 'maxTokens',
    label: 'Max tokens',
    ...MAX_TOKENS_RANGE,
    integer: true,
  },
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
  {
    key: 'repetitionPenalty',
    label: 'Repetition penalty',
    ...REPETITION_PENALTY_RANGE,
  },
  {
    key: 'presencePenalty',
    label: 'Presence penalty',
    ...PRESENCE_PENALTY_RANGE,
  },
  {
    key: 'frequencyPenalty',
    label: 'Frequency penalty',
    ...FREQUENCY_PENALTY_RANGE,
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

function readOptionalNumber(value) {
  if (value === '' || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeConfig(config) {
  const safeConfig = config ?? DEFAULT_CONFIG;
  const maxTokens = clampNumber(
    readNumber(safeConfig.maxTokens, DEFAULT_CONFIG.maxTokens),
    MAX_TOKENS_RANGE.min,
    MAX_TOKENS_RANGE.max,
  );
  const temperature = clampNumber(
    readNumber(safeConfig.temperature, DEFAULT_CONFIG.temperature),
    0,
    2,
  );
  const topP = clampNumber(readNumber(safeConfig.topP, DEFAULT_CONFIG.topP), 0, 1);
  const topK = clampNumber(readNumber(safeConfig.topK, DEFAULT_CONFIG.topK), 1, 100);
  const repetitionPenalty = clampNumber(
    readNumber(safeConfig.repetitionPenalty, DEFAULT_CONFIG.repetitionPenalty),
    REPETITION_PENALTY_RANGE.min,
    REPETITION_PENALTY_RANGE.max,
  );
  const presencePenalty = clampNumber(
    readNumber(safeConfig.presencePenalty, DEFAULT_CONFIG.presencePenalty),
    PRESENCE_PENALTY_RANGE.min,
    PRESENCE_PENALTY_RANGE.max,
  );
  const frequencyPenalty = clampNumber(
    readNumber(safeConfig.frequencyPenalty, DEFAULT_CONFIG.frequencyPenalty),
    FREQUENCY_PENALTY_RANGE.min,
    FREQUENCY_PENALTY_RANGE.max,
  );
  const seedValue = readOptionalNumber(safeConfig.seed);
  const seed =
    seedValue === null
      ? null
      : Math.round(clampNumber(seedValue, SEED_RANGE.min, SEED_RANGE.max));

  return {
    maxTokens: Math.round(maxTokens),
    temperature,
    topP,
    topK: Math.round(topK),
    repetitionPenalty,
    presencePenalty,
    frequencyPenalty,
    seed,
    accelerator: safeConfig.accelerator === 'CPU' ? 'CPU' : 'GPU',
  };
}

function getDecimals(step) {
  const text = String(step);
  const parts = text.split('.');
  return parts.length > 1 ? parts[1].length : 0;
}

function getPercent(value, min, max) {
  if (!Number.isFinite(value)) return 0;
  if (max <= min) return 0;
  const raw = ((value - min) / (max - min)) * 100;
  return clampNumber(raw, 0, 100);
}

function SliderRow({ label, value, min, max, step, onChange }) {
  const decimals = getDecimals(step);
  const displayValue = Number.isFinite(value) ? value.toFixed(decimals) : '';
  const safeValue = Number.isFinite(value) ? value : min;
  const percent = getPercent(safeValue, min, max);
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
          style={{
            background: `linear-gradient(to right, var(--slider-accent) 0%, var(--slider-accent) ${percent}%, #e7e0ec ${percent}%, #e7e0ec 100%)`,
          }}
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

function SeedRow({ value, min, max, step, onChange }) {
  const safeValue = Number.isFinite(value) ? value : min;
  const percent = getPercent(safeValue, min, max);
  const displayValue = Number.isFinite(value) ? String(Math.round(value)) : '';
  const isAuto = value === null || value === undefined || value === '';

  const handleSliderChange = (event) => {
    const nextValue = Number(event.target.value);
    if (!Number.isFinite(nextValue)) return;
    const clamped = clampNumber(nextValue, min, max);
    onChange(Math.round(clamped));
  };

  const handleInputChange = (event) => {
    const rawValue = event.target.value;
    if (rawValue === '') {
      onChange(null);
      return;
    }
    const nextValue = Number(rawValue);
    if (!Number.isFinite(nextValue)) return;
    const clamped = clampNumber(nextValue, min, max);
    onChange(Math.round(clamped));
  };

  return (
    <div className={styles.row}>
      <span className={styles.label}>Seed</span>
      <div className={styles.sliderRow}>
        <input
          className={styles.slider}
          type="range"
          min={min}
          max={max}
          step={step}
          value={safeValue}
          onChange={handleSliderChange}
          aria-label="Seed slider"
          style={{
            background: `linear-gradient(to right, var(--slider-accent) 0%, var(--slider-accent) ${percent}%, #e7e0ec ${percent}%, #e7e0ec 100%)`,
          }}
        />
        <input
          className={styles.numberInput}
          type="number"
          min={min}
          max={max}
          step={step}
          value={displayValue}
          onChange={handleInputChange}
          placeholder="auto"
          aria-label="Seed value"
        />
        <button
          className={styles.seedButton}
          type="button"
          onClick={() => onChange(null)}
          disabled={isAuto}
        >
          Auto
        </button>
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
      aria-label="Modelleinstellungen"
      onClick={() => onClose?.()}
    >
      <div
        className={styles.card}
        data-state={dialogState}
        role="document"
        onClick={(event) => event.stopPropagation()}
      >
        <header className={styles.header}>
          <h2 className={styles.title}>Modelleinstellungen</h2>
        </header>

        <div className={styles.body}>
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
          <SeedRow
            value={localConfig.seed}
            min={SEED_RANGE.min}
            max={SEED_RANGE.max}
            step={SEED_RANGE.step}
            onChange={(nextValue) =>
              setLocalConfig((prev) => ({
                ...prev,
                seed: nextValue,
              }))
            }
          />

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
