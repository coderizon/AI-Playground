import styles from './MaterialSlider.module.css';

function clampValue(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export default function MaterialSlider({ value, min = 0, max = 100, step = 1, onChange, ...props }) {
  const safeMin = Number.isFinite(min) ? min : 0;
  const safeMax = Number.isFinite(max) ? max : safeMin + 1;
  const clampedValue = clampValue(Number(value), safeMin, safeMax);
  const range = safeMax - safeMin;
  const progress = range > 0 ? ((clampedValue - safeMin) / range) * 100 : 0;

  const handleChange = (event) => {
    const nextValue = Number(event.target.value);
    if (!Number.isFinite(nextValue)) return;
    onChange?.(nextValue, event);
  };

  return (
    <input
      className={styles.slider}
      type="range"
      min={safeMin}
      max={safeMax}
      step={step}
      value={clampedValue}
      onChange={handleChange}
      style={{
        background: `linear-gradient(to right, #6750A4 ${progress}%, #E7E0EC ${progress}%)`,
      }}
      {...props}
    />
  );
}
