import styles from '../../image-classification/ImageClassification.module.css';

export default function LossSparkline({ values }) {
  if (!values || values.length < 2) return null;

  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;

  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg className={styles['loss-chart']} viewBox="0 0 100 100" preserveAspectRatio="none">
      <polyline className={styles['loss-line']} fill="none" points={points} />
    </svg>
  );
}
