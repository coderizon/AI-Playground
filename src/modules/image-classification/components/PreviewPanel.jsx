import WebcamCapture from './WebcamCapture.jsx';
import styles from '../ImageClassification.module.css';

const CLASS_COLORS = [
  '#3f73ff', // blue
  '#22c55e', // green
  '#f59e0b', // amber
  '#a855f7', // violet
  '#ef4444', // red
  '#14b8a6', // teal
  '#e11d48', // rose
  '#6366f1', // indigo
  '#84cc16', // lime
  '#0ea5e9', // sky
  '#f97316', // orange
  '#8b5cf6', // purple
];

function getClassColor(index) {
  if (index < CLASS_COLORS.length) return CLASS_COLORS[index];

  const hue = Math.round((index * 137.508) % 360);
  return `hsl(${hue} 82% 55%)`;
}

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function PreviewPanel({
  stream,
  classes,
  probabilities,
  showCameraSwitch,
  isMirrored,
  onToggleCamera,
  captureRef,
}) {
  return (
    <div className={cx(styles.card, styles['preview-card'])}>
      <div className={styles['preview-body']}>
        <WebcamCapture
          ref={captureRef}
          stream={stream}
          showCameraSwitch={showCameraSwitch}
          isMirrored={isMirrored}
          onToggleCamera={onToggleCamera}
          variant="preview"
        />

        <div className={styles['preview-output']}>
          <div className={styles['preview-output-header']}>
            <span>Ausgabe</span>
          </div>

          <div className={styles.probabilityList}>
            {classes.map((cls, index) => {
              const probability = probabilities[index] ?? 0;
              const percent = Math.round(probability * 100);
              const color = getClassColor(index);

              return (
                <div key={cls.id} className={styles['probability-row']}>
                  <div className={styles['probability-row-header']}>
                    <span>{cls.name}</span>
                    <span>{percent}%</span>
                  </div>
                  <div
                    className={styles['probability-bar']}
                    role="meter"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={percent}
                  >
                    <div
                      className={styles['probability-bar-fill']}
                      style={{ width: `${percent}%`, backgroundColor: color }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
