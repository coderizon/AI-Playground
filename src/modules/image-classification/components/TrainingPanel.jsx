import styles from '../ImageClassification.module.css';
import NeuralNetworkAnimation from './NeuralNetworkAnimation.jsx';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function TrainingPanel({
  epochs,
  batchSize,
  learningRate,
  onEpochsChange,
  onBatchSizeChange,
  onLearningRateChange,
  onTrain,
  canTrain,
  isTraining,
  trainingPercent,
}) {
  return (
    <div className={cx(styles.card, styles['training-card'])}>
      <div className={cx(styles['card-header'], styles.spaced)}>
        <h3>Training</h3>
        <button className={styles.primary} type="button" onClick={onTrain} disabled={!canTrain}>
          Modell trainieren
        </button>
      </div>

      <div className={cx(styles['training-progress'], !isTraining && styles.hidden)}>
        <div className={styles['training-progress-meta']}>
          <span>Training läuft...</span>
          <span>{trainingPercent}%</span>
        </div>
        <div className={styles['training-progress-bar']}>
          <div
            className={styles['training-progress-fill']}
            style={{ width: `${trainingPercent}%` }}
          />
        </div>
      </div>
      {isTraining ? (
        <NeuralNetworkAnimation className={styles['training-animation']} />
      ) : null}

      <details open className={styles.accordion}>
        <summary>Erweitert</summary>
        <div className={styles['form-grid']}>
          <label>
            <span>Epochen</span>
            <input type="number" min={1} step={1} value={epochs} onChange={onEpochsChange} />
          </label>
          <label>
            <span>Batchgröße</span>
            <input type="number" min={1} step={1} value={batchSize} onChange={onBatchSizeChange} />
          </label>
          <label>
            <span>Lernrate</span>
            <input
              type="number"
              min={0.000001}
              step={0.0001}
              value={learningRate}
              onChange={onLearningRateChange}
            />
          </label>
        </div>
      </details>
    </div>
  );
}
