import styles from '../ImageClassification.module.css';

export default function BluetoothButton({
  label,
  onClick,
  isConnected = false,
  deviceName,
}) {
  const resolvedLabel = label ?? (isConnected ? 'Trennen' : 'Verbinden');
  const accessibleLabel =
    isConnected && deviceName ? `${resolvedLabel} von ${deviceName}` : resolvedLabel;

  return (
    <button
      className={styles['bluetooth-button']}
      type="button"
      onClick={onClick}
      aria-pressed={isConnected}
      aria-label={accessibleLabel}
      title={isConnected && deviceName ? `Verbunden mit ${deviceName}` : undefined}
    >
      <span className={styles['bluetooth-content']}>
        <span className={styles['bluetooth-circle']} aria-hidden="true">
          <svg
            className={styles['bluetooth-icon']}
            viewBox="-2 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M6.5 6.5L17.5 17.5L12 23V1L17.5 6.5L6.5 17.5"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        <span className={styles['bluetooth-text']}>{resolvedLabel}</span>
      </span>
    </button>
  );
}
