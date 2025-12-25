import { useEffect } from 'react';

import { DEVICES } from '../../../hooks/bluetoothConfig.js';
import styles from '../ImageClassification.module.css';

export default function BluetoothModal({ isOpen, onClose, onSelectDevice }) {
  useEffect(() => {
    if (!isOpen) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose?.();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className={styles['bluetooth-modal-overlay']}
      role="dialog"
      aria-modal="true"
      aria-label="Bluetooth Geraete"
      onClick={() => onClose?.()}
    >
      <div
        className={`${styles.card} ${styles['bluetooth-modal']}`}
        role="document"
        onClick={(event) => event.stopPropagation()}
      >
        <div className={styles['bluetooth-modal-header']}>
          <div>
            <p className={styles['bluetooth-modal-title']}>Bluetooth Geraete</p>
            <p className={styles['bluetooth-modal-intro']}>
              Waehle das Geraet aus, mit dem du dich verbinden moechtest.
            </p>
          </div>
          <button
            className={styles['bluetooth-modal-close']}
            type="button"
            onClick={() => onClose?.()}
            aria-label="Modal schliessen"
          >
            x
          </button>
        </div>

        <div className={styles['bluetooth-device-grid']}>
          {DEVICES.map((device) => (
            <button
              key={device.id}
              className={styles['bluetooth-device-card']}
              type="button"
              onClick={() => onSelectDevice?.(device)}
            >
              <span className={styles['bluetooth-device-image']}>
                <img src={device.image} alt={device.name} loading="lazy" />
              </span>
              <span className={styles['bluetooth-device-name']}>{device.name}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
