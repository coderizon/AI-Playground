import { useEffect, useState } from 'react';

import { DEVICES } from '../../../hooks/bluetoothConfig.js';
import styles from '../ImageClassification.module.css';

const formatOffsetValue = (value) => (typeof value === 'number' ? `${value}px` : value);

const getIllustrationStyle = (imageOffset) => {
  if (!imageOffset) return undefined;
  const style = {};
  if (imageOffset.right != null) {
    style['--bt-illustration-right-adjust'] = formatOffsetValue(imageOffset.right);
  }
  if (imageOffset.bottom != null) {
    style['--bt-illustration-bottom-adjust'] = formatOffsetValue(imageOffset.bottom);
  }
  return Object.keys(style).length ? style : undefined;
};

export default function BluetoothModal({ isOpen, onClose, onSelectDevice }) {
  const [isRendered, setIsRendered] = useState(isOpen);
  const [isClosing, setIsClosing] = useState(false);

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
    }, 180);

    return () => window.clearTimeout(timeoutId);
  }, [isOpen, isRendered]);

  useEffect(() => {
    if (!isOpen) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose?.();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isRendered) return null;

  const modalState = isClosing ? 'closing' : 'open';

  return (
    <div
      className={styles['bluetooth-modal-overlay']}
      role="dialog"
      aria-modal="true"
      aria-label="Bluetooth Geraete"
      onClick={() => onClose?.()}
      data-state={modalState}
    >
      <div
        className={styles['bluetooth-modal']}
        role="document"
        onClick={(event) => event.stopPropagation()}
        data-state={modalState}
      >
        <div className={styles['bluetooth-modal-surface']}>
          <div className={styles['bluetooth-modal-header']}>
            <div>
              <p className={styles['bluetooth-modal-title']}>Verbinden mit</p>
              <p className={styles['bluetooth-modal-intro']}>Wähle ein Geraet aus der Liste.</p>
            </div>
            <button
              className={styles['bluetooth-modal-close']}
              type="button"
              onClick={() => onClose?.()}
              aria-label="Modal schliessen"
            >
              <span className={styles['bluetooth-close-icon']} aria-hidden="true">
                x
              </span>
            </button>
          </div>

          <div className={styles['bluetooth-device-list']}>
            {DEVICES.map((device) => {
              const illustrationStyle = getIllustrationStyle(device.imageOffset);
              return (
                <button
                  key={device.id}
                  className={styles['bluetooth-device-card']}
                  type="button"
                  onClick={() => onSelectDevice?.(device)}
                  style={illustrationStyle}
                  data-device-id={device.id}
                >
                  <div className={styles['bluetooth-device-info']}>
                    <span className={styles['bluetooth-device-title']}>{device.name}</span>
                  </div>
                  <span className={styles['bluetooth-device-arrow']} aria-hidden="true">
                    &gt;
                  </span>
                  <span className={styles['bluetooth-device-illustration']} aria-hidden="true">
                    <img
                      className={styles['bluetooth-device-image']}
                      src={device.image}
                      alt=""
                      loading="lazy"
                    />
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
