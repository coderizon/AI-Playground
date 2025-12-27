import { useEffect, useRef, useState } from 'react';

import { Mic, MoreVertical } from 'lucide-react';

import SpectrogramCanvas from './SpectrogramCanvas.jsx';
import styles from '../../image-classification/ImageClassification.module.css';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function AudioClassCard({
  classNameValue,
  exampleCount,
  isCollecting,
  spectrogramRef,
  recordingProgress,
  recordingSecondsLeft,
  recordingDurationSeconds,
  canCollect,
  onClassNameChange,
  onClassNameFocus,
  onClassNameBlur,
  onCollect,
  onCollectStop,
  onClearExamples,
  canRemoveClass,
  onRemoveClass,
}) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    if (!isMenuOpen) return undefined;

    const handleOutsideClick = (event) => {
      if (!menuRef.current) return;
      if (!menuRef.current.contains(event.target)) {
        setIsMenuOpen(false);
      }
    };

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleOutsideClick);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isMenuOpen]);

  return (
    <div className={cx(styles.card, styles['class-card'], isCollecting && styles['audio-card-active'])}>
      <div className={styles['card-header']}>
        <input
          className={styles['class-name-input']}
          value={classNameValue}
          onChange={onClassNameChange}
          onFocus={onClassNameFocus}
          onBlur={onClassNameBlur}
        />
        <div className={styles['class-card-actions']}>
          <div className={styles['menu-wrapper']} ref={menuRef}>
            <button
              className={styles['menu-button']}
              type="button"
              aria-label="Klassenoptionen"
              aria-haspopup="menu"
              aria-expanded={isMenuOpen}
              onClick={() => setIsMenuOpen((prev) => !prev)}
            >
              <MoreVertical className={styles.dots} aria-hidden="true" />
            </button>
            {isMenuOpen ? (
              <div className={styles.menu} role="menu">
                <button
                  className={styles['menu-item']}
                  type="button"
                  role="menuitem"
                  disabled={exampleCount === 0}
                  onClick={() => {
                    setIsMenuOpen(false);
                    if (onClearExamples) onClearExamples();
                  }}
                >
                  Alle Beispiele entfernen
                </button>
                <button
                  className={styles['menu-item']}
                  type="button"
                  role="menuitem"
                  disabled={!canRemoveClass}
                  onClick={() => {
                    setIsMenuOpen(false);
                    if (onRemoveClass) onRemoveClass();
                  }}
                >
                  Klasse löschen
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      <div className={styles['audio-card-body']}>
        <div className={styles['spectrogram-frame']}>
          <SpectrogramCanvas spectrogramRef={spectrogramRef} isActive={isCollecting} />
        </div>

        <button
          className={cx(styles.primary, styles.block, styles['audio-record-button'])}
          type="button"
          disabled={!canCollect}
          onClick={onCollect}
          aria-pressed={isCollecting}
        >
          <Mic className={styles['audio-record-icon']} aria-hidden="true" />
          {isCollecting ? 'Aufnahme läuft…' : '10s aufnehmen'}
        </button>

        {isCollecting ? (
          <div className={styles['recording-progress']}>
            <div className={styles['recording-progress-meta']}>
              <span>Aufnahme läuft…</span>
              <span>{recordingSecondsLeft}s</span>
            </div>
            <div
              className={styles['recording-progress-bar']}
              style={{
                '--recording-ticks': Math.max(1, recordingDurationSeconds ?? 1),
              }}
            >
              <div
                className={styles['recording-progress-fill']}
                style={{ width: `${recordingProgress}%` }}
              />
            </div>
            <button
              className={styles['recording-stop']}
              type="button"
              onClick={onCollectStop}
            >
              Aufnahme stoppen
            </button>
          </div>
        ) : null}

        <div className={styles['count-row']}>
          <div className={cx(styles['count-box'], isCollecting && styles.recording)}>
            <div className={styles['count-meta']}>
              <span className={styles['count-label']}>Anzahl Beispiele</span>
            </div>
            <div className={styles['count-number']}>{exampleCount}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
