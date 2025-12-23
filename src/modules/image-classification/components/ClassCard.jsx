import { useCallback, useEffect, useRef, useState } from 'react';

import { MoreVertical, Video, X } from 'lucide-react';

import WebcamCapture from './WebcamCapture.jsx';
import styles from '../ImageClassification.module.css';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function ClassCard({
  classNameValue,
  exampleCount,
  isCollecting,
  stream,
  showCameraSwitch,
  isMirrored,
  onToggleCamera,
  onClassNameChange,
  onClassNameFocus,
  onClassNameBlur,
  onCollectStart,
  onCollectStop,
  canCollect,
  isWebcamEnabled,
  captureRef,
  onToggleWebcam,
  onClearExamples,
}) {
  const [particles, setParticles] = useState([]);
  const [bumpAnimation, setBumpAnimation] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const bumpTimeoutRef = useRef(null);
  const particleTimeoutsRef = useRef(new Set());
  const previousExampleCountRef = useRef(exampleCount);
  const menuRef = useRef(null);

  const addParticle = useCallback(() => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const scale = 0.8 + Math.random() * 0.4;
    const duration = 0.8 + Math.random() * 0.4;
    setParticles((prev) => [...prev, { id, scale, duration }]);

    const timeoutId = window.setTimeout(() => {
      setParticles((prev) => prev.filter((particle) => particle.id !== id));
      particleTimeoutsRef.current.delete(timeoutId);
    }, duration * 1000);

    particleTimeoutsRef.current.add(timeoutId);
  }, []);

  const triggerIncrementEffect = useCallback(() => {
    setBumpAnimation(true);
    if (bumpTimeoutRef.current) window.clearTimeout(bumpTimeoutRef.current);
    bumpTimeoutRef.current = window.setTimeout(() => setBumpAnimation(false), 100);
    addParticle();
  }, [addParticle]);

  useEffect(() => {
    if (exampleCount > previousExampleCountRef.current) {
      triggerIncrementEffect();
    }
    previousExampleCountRef.current = exampleCount;
  }, [exampleCount, triggerIncrementEffect]);

  useEffect(() => {
    return () => {
      if (bumpTimeoutRef.current) window.clearTimeout(bumpTimeoutRef.current);
      for (const timeoutId of particleTimeoutsRef.current) {
        window.clearTimeout(timeoutId);
      }
      particleTimeoutsRef.current.clear();
    };
  }, []);

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
    <div className={cx(styles.card, styles['class-card'])}>
      <div className={styles['card-header']}>
        <input
          className={styles['class-name-input']}
          value={classNameValue}
          onChange={onClassNameChange}
          onFocus={onClassNameFocus}
          onBlur={onClassNameBlur}
        />
        <div className={styles['class-card-actions']}>
          <button
            className={styles['ic-webcam-toggle']}
            type="button"
            aria-label={isWebcamEnabled ? 'Webcam schließen' : 'Webcam öffnen'}
            aria-pressed={isWebcamEnabled}
            onClick={onToggleWebcam}
          >
            <Video className={styles['ic-webcam-icon']} aria-hidden="true" />
            {isWebcamEnabled ? <X className={styles['ic-webcam-x']} aria-hidden="true" /> : null}
          </button>
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
                  onClick={() => {
                    setIsMenuOpen(false);
                    if (onClearExamples) onClearExamples();
                  }}
                >
                  Alle Beispiele löschen
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {isWebcamEnabled ? (
        <div className={styles['webcam-panel']}>
          <WebcamCapture
            ref={captureRef}
            stream={stream}
            showCameraSwitch={showCameraSwitch}
            isMirrored={isMirrored}
            onToggleCamera={onToggleCamera}
          />

          <button
            className={cx(styles.dataCollector, styles.primary, styles.block)}
            type="button"
            disabled={!canCollect}
            aria-pressed={isCollecting}
            onPointerDown={(event) => {
              if (!canCollect) return;
              event.currentTarget.setPointerCapture(event.pointerId);
              onCollectStart();
            }}
            onPointerUp={onCollectStop}
            onPointerCancel={onCollectStop}
            onLostPointerCapture={onCollectStop}
          >
            Zum Aufnehmen halten
          </button>
        </div>
      ) : null}

      <div className={styles['count-row']}>
        <div className={cx(styles['count-box'], isCollecting && styles.recording)}>
          <div className={styles['count-bubbles']} aria-hidden="true">
            {particles.map((particle) => (
              <div
                key={particle.id}
                className={styles['count-bubble']}
                style={{
                  animationDuration: `${particle.duration}s`,
                  '--scale': `${particle.scale}`,
                }}
              >
                +PNG
              </div>
            ))}
          </div>

          <div className={styles['count-meta']}>
            <span className={styles['count-label']}>Anzahl Beispiele</span>
          </div>
          <div className={cx(styles['count-number'], bumpAnimation && styles.bump)}>
            {exampleCount}
          </div>
        </div>
      </div>
    </div>
  );
}
