import { forwardRef, useCallback, useEffect, useRef } from 'react';

import { SwitchCamera } from 'lucide-react';

import styles from '../ImageClassification.module.css';

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

const StreamVideo = forwardRef(function StreamVideo({ stream, className, ...props }, forwardedRef) {
  const internalRef = useRef(null);

  const setRef = useCallback(
    (node) => {
      internalRef.current = node;

      if (typeof forwardedRef === 'function') {
        forwardedRef(node);
      } else if (forwardedRef) {
        forwardedRef.current = node;
      }
    },
    [forwardedRef],
  );

  useEffect(() => {
    if (!internalRef.current) return;
    internalRef.current.srcObject = stream ?? null;
  }, [stream]);

  return <video ref={setRef} className={className} autoPlay muted playsInline {...props} />;
});

const WebcamCapture = forwardRef(function WebcamCapture(
  { stream, isMirrored, showCameraSwitch, onToggleCamera, variant = 'capture', className },
  forwardedRef,
) {
  const containerClass = variant === 'preview' ? styles['video-shell'] : styles['capture-slot'];

  return (
    <div className={cx(containerClass, isMirrored && styles.mirrored, className)}>
      <StreamVideo ref={forwardedRef} stream={stream} />
      {showCameraSwitch ? (
        <button
          className={styles['ic-camera-switch']}
          type="button"
          onClick={onToggleCamera}
          aria-label="Kamera wechseln"
        >
          <SwitchCamera aria-hidden="true" />
        </button>
      ) : null}
    </div>
  );
});

export default WebcamCapture;
