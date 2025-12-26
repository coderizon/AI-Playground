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
  {
    stream,
    isMirrored,
    showCameraSwitch,
    onToggleCamera,
    variant = 'capture',
    className,
    poseOverlay,
  },
  forwardedRef,
) {
  const containerClass = variant === 'preview' ? styles['video-shell'] : styles['capture-slot'];
  const videoRef = useRef(null);
  const overlayRef = useRef(null);

  const setVideoRef = useCallback(
    (node) => {
      videoRef.current = node;

      if (typeof forwardedRef === 'function') {
        forwardedRef(node);
      } else if (forwardedRef) {
        forwardedRef.current = node;
      }
    },
    [forwardedRef],
  );

  useEffect(() => {
    const canvas = overlayRef.current;
    const video = videoRef.current;

    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width: cssWidth, height: cssHeight } = canvas.getBoundingClientRect();
    if (!cssWidth || !cssHeight) return;

    const devicePixelRatio = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;
    const canvasWidth = Math.round(cssWidth * devicePixelRatio);
    const canvasHeight = Math.round(cssHeight * devicePixelRatio);

    if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
    }

    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    if (!poseOverlay?.keypoints?.length) return;

    const videoWidth = video.videoWidth || cssWidth;
    const videoHeight = video.videoHeight || cssHeight;
    if (!videoWidth || !videoHeight) return;

    // Map video-space keypoints into the square viewport with object-fit: cover.
    const scale = Math.max(cssWidth / videoWidth, cssHeight / videoHeight);
    const offsetX = (videoWidth * scale - cssWidth) / 2;
    const offsetY = (videoHeight * scale - cssHeight) / 2;

    const minConfidence = poseOverlay.minConfidence ?? 0.3;

    const toCanvasPoint = (keypoint) => ({
      x: keypoint.x * scale - offsetX,
      y: keypoint.y * scale - offsetY,
      score: keypoint.score ?? 1,
    });

    const points = poseOverlay.keypoints.map(toCanvasPoint);

    if (poseOverlay.adjacentPairs?.length) {
      ctx.strokeStyle = poseOverlay.lineColor ?? 'rgba(56, 189, 248, 0.85)';
      ctx.lineWidth = 2;

      ctx.beginPath();
      for (const [start, end] of poseOverlay.adjacentPairs) {
        const startPoint = points[start];
        const endPoint = points[end];

        if (!startPoint || !endPoint) continue;
        if (startPoint.score < minConfidence || endPoint.score < minConfidence) continue;

        ctx.moveTo(startPoint.x, startPoint.y);
        ctx.lineTo(endPoint.x, endPoint.y);
      }
      ctx.stroke();
    }

    ctx.fillStyle = poseOverlay.pointColor ?? 'rgba(34, 197, 94, 0.9)';
    for (const point of points) {
      if (!point || point.score < minConfidence) continue;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [poseOverlay]);

  return (
    <div className={cx(containerClass, isMirrored && styles.mirrored, className)}>
      <StreamVideo ref={setVideoRef} stream={stream} />
      {poseOverlay ? (
        <canvas ref={overlayRef} className={styles['pose-overlay']} aria-hidden="true" />
      ) : null}
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
