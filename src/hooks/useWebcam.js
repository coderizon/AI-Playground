import { useCallback, useEffect, useRef, useState } from 'react';

const DEFAULT_FACING_MODE = 'user';
const CAMERA_IDEAL_SIZE = 720;

function buildVideoConstraints(facingMode, idealSize) {
  const supported = navigator?.mediaDevices?.getSupportedConstraints?.() ?? {};
  return {
    facingMode,
    ...(supported.width ? { width: { ideal: idealSize } } : {}),
    ...(supported.height ? { height: { ideal: idealSize } } : {}),
    ...(supported.aspectRatio ? { aspectRatio: { ideal: 1 } } : {}),
    ...(supported.frameRate ? { frameRate: { ideal: 30, max: 30 } } : {}),
  };
}

export function useWebcam({ enabled = true, idealSize = CAMERA_IDEAL_SIZE } = {}) {
  const [status, setStatus] = useState(enabled ? 'loading' : 'disabled');
  const [facingMode, setFacingMode] = useState(DEFAULT_FACING_MODE);
  const [canSwitchCamera, setCanSwitchCamera] = useState(false);
  const streamRef = useRef(null);

  const stopStream = useCallback(() => {
    const stream = streamRef.current;
    streamRef.current = null;
    if (stream) stream.getTracks().forEach((track) => track.stop());
  }, []);

  const toggleFacingMode = useCallback(() => {
    setFacingMode((prev) => (prev === 'user' ? 'environment' : 'user'));
  }, []);

  useEffect(() => {
    if (!enabled) {
      stopStream();
      setStatus('disabled');
      return undefined;
    }

    let cancelled = false;

    async function startWebcam() {
      if (!navigator?.mediaDevices?.getUserMedia) {
        setStatus('error');
        return;
      }

      setStatus('loading');

      try {
        let stream = null;

        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: buildVideoConstraints(facingMode, idealSize),
          });
        } catch (error) {
          if (error?.name === 'OverconstrainedError' || error?.name === 'ConstraintNotSatisfiedError') {
            stream = await navigator.mediaDevices.getUserMedia({
              audio: false,
              video: { facingMode },
            });
          } else {
            throw error;
          }
        }

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        setStatus('ready');
      } catch (error) {
        console.error(error);
        if (!cancelled && facingMode === 'environment') {
          setFacingMode('user');
          return;
        }
        setStatus('error');
      }
    }

    startWebcam();

    return () => {
      cancelled = true;
      stopStream();
    };
  }, [enabled, facingMode, idealSize, stopStream]);

  useEffect(() => {
    if (status !== 'ready') {
      setCanSwitchCamera(false);
      return undefined;
    }

    let cancelled = false;

    const update = async () => {
      if (!navigator?.mediaDevices?.enumerateDevices) {
        setCanSwitchCamera(false);
        return;
      }

      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (cancelled) return;

        const videoInputs = devices.filter((device) => device.kind === 'videoinput');
        const uniqueIds = new Set(videoInputs.map((device) => device.deviceId).filter(Boolean));
        const count = uniqueIds.size || videoInputs.length;
        setCanSwitchCamera(count > 1);
      } catch (error) {
        console.error(error);
        if (!cancelled) setCanSwitchCamera(false);
      }
    };

    update();

    navigator.mediaDevices?.addEventListener?.('devicechange', update);

    return () => {
      cancelled = true;
      navigator.mediaDevices?.removeEventListener?.('devicechange', update);
    };
  }, [status]);

  return {
    status,
    stream: streamRef.current,
    facingMode,
    isMirrored: facingMode === 'user',
    canSwitchCamera,
    toggleFacingMode,
    setFacingMode,
  };
}
