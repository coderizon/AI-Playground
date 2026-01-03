const IRIS_COLOR = 'rgba(34, 197, 94, 0.8)'; // Green
const NOSE_COLOR = 'rgba(239, 68, 68, 0.9)'; // Red
const TEXT_COLOR = 'rgba(255, 255, 255, 0.95)';
const IRIS_POINT_RADIUS = 2;
const NOSE_POINT_RADIUS = 4;

// Iris landmark indices (left eye)
const LEFT_IRIS_CENTER = 468;
const LEFT_IRIS_LEFT = 469;
const LEFT_IRIS_RIGHT = 471;

// Iris landmark indices (right eye)
const RIGHT_IRIS_CENTER = 473;
const RIGHT_IRIS_LEFT = 474;
const RIGHT_IRIS_RIGHT = 476;

// Nose tip landmark index
const NOSE_TIP = 1;

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

/**
 * Draw head tracking visualization
 * Shows iris circle and nose point for distance and position tracking
 */
export function drawHeadTracking({
  ctx,
  canvas,
  video,
  width,
  height,
  landmarks,
  distanceCM = null,
} = {}) {
  if (!ctx || !video) return;
  if (!Array.isArray(landmarks) || landmarks.length === 0) return;

  const cssWidth = width ?? canvas?.getBoundingClientRect?.().width ?? 0;
  const cssHeight = height ?? canvas?.getBoundingClientRect?.().height ?? 0;
  if (!cssWidth || !cssHeight) return;

  const videoWidth = video.videoWidth || cssWidth;
  const videoHeight = video.videoHeight || cssHeight;
  if (!videoWidth || !videoHeight) return;

  const scale = Math.max(cssWidth / videoWidth, cssHeight / videoHeight);
  const offsetX = (videoWidth * scale - cssWidth) / 2;
  const offsetY = (videoHeight * scale - cssHeight) / 2;

  // Transform landmark to canvas coordinates
  const transformPoint = (landmark) => {
    if (!landmark) return null;
    const x = clamp01(landmark.x ?? 0);
    const y = clamp01(landmark.y ?? 0);
    return {
      x: x * videoWidth * scale - offsetX,
      y: y * videoHeight * scale - offsetY,
    };
  };

  // Draw iris circle (left eye)
  const leftIrisCenter = transformPoint(landmarks[LEFT_IRIS_CENTER]);
  const leftIrisLeft = transformPoint(landmarks[LEFT_IRIS_LEFT]);
  const leftIrisRight = transformPoint(landmarks[LEFT_IRIS_RIGHT]);

  if (leftIrisCenter && leftIrisLeft && leftIrisRight) {
    // Calculate iris radius from left-right points
    const leftIrisRadius = Math.abs(leftIrisRight.x - leftIrisLeft.x) / 2;

    // Draw iris circle
    ctx.strokeStyle = IRIS_COLOR;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(leftIrisCenter.x, leftIrisCenter.y, leftIrisRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Draw center point
    ctx.fillStyle = IRIS_COLOR;
    ctx.beginPath();
    ctx.arc(leftIrisCenter.x, leftIrisCenter.y, IRIS_POINT_RADIUS, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw iris circle (right eye)
  const rightIrisCenter = transformPoint(landmarks[RIGHT_IRIS_CENTER]);
  const rightIrisLeft = transformPoint(landmarks[RIGHT_IRIS_LEFT]);
  const rightIrisRight = transformPoint(landmarks[RIGHT_IRIS_RIGHT]);

  if (rightIrisCenter && rightIrisLeft && rightIrisRight) {
    // Calculate iris radius from left-right points
    const rightIrisRadius = Math.abs(rightIrisRight.x - rightIrisLeft.x) / 2;

    // Draw iris circle
    ctx.strokeStyle = IRIS_COLOR;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(rightIrisCenter.x, rightIrisCenter.y, rightIrisRadius, 0, Math.PI * 2);
    ctx.stroke();

    // Draw center point
    ctx.fillStyle = IRIS_COLOR;
    ctx.beginPath();
    ctx.arc(rightIrisCenter.x, rightIrisCenter.y, IRIS_POINT_RADIUS, 0, Math.PI * 2);
    ctx.fill();
  }
}
