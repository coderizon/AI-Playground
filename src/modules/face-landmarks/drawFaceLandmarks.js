const DEFAULT_LINE_COLOR = 'rgba(56, 189, 248, 0.75)';
const DEFAULT_POINT_COLOR = 'rgba(34, 197, 94, 0.9)';
const DEFAULT_LINE_WIDTH = 1.2;
const DEFAULT_POINT_RADIUS = 1.5;

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function getConnectionPair(connection) {
  if (Array.isArray(connection)) return connection;
  if (connection && typeof connection === 'object') {
    return [connection.start, connection.end];
  }
  return [null, null];
}

export function drawFaceLandmarks({
  ctx,
  canvas,
  video,
  width,
  height,
  landmarks,
  connections = [],
  minConfidence = 0,
  lineColor = DEFAULT_LINE_COLOR,
  pointColor = DEFAULT_POINT_COLOR,
  lineWidth = DEFAULT_LINE_WIDTH,
  pointRadius = DEFAULT_POINT_RADIUS,
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

  const points = landmarks.map((landmark) => {
    const x = clamp01(landmark?.x ?? 0);
    const y = clamp01(landmark?.y ?? 0);
    const visibility = landmark?.visibility;
    const presence = landmark?.presence;
    const score =
      typeof visibility === 'number'
        ? visibility
        : typeof presence === 'number'
          ? presence
          : 1;

    return {
      x: x * videoWidth * scale - offsetX,
      y: y * videoHeight * scale - offsetY,
      score,
    };
  });

  if (connections?.length) {
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = lineWidth;

    ctx.beginPath();
    for (const connection of connections) {
      const [start, end] = getConnectionPair(connection);
      if (!Number.isFinite(start) || !Number.isFinite(end)) continue;

      const startPoint = points[start];
      const endPoint = points[end];
      if (!startPoint || !endPoint) continue;
      if (startPoint.score < minConfidence || endPoint.score < minConfidence) continue;

      ctx.moveTo(startPoint.x, startPoint.y);
      ctx.lineTo(endPoint.x, endPoint.y);
    }
    ctx.stroke();
  }

  ctx.fillStyle = pointColor;
  for (const point of points) {
    if (!point || point.score < minConfidence) continue;
    ctx.beginPath();
    ctx.arc(point.x, point.y, pointRadius, 0, Math.PI * 2);
    ctx.fill();
  }
}
