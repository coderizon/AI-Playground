const DEFAULT_PALM_COLOR = 'rgba(226, 232, 240, 0.85)';
const DEFAULT_LINE_WIDTH = 2.3;
const DEFAULT_POINT_RADIUS = 3.4;
const DEFAULT_FINGER_COLORS = {
  thumb: 'rgba(250, 204, 21, 0.9)',
  index: 'rgba(34, 197, 94, 0.9)',
  middle: 'rgba(34, 211, 238, 0.9)',
  ring: 'rgba(59, 130, 246, 0.9)',
  pinky: 'rgba(168, 85, 247, 0.9)',
};
const FINGER_GROUP_BY_INDEX = [
  null,
  'thumb',
  'thumb',
  'thumb',
  'thumb',
  'index',
  'index',
  'index',
  'index',
  'middle',
  'middle',
  'middle',
  'middle',
  'ring',
  'ring',
  'ring',
  'ring',
  'pinky',
  'pinky',
  'pinky',
  'pinky',
];

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

function getFingerGroup(index) {
  if (!Number.isFinite(index)) return null;
  return FINGER_GROUP_BY_INDEX[index] ?? null;
}

export function drawHandLandmarks({
  ctx,
  canvas,
  video,
  width,
  height,
  hands,
  connections = [],
  minConfidence = 0,
  fingerColors,
  palmColor = DEFAULT_PALM_COLOR,
  lineWidth = DEFAULT_LINE_WIDTH,
  pointRadius = DEFAULT_POINT_RADIUS,
} = {}) {
  if (!ctx || !video) return;
  if (!Array.isArray(hands) || hands.length === 0) return;

  const palette = { ...DEFAULT_FINGER_COLORS, ...(fingerColors ?? {}) };

  const cssWidth = width ?? canvas?.getBoundingClientRect?.().width ?? 0;
  const cssHeight = height ?? canvas?.getBoundingClientRect?.().height ?? 0;
  if (!cssWidth || !cssHeight) return;

  const videoWidth = video.videoWidth || cssWidth;
  const videoHeight = video.videoHeight || cssHeight;
  if (!videoWidth || !videoHeight) return;

  const scale = Math.max(cssWidth / videoWidth, cssHeight / videoHeight);
  const offsetX = (videoWidth * scale - cssWidth) / 2;
  const offsetY = (videoHeight * scale - cssHeight) / 2;

  for (const hand of hands) {
    const landmarks = Array.isArray(hand?.landmarks) ? hand.landmarks : hand;
    if (!Array.isArray(landmarks) || landmarks.length === 0) continue;

    const points = landmarks.map((landmark, index) => {
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
        index,
      };
    });

    if (connections?.length) {
      ctx.lineWidth = lineWidth;

      for (const connection of connections) {
        const [start, end] = getConnectionPair(connection);
        if (!Number.isFinite(start) || !Number.isFinite(end)) continue;

        const startPoint = points[start];
        const endPoint = points[end];
        if (!startPoint || !endPoint) continue;
        if (startPoint.score < minConfidence || endPoint.score < minConfidence) continue;

        const startGroup = getFingerGroup(start);
        const endGroup = getFingerGroup(end);
        const group = startGroup && startGroup === endGroup ? startGroup : null;
        ctx.strokeStyle = group ? palette[group] ?? palmColor : palmColor;

        ctx.beginPath();
        ctx.moveTo(startPoint.x, startPoint.y);
        ctx.lineTo(endPoint.x, endPoint.y);
        ctx.stroke();
      }
    }

    for (const point of points) {
      if (!point || point.score < minConfidence) continue;
      const group = getFingerGroup(point.index);
      ctx.fillStyle = group ? palette[group] ?? palmColor : palmColor;
      ctx.beginPath();
      ctx.arc(point.x, point.y, pointRadius, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
