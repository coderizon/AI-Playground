const DEFAULT_LINE_COLOR = '#00E5FF';
const DEFAULT_LABEL_COLOR = '#0f172a';
const DEFAULT_LABEL_BG = '#00E5FF';
const DEFAULT_LINE_WIDTH = 2;
const DEFAULT_FONT = '14px "Nunito", sans-serif';

function getTopCategory(detection) {
  const categories = detection?.categories;
  if (!Array.isArray(categories) || categories.length === 0) return null;

  return categories.reduce((best, current) => {
    const bestScore = typeof best?.score === 'number' ? best.score : 0;
    const currentScore = typeof current?.score === 'number' ? current.score : 0;
    return currentScore > bestScore ? current : best;
  }, categories[0]);
}

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(Math.max(value, min), max);
}

export function drawObjectDetections({
  ctx,
  canvas,
  video,
  width,
  height,
  detections,
  lineColor = DEFAULT_LINE_COLOR,
  labelColor = DEFAULT_LABEL_COLOR,
  labelBackground = DEFAULT_LABEL_BG,
  lineWidth = DEFAULT_LINE_WIDTH,
  font = DEFAULT_FONT,
  minScore = 0,
} = {}) {
  if (!ctx || !video) return;
  if (!Array.isArray(detections) || detections.length === 0) return;

  const cssWidth = width ?? canvas?.getBoundingClientRect?.().width ?? 0;
  const cssHeight = height ?? canvas?.getBoundingClientRect?.().height ?? 0;
  if (!cssWidth || !cssHeight) return;

  const videoWidth = video.videoWidth || cssWidth;
  const videoHeight = video.videoHeight || cssHeight;
  if (!videoWidth || !videoHeight) return;

  const scale = Math.max(cssWidth / videoWidth, cssHeight / videoHeight);
  const offsetX = (videoWidth * scale - cssWidth) / 2;
  const offsetY = (videoHeight * scale - cssHeight) / 2;

  ctx.strokeStyle = lineColor;
  ctx.lineWidth = lineWidth;
  ctx.font = font;
  ctx.textBaseline = 'top';

  detections.forEach((detection, index) => {
    const boundingBox = detection?.boundingBox;
    if (!boundingBox) return;

    const category = getTopCategory(detection);
    const label =
      category?.categoryName ??
      category?.displayName ??
      `Objekt ${index + 1}`;
    const score = typeof category?.score === 'number' ? category.score : 0;
    if (score < minScore) return;

    const x = boundingBox.originX * scale - offsetX;
    const y = boundingBox.originY * scale - offsetY;
    const boxWidth = boundingBox.width * scale;
    const boxHeight = boundingBox.height * scale;

    ctx.strokeRect(x, y, boxWidth, boxHeight);

    if (!label) return;

    const percent = Number.isFinite(score) ? ` ${Math.round(score * 100)}%` : '';
    const text = `${label}${percent}`;
    const metrics = ctx.measureText(text);
    const paddingX = 6;
    const paddingY = 4;
    const ascent = Number.isFinite(metrics.actualBoundingBoxAscent)
      ? metrics.actualBoundingBoxAscent
      : 12;
    const descent = Number.isFinite(metrics.actualBoundingBoxDescent)
      ? metrics.actualBoundingBoxDescent
      : 4;
    const textHeight = Math.max(14, ascent + descent);

    let textX = x;
    if (textX + metrics.width + paddingX * 2 > cssWidth) {
      textX = cssWidth - metrics.width - paddingX * 2;
    }
    textX = clamp(textX, 0, cssWidth);

    let textY = y - textHeight - paddingY * 2;
    if (textY < 0) {
      textY = y + 4;
    }
    textY = clamp(textY, 0, cssHeight);

    ctx.fillStyle = labelBackground;
    ctx.fillRect(textX, textY, metrics.width + paddingX * 2, textHeight + paddingY * 2);
    ctx.fillStyle = labelColor;
    ctx.fillText(text, textX + paddingX, textY + paddingY);
  });
}
