import { useEffect, useRef } from 'react';

const LAYER_CONFIG = [2, 4, 4, 2];
const COLORS = {
  background: '#ffffff',
  neuron: '#3f73ff',
  connection: '#e0e0e0',
  signal: '#4facfe',
};

class Neuron {
  constructor(x, y, layerIndex, radius, floatRange) {
    this.x = x;
    this.y = y;
    this.baseX = x;
    this.baseY = y;
    this.layerIndex = layerIndex;
    this.radius = radius;
    this.activation = 0;
    this.phaseX = Math.random() * Math.PI * 2;
    this.phaseY = Math.random() * Math.PI * 2;
    this.floatSpeed = 0.001 + Math.random() * 0.001;
    this.floatRange = floatRange;
  }

  update(time) {
    this.x = this.baseX + Math.sin(time * this.floatSpeed + this.phaseX) * this.floatRange;
    this.y = this.baseY + Math.cos(time * this.floatSpeed + this.phaseY) * this.floatRange;
  }

  draw(ctx) {
    if (this.activation > 0) {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius + this.activation * 10, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(63, 115, 255, ${this.activation * 0.3})`;
      ctx.fill();
      this.activation = Math.max(0, this.activation - 0.05);
    }

    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.neuron;
    ctx.shadowBlur = Math.max(6, this.radius * 1.4);
    ctx.shadowOffsetY = 2;
    ctx.shadowColor = 'rgba(63, 115, 255, 0.45)';
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.shadowOffsetY = 0;

    ctx.strokeStyle = 'rgba(63, 115, 255, 0)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

class Signal {
  constructor(startNeuron, endNeuron) {
    this.startNeuron = startNeuron;
    this.endNeuron = endNeuron;
    this.progress = 0;
    this.speed = 0.02 + Math.random() * 0.02;
    this.active = true;
  }

  update(triggerNext) {
    this.progress += this.speed;
    if (this.progress >= 1) {
      this.progress = 1;
      this.active = false;
      this.endNeuron.activation = 1.0;
      triggerNext(this.endNeuron);
    }
  }

  draw(ctx) {
    const currentX =
      this.startNeuron.x + (this.endNeuron.x - this.startNeuron.x) * this.progress;
    const currentY =
      this.startNeuron.y + (this.endNeuron.y - this.startNeuron.y) * this.progress;

    ctx.beginPath();
    ctx.arc(currentX, currentY, 1.8, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.signal;
    ctx.shadowBlur = 10;
    ctx.shadowColor = COLORS.signal;
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}

function cx(...classes) {
  return classes.filter(Boolean).join(' ');
}

export default function NeuralNetworkAnimation({ className }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const animationRef = useRef(null);
  const intervalRef = useRef(null);
  const resizeObserverRef = useRef(null);

  const gameState = useRef({
    neurons: [],
    connections: [],
    signals: [],
    size: { width: 0, height: 0 },
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return undefined;

    const ctx = canvas.getContext('2d');
    if (!ctx) return undefined;
    ctxRef.current = ctx;

    const triggerSignalsFrom = (startNeuron) => {
      const outgoing = gameState.current.connections.filter((c) => c.start === startNeuron);
      for (const conn of outgoing) {
        gameState.current.signals.push(new Signal(conn.start, conn.end));
      }
    };

    const initNetwork = () => {
      const { width, height } = gameState.current.size;
      if (!width || !height) return;

      gameState.current.neurons = [];
      gameState.current.connections = [];
      gameState.current.signals = [];

      const paddingX = Math.min(36, width * 0.18);
      const layerGap = (width - paddingX * 2) / (LAYER_CONFIG.length - 1);
      const startX = paddingX;
      const radius = Math.max(2.5, Math.min(7, height * 0.05));
      const floatRange = Math.max(1.2, Math.min(4.5, radius * 0.6));

      LAYER_CONFIG.forEach((neuronCount, layerIndex) => {
        const verticalStep = height / (neuronCount + 1);
        for (let i = 0; i < neuronCount; i += 1) {
          const x = startX + layerIndex * layerGap;
          const y = verticalStep * (i + 1);
          gameState.current.neurons.push(new Neuron(x, y, layerIndex, radius, floatRange));
        }
      });

      for (let i = 0; i < LAYER_CONFIG.length - 1; i += 1) {
        const currentLayer = gameState.current.neurons.filter((n) => n.layerIndex === i);
        const nextLayer = gameState.current.neurons.filter((n) => n.layerIndex === i + 1);

        for (const start of currentLayer) {
          for (const end of nextLayer) {
            gameState.current.connections.push({ start, end });
          }
        }
      }
    };

    const handleResize = () => {
      const rect = container.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));
      const dpr = window.devicePixelRatio || 1;

      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      gameState.current.size = { width, height };
      initNetwork();
    };

    if (typeof ResizeObserver !== 'undefined') {
      resizeObserverRef.current = new ResizeObserver(handleResize);
      resizeObserverRef.current.observe(container);
    } else {
      window.addEventListener('resize', handleResize);
    }

    handleResize();

    intervalRef.current = window.setInterval(() => {
      const inputNeurons = gameState.current.neurons.filter((n) => n.layerIndex === 0);
      if (!inputNeurons.length) return;
      const randomInput = inputNeurons[Math.floor(Math.random() * inputNeurons.length)];
      randomInput.activation = 1.0;
      triggerSignalsFrom(randomInput);
    }, 800);

    const animate = () => {
      const { width, height } = gameState.current.size;
      const context = ctxRef.current;
      if (context && width && height) {
        context.clearRect(0, 0, width, height);
        context.fillStyle = COLORS.background;
        context.fillRect(0, 0, width, height);

        const now = performance.now();
        for (const neuron of gameState.current.neurons) {
          neuron.update(now);
        }

        context.lineWidth = 0.6;
        context.shadowBlur = 6;
        context.shadowOffsetY = 2;
        context.shadowColor = 'rgba(63, 115, 255, 0.25)';
        for (const conn of gameState.current.connections) {
          context.beginPath();
          context.moveTo(conn.start.x, conn.start.y);
          context.lineTo(conn.end.x, conn.end.y);
          context.strokeStyle = COLORS.connection;
          context.stroke();
        }
        context.shadowBlur = 0;
        context.shadowOffsetY = 0;

        for (let i = gameState.current.signals.length - 1; i >= 0; i -= 1) {
          const signal = gameState.current.signals[i];
          signal.update(triggerSignalsFrom);
          signal.draw(context);
          if (!signal.active) {
            gameState.current.signals.splice(i, 1);
          }
        }

        for (const neuron of gameState.current.neurons) {
          neuron.draw(context);
        }
      }

      animationRef.current = window.requestAnimationFrame(animate);
    };

    animationRef.current = window.requestAnimationFrame(animate);

    return () => {
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
        resizeObserverRef.current = null;
      } else {
        window.removeEventListener('resize', handleResize);
      }

      if (animationRef.current) {
        window.cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);

  return (
    <div className={cx(className)} ref={containerRef} aria-hidden="true">
      <canvas ref={canvasRef} />
    </div>
  );
}
