import { useState, useEffect, useCallback, useRef } from 'react';
import { TeachableLLM, selectBackend } from '@genai-fi/nanogpt';
import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import './LLMTraining.css';

const DEFAULT_TEXT = `Habe nun, ach! Philosophie,
Juristerei und Medizin,
Und leider auch Theologie
Durchaus studiert, mit heißem Bemühn.
Da steh ich nun, ich armer Tor!
Und bin so klug als wie zuvor;
Heiße Magister, heiße Doktor gar
Und ziehe schon an die zehen Jahr
Herauf, herab und quer und krumm
Meine Schüler an der Nase herum-
Und sehe, daß wir nichts wissen können!
Das will mir schier das Herz verbrennen.`;

export default function LLMTraining() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [status, setStatus] = useState('Bereit');
  const [progress, setProgress] = useState(0);
  const [loss, setLoss] = useState(null);
  const [generatedText, setGeneratedText] = useState('');
  const [trainingText, setTrainingText] = useState(DEFAULT_TEXT);
  const [isTraining, setIsTraining] = useState(false);

  const modelRef = useRef(null);

  useEffect(() => {
    const initBackend = async () => {
      try {
        await selectBackend('webgpu');
        setStatus('Backend: WebGPU initialisiert');
      } catch (e) {
        console.warn('WebGPU nicht verfügbar, nutze Fallback', e);
        await selectBackend('cpu');
        setStatus('Backend: CPU (langsam) initialisiert');
      }
    };
    initBackend();
  }, []);

  const handleTrain = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setStatus('Bereite Training vor...');
    setProgress(0);
    setGeneratedText('');

    try {
      // Configuration for a tiny demo model
      const config = {
        vocabSize: 100,
        blockSize: 32,
        nLayer: 2,
        nHead: 2,
        nEmbed: 32,
        dropout: 0.0,
      };

      const model = TeachableLLM.create('char', config);
      modelRef.current = model;

      model.on('trainStep', (step, prog) => {
        setLoss(step.loss.toFixed(4));
        setProgress(prog.progress * 100);
        setStatus(`Training... Step ${step.step}/${prog.totalSteps}`);
      });

      const data = [trainingText];

      await model.train(data, {
        batchSize: 4,
        learningRate: 1e-3,
        maxSteps: 200,
        logInterval: 10,
      });

      setStatus('Training abgeschlossen!');
      generateSample(model);
    } catch (error) {
      console.error(error);
      setStatus(`Fehler: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const generateSample = async (modelInstance = modelRef.current) => {
    if (!modelInstance) return;
    setStatus('Generiere Text...');

    try {
      const seed = trainingText.slice(0, 5);
      const output = await modelInstance.generateText(seed, {
        maxLength: 100,
        temperature: 0.8,
        topP: 0.9,
      });
      setGeneratedText(output);
      setStatus('Fertig.');
    } catch (e) {
      setStatus('Fehler bei Generierung');
    }
  };

  return (
    <div className="llm-training-page">
      <NavigationDrawer
        open={isNavOpen}
        onClose={() => setIsNavOpen(false)}
        drawerId="llm-nav-drawer"
      />

      <header className="page-header">
        <button
          className="icon-button"
          onClick={() => setIsNavOpen(true)}
          aria-label="Menü öffnen"
        >
          <span className="icon-lines" />
        </button>
        <h1>LLM Training from Scratch</h1>
      </header>

      <main className="training-container">
        <div className="card control-panel">
          <h2>Trainingsdaten</h2>
          <p>Gib hier Text ein, den das KI-Modell "lernen" soll:</p>
          <textarea
            value={trainingText}
            onChange={(e) => setTrainingText(e.target.value)}
            rows={10}
            className="text-input"
          />
          <div className="actions">
            <button
              className="btn-primary"
              onClick={handleTrain}
              disabled={isTraining}
            >
              {isTraining ? 'Trainiere...' : 'Modell Trainieren'}
            </button>
          </div>
        </div>

        <div className="card visualization-panel">
          <h2>Status & Ergebnis</h2>
          <div className="status-bar">
            <strong>Status:</strong> {status}
          </div>

          {loss && (
            <div className="metric">
              <span>Aktueller Loss (Fehler):</span>
              <span className="loss-value">{loss}</span>
            </div>
          )}

          <div className="progress-container">
            <div
              className="progress-bar"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="output-area">
            <h3>Generierter Text:</h3>
            <pre>{generatedText || 'Noch kein Modell trainiert.'}</pre>
          </div>

          <button
            className="btn-secondary"
            onClick={() => generateSample()}
            disabled={isTraining || !modelRef.current}
          >
            Erneut Generieren
          </button>
        </div>
      </main>
    </div>
  );
}
