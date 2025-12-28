import { useCallback, useState } from 'react';

import { useNavigate } from 'react-router-dom';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import './LandingPage.css';

const MODELS = [
  {
    id: 'bildklassifikation',
    mode: 'image',
    title: 'Bildklassifikation',
    description: 'Bilder automatisch klassifizieren.',
    imageSrc: '/assets/images/Bildklassifikation.png',
    imageAlt: 'Bildklassifikation Vorschau',
    tags: [{ label: 'Trainierbar', variant: 'primary' }, { label: 'BLE connect' }],
    hasHero: true,
  },
  {
    id: 'objektdetektion',
    mode: 'image',
    title: 'Objektdetektion',
    description: 'Objekte in Bildern erkennen.',
    imageSrc: '/assets/images/objektdetektion.png',
    imageAlt: 'Objektdetektion Vorschau',
    tags: [{ label: 'Trainierbar', variant: 'primary' }, { label: 'Live' }],
    hasHero: true,
  },
  {
    id: 'posenerkennung',
    mode: 'image',
    title: 'Posenerkennung',
    description: 'Körperhaltung und Gelenkpunkte erkennen.',
    imageSrc: '/assets/images/posenerkennung.png',
    imageAlt: 'Posenerkennung Vorschau',
    tags: [{ label: 'Live' }],
    hasHero: true,
  },
  {
    id: 'gesichtsmerkmale',
    mode: 'image',
    title: 'Gesichtsmerkmale',
    description: 'Gesichtsmerkmale analysieren.',
    imageSrc: '/assets/images/Gesichtsmerkmale.png',
    imageAlt: 'Gesichtsmerkmale Vorschau',
    tags: [{ label: 'Live' }],
    hasHero: true,
  },
  {
    id: 'gestenerkennung',
    mode: 'image',
    title: 'Gestenerkennung',
    description: 'Gesten per Kamera erkennen.',
    imageSrc: '/assets/images/Gestenerkennung.png',
    imageAlt: 'Gestenerkennung Vorschau',
    tags: [{ label: 'Live' }],
    hasHero: true,
  },
  {
    id: 'audioerkennung',
    mode: 'audio',
    title: 'Audioerkennung',
    description: 'Geräusche automatisch erkennen.',
    imageSrc: '/assets/images/audioerkennung.png',
    imageAlt: 'Audioerkennung Vorschau',
    tags: [{ label: 'Trainierbar', variant: 'primary' }],
    hasHero: true,
  },
];

export default function LandingPage() {
  const navigate = useNavigate();
  const [isNavOpen, setIsNavOpen] = useState(false);

  const handleMenuClick = useCallback(() => {
    setIsNavOpen((prev) => !prev);
  }, []);

  const handleSearchClick = useCallback(() => {
    console.log('[Landing] search');
  }, []);

  const handleCardActivate = useCallback(
    (model) => {
      if (model?.id === 'bildklassifikation') {
        navigate('/image-classification');
        return;
      }
      if (model?.id === 'posenerkennung') {
        navigate('/pose-estimation');
        return;
      }
      if (model?.id === 'gesichtsmerkmale') {
        navigate('/face-landmarks');
        return;
      }
      if (model?.id === 'audioerkennung') {
        navigate('/audio-classification');
        return;
      }

      console.log('[Landing] card', model);
    },
    [navigate],
  );

  const handleCardKeyDown = useCallback(
    (event, model) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      handleCardActivate(model);
    },
    [handleCardActivate],
  );

  return (
    <div id="landing-page" role="dialog" aria-label="Landing">
      <NavigationDrawer
        open={isNavOpen}
        onClose={() => setIsNavOpen(false)}
        drawerId="navigation-drawer"
      />
      <div className="landing-shell">
        <header className="landing-header">
          <button
            className="icon-button landing-menu nav-toggle"
            aria-label={isNavOpen ? 'Menü schließen' : 'Menü öffnen'}
            aria-controls="navigation-drawer"
            aria-expanded={isNavOpen}
            type="button"
            onClick={handleMenuClick}
          >
            <span className="icon-lines" />
          </button>
        </header>

        <div className="landing-identity" aria-label="AI Playground">
          <span className="identity-word">AI</span>
          <img className="identity-logo" src="/assets/images/cr.png" alt="Code Rizon Logo" />
          <span className="identity-word accent">Playground</span>
        </div>

        <button
          className="search-pill"
          type="button"
          data-mode="image"
          aria-label="Suche KI-Modell"
          onClick={handleSearchClick}
        >
          <span className="search-label">Suche KI-Modell</span>
          <span className="pill-arrow" aria-hidden="true">
            {'>'}
          </span>
        </button>

        <div className="model-marquee" role="list">
          <div className="model-track">
            {MODELS.map((model) => (
              <div
                key={model.id}
                className={`model-card${model.hasHero ? ' has-hero' : ''}`}
                data-mode={model.mode}
                role="listitem"
                tabIndex={0}
                onClick={() => handleCardActivate(model)}
                onKeyDown={(event) => handleCardKeyDown(event, model)}
              >
                {model.hasHero ? (
                  <div className="model-card-hero">
                    <img src={model.imageSrc} alt={model.imageAlt} loading="lazy" />
                  </div>
                ) : null}

                <div className="model-card-body">
                  <h3 className="model-card-title">{model.title}</h3>
                  <p className="model-card-description">{model.description}</p>

                  {model.tags?.length ? (
                    <div className="model-card-actions">
                      {model.tags.map((tag) => (
                        <span
                          key={`${model.id}:${tag.label}`}
                          className={`md-tag${tag.variant ? ` ${tag.variant}` : ''}`}
                        >
                          {tag.label}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
