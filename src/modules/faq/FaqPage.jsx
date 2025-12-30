import { useCallback, useState } from 'react';

import {
  ChevronDown,
  ClipboardCheck,
  CloudOff,
  Gauge,
  Globe,
  Layers,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  TrendingUp,
  Users,
  VideoOff,
  WandSparkles,
} from 'lucide-react';

import NavigationDrawer from '../../components/common/NavigationDrawer.jsx';
import styles from './FaqPage.module.css';

const FAQ_CATEGORIES = [
  {
    id: 'allgemein',
    label: 'Allgemein',
    items: [
      {
        icon: Sparkles,
        question: 'Was ist AI Playground?',
        answer: [
          'AI Playground ist eine webbasierte Umgebung, in der du KI-Modelle direkt im Browser ausprobieren kannst.',
          'Du findest Module für Bild, Audio und Pose. Trainierbare Module lassen dich eigene Klassen erstellen.',
        ],
      },
      {
        icon: Users,
        question: 'Für wen ist AI Playground gedacht?',
        answer: [
          'Für Einsteiger, Lehrende und Teams, die KI schnell und praktisch testen wollen.',
          'Vorkenntnisse sind hilfreich, aber nicht notwendig.',
        ],
      },
      {
        icon: Layers,
        question: 'Welche Module gibt es?',
        answer: [
          'Bildklassifikation, Bildbeschreibung, Objekterkennung, Pose-, Gesichts- und Gestenerkennung sowie Audioerkennung.',
          'Einige Module sind trainierbar, andere nutzen feste Modelle für sofortige Ergebnisse.',
        ],
      },
      {
        icon: WandSparkles,
        question: 'Wie starte ich?',
        answer: [
          'Öffne die Home-Seite und wähle ein Modul aus.',
          'Erlaube Kamera oder Mikrofon, falls das Modul es benötigt.',
        ],
      },
      {
        icon: Globe,
        question: 'Welche Browser werden empfohlen?',
        answer: [
          'Am stabilsten laufen aktuelle Versionen von Chrome, Edge oder Firefox.',
          'Andere Browser können funktionieren, liefern aber je nach Gerät weniger Leistung.',
        ],
      },
      {
        icon: VideoOff,
        question: 'Warum sehe ich kein Live-Bild?',
        answer: [
          'Prüfe die Berechtigungen für Kamera oder Mikrofon und ob eine andere App sie blockiert.',
          'Ein Neuladen der Seite behebt häufig den Zustand.',
        ],
      },
    ],
  },
  {
    id: 'training',
    label: 'Training',
    items: [
      {
        icon: ClipboardCheck,
        question: 'Wie sammle ich gute Trainingsdaten?',
        answer: [
          'Nimm viele Beispiele pro Klasse auf und variiere Licht, Hintergrund und Perspektive.',
          'Achte darauf, dass die Klassen ähnlich viele Beispiele enthalten.',
        ],
      },
      {
        icon: TrendingUp,
        question: 'Warum sind Ergebnisse ungenau?',
        answer: [
          'Typische Ursachen sind zu wenige oder einseitige Beispiele.',
          'Teste dein Modell mit neuen Beispielen und sammle gezielt Daten für schwierige Fälle.',
        ],
      },
      {
        icon: SlidersHorizontal,
        question: 'Kann ich Trainingsparameter anpassen?',
        answer: [
          'In trainierbaren Modulen kannst du Parameter wie Epochen, Batch-Größe oder Lernrate anpassen.',
          'Starte mit den Standardwerten und ändere sie schrittweise.',
        ],
      },
    ],
  },
  {
    id: 'datenschutz',
    label: 'Datenschutz',
    items: [
      {
        icon: ShieldCheck,
        question: 'Werden meine Daten hochgeladen?',
        answer: [
          'AI Playground verarbeitet Kamera- und Audiodaten lokal im Browser.',
          'Es findet kein automatisches Hochladen deiner Beispiele statt.',
        ],
      },
      {
        icon: RefreshCw,
        question: 'Was passiert beim Neuladen der Seite?',
        answer: [
          'Trainingsdaten liegen im Arbeitsspeicher des Browsers.',
          'Wenn du neu lädst oder den Tab schließt, musst du das Training erneut starten.',
        ],
      },
    ],
  },
  {
    id: 'hilfe',
    label: 'Hilfe',
    items: [
      {
        icon: CloudOff,
        question: 'Das Modell lädt nicht oder bleibt hängen.',
        answer: [
          'Prüfe deine Internetverbindung und lade die Seite neu.',
          'Wenn das Problem bleibt, hilft oft ein Browser-Neustart.',
        ],
      },
      {
        icon: Gauge,
        question: 'Mein Gerät ist sehr langsam.',
        answer: [
          'Schließe andere Tabs und Apps, damit mehr Leistung frei ist.',
          'Reduziere die Anzahl der Klassen oder der Beispiele pro Klasse.',
        ],
      },
    ],
  },
];

export default function FaqPage() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [activeCategoryId, setActiveCategoryId] = useState(FAQ_CATEGORIES[0]?.id);

  const handleMenuClick = useCallback(() => {
    setIsNavOpen((prev) => !prev);
  }, []);

  const activeCategory =
    FAQ_CATEGORIES.find((category) => category.id === activeCategoryId) ?? FAQ_CATEGORIES[0];

  return (
    <div className={styles['faq-page']}>
      <NavigationDrawer
        open={isNavOpen}
        onClose={() => setIsNavOpen(false)}
        drawerId="navigation-drawer"
      />

      <div className={styles['faq-shell']}>
        <header className={styles['faq-topbar']}>
          <button
            className={styles['faq-menu']}
            type="button"
            aria-label={isNavOpen ? 'Menü schließen' : 'Menü öffnen'}
            aria-controls="navigation-drawer"
            aria-expanded={isNavOpen}
            onClick={handleMenuClick}
          >
            <span className={styles['faq-menu-lines']} />
          </button>
          <div className={styles['faq-title']}>FAQs</div>
        </header>

        <main className={styles['faq-main']}>
          <section className={styles['faq-hero']}>
            <h1 className={styles['faq-heading']}>Häufige Fragen</h1>
            <p className={styles['faq-subtitle']}>
              Die wichtigsten Antworten zu AI Playground auf einen Blick. Wenn du etwas nicht
              findest, sprich uns gern an.
            </p>
          </section>

          <div className={styles['faq-tabs']}>
            {FAQ_CATEGORIES.map((category) => {
              const isActive = category.id === activeCategoryId;
              return (
                <button
                  key={category.id}
                  className={`${styles['faq-tab']} ${isActive ? styles['faq-tab-active'] : ''}`}
                  type="button"
                  aria-pressed={isActive}
                  onClick={() => setActiveCategoryId(category.id)}
                >
                  {category.label}
                </button>
              );
            })}
          </div>

          <section className={styles['faq-list']}>
            {activeCategory?.items.map((item, itemIndex) => {
              const Icon = item.icon;
              return (
                <details key={item.question} className={styles['faq-item']} defaultOpen={itemIndex === 0}>
                  <summary className={styles['faq-item-summary']}>
                    <span className={styles['faq-item-icon']} aria-hidden="true">
                      <Icon />
                    </span>
                    <span className={styles['faq-item-question']}>{item.question}</span>
                    <ChevronDown className={styles['faq-item-chevron']} aria-hidden="true" />
                  </summary>
                  <div className={styles['faq-answer']}>
                    {item.answer.map((paragraph, paragraphIndex) => (
                      <p key={`${item.question}-${paragraphIndex}`}>{paragraph}</p>
                    ))}
                  </div>
                </details>
              );
            })}
          </section>
        </main>
      </div>
    </div>
  );
}
