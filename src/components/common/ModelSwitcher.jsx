import { useEffect, useMemo, useRef, useState, useId } from 'react';

import { useLocation, useNavigate } from 'react-router-dom';

import styles from './ModelSwitcher.module.css';

const MODELS = [
  {
    id: 'image-classification',
    label: 'Bildklassifikation',
    path: '/image-classification',
  },
  {
    id: 'visual-qa',
    label: 'Bildbeschreibung',
    path: '/visual-qa',
  },
  {
    id: 'image-segmentation',
    label: 'Bildsegmentierung',
    path: '/image-segmentation',
  },
  {
    id: 'object-detection',
    label: 'Objektdetektion',
    path: '/object-detection',
  },
  {
    id: 'pose-estimation',
    label: 'Posenerkennung',
    path: '/pose-estimation',
  },
  {
    id: 'face-landmarks',
    label: 'Gesichtsmerkmale',
    path: '/face-landmarks',
  },
  {
    id: 'hand-gestures',
    label: 'Gestenerkennung',
    path: '/gestenerkennung',
  },
  {
    id: 'audio-classification',
    label: 'Audioerkennung',
    path: '/audio-classification',
  },
];

export default function ModelSwitcher() {
  const navigate = useNavigate();
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const buttonRef = useRef(null);
  const menuRef = useRef(null);
  const menuId = useId();

  const activeModel = useMemo(
    () => MODELS.find((model) => model.path === location.pathname),
    [location.pathname],
  );

  const otherModels = useMemo(
    () => MODELS.filter((model) => model.path !== location.pathname),
    [location.pathname],
  );

  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (!isOpen) return undefined;

    const handlePointerDown = (event) => {
      const target = event.target;
      if (buttonRef.current?.contains(target)) return;
      if (menuRef.current?.contains(target)) return;
      setIsOpen(false);
    };

    const handleKeyDown = (event) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      setIsOpen(false);
      buttonRef.current?.focus();
    };

    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  const handleToggle = () => {
    setIsOpen((prev) => !prev);
  };

  const handleSelect = (path) => {
    navigate(path);
    setIsOpen(false);
  };

  const label = activeModel?.label ?? 'Modelle';

  return (
    <div className={styles.switcher}>
      <button
        ref={buttonRef}
        className={styles.switcherButton}
        type="button"
        aria-label="Modelle wechseln"
        aria-haspopup="menu"
        aria-expanded={isOpen}
        aria-controls={menuId}
        onClick={handleToggle}
      >
        <span className={styles.emphasis}>{label}</span>
        <span className={styles.caret} aria-hidden="true" />
      </button>
      {isOpen ? (
        <div
          id={menuId}
          className={styles.menu}
          role="menu"
          aria-label="Andere Modelle"
          ref={menuRef}
        >
          {otherModels.map((model) => (
            <button
              key={model.path}
              className={styles.menuItem}
              type="button"
              role="menuitem"
              onClick={() => handleSelect(model.path)}
            >
              {model.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
