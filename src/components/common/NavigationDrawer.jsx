import { useCallback, useEffect, useRef } from 'react';

import { Home, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

import './NavigationDrawer.css';

const FOCUSABLE_SELECTOR =
  'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';

function getFocusableElements(container) {
  if (!container) return [];

  const nodes = Array.from(container.querySelectorAll(FOCUSABLE_SELECTOR));
  return nodes.filter((node) => {
    if (node.hasAttribute('disabled')) return false;
    if (node.getAttribute('aria-hidden') === 'true') return false;

    const rect = node.getBoundingClientRect();
    return rect.width > 0 || rect.height > 0;
  });
}

export default function NavigationDrawer({ open, onClose, drawerId = 'navigation-drawer' }) {
  const navigate = useNavigate();
  const panelRef = useRef(null);
  const previouslyFocusedRef = useRef(null);

  const closeDrawer = useCallback(() => {
    if (typeof onClose === 'function') onClose();
  }, [onClose]);

  const handleHome = useCallback(() => {
    navigate('/');
    closeDrawer();
  }, [closeDrawer, navigate]);

  useEffect(() => {
    if (!open) return undefined;

    previouslyFocusedRef.current = document.activeElement;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    const focusTimer = window.setTimeout(() => {
      const focusables = getFocusableElements(panelRef.current);
      const target = focusables[0] ?? panelRef.current;
      if (target && typeof target.focus === 'function') target.focus();
    }, 0);

    return () => {
      window.clearTimeout(focusTimer);
      document.body.style.overflow = previousOverflow;

      const previous = previouslyFocusedRef.current;
      if (previous && typeof previous.focus === 'function') {
        previous.focus();
      }
    };
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        closeDrawer();
        return;
      }

      if (event.key !== 'Tab') return;

      const focusables = getFocusableElements(panelRef.current);
      if (!focusables.length) return;

      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement;

      if (event.shiftKey) {
        if (active === first || !panelRef.current?.contains(active)) {
          event.preventDefault();
          last.focus();
        }
        return;
      }

      if (active === last || !panelRef.current?.contains(active)) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [closeDrawer, open]);

  if (!open) return null;

  return (
    <div className="nav-drawer" aria-hidden={!open}>
      <button className="nav-drawer-overlay" type="button" aria-label="Menü schließen" onClick={closeDrawer} />

      <aside
        id={drawerId}
        ref={panelRef}
        className="nav-drawer-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Navigation"
        tabIndex={-1}
      >
        <div className="nav-drawer-header">
          <div className="nav-drawer-title">Menü</div>
          <button className="nav-drawer-close" type="button" aria-label="Schließen" onClick={closeDrawer}>
            <X aria-hidden="true" />
          </button>
        </div>

        <nav className="nav-drawer-nav" aria-label="Navigation">
          <button className="nav-drawer-item" type="button" onClick={handleHome}>
            <Home className="nav-drawer-item-icon" aria-hidden="true" />
            <span>Home</span>
          </button>
        </nav>
      </aside>
    </div>
  );
}

