import { useCallback, useState } from 'react';

import ImageClassification from './ImageClassification.jsx';
import LandingPage from './LandingPage.jsx';

export default function App() {
  const [activeView, setActiveView] = useState('landing');

  const handleSelectModel = useCallback((model) => {
    if (model?.id === 'bildklassifikation') {
      setActiveView('bildklassifikation');
      return;
    }

    console.log('[App] Unimplemented model', model);
  }, []);

  if (activeView === 'bildklassifikation') {
    return <ImageClassification />;
  }

  return <LandingPage onSelectModel={handleSelectModel} />;
}
