import { Navigate, Route, Routes } from 'react-router-dom';

import ImageClassification from './ImageClassification.jsx';
import LandingPage from './LandingPage.jsx';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/image-classification" element={<ImageClassification />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
