import { Navigate, Route, Routes } from 'react-router-dom';

import ImageClassification from './modules/image-classification/ImageClassification.jsx';
import FaceLandmarks from './modules/face-landmarks/FaceLandmarks.jsx';
import LandingPage from './modules/landing/LandingPage.jsx';
import PoseEstimation from './modules/pose-estimation/PoseEstimation.jsx';
import AudioClassification from './modules/audio-classification/AudioClassification.jsx';
import ObjectDetection from './modules/object-detection/ObjectDetection.jsx';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/image-classification" element={<ImageClassification />} />
      <Route path="/object-detection" element={<ObjectDetection />} />
      <Route path="/audio-classification" element={<AudioClassification />} />
      <Route path="/pose-estimation" element={<PoseEstimation />} />
      <Route path="/face-landmarks" element={<FaceLandmarks />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
