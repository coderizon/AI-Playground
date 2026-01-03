import { Suspense, lazy } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';

import LandingPage from './modules/landing/LandingPage.jsx';

const ImageClassification = lazy(() =>
  import('./modules/image-classification/ImageClassification.jsx'),
);
const FaceLandmarks = lazy(() =>
  import('./modules/face-landmarks/FaceLandmarks.jsx'),
);
const PoseEstimation = lazy(() =>
  import('./modules/pose-estimation/PoseEstimation.jsx'),
);
const AudioClassification = lazy(() =>
  import('./modules/audio-classification/AudioClassification.jsx'),
);
const ObjectDetection = lazy(() =>
  import('./modules/object-detection/ObjectDetection.jsx'),
);
const HandGestures = lazy(() =>
  import('./modules/hand-gestures/HandGestures.jsx'),
);
const HeadTracking = lazy(() =>
  import('./modules/head-tracking/HeadTracking.jsx'),
);
const VisualQA = lazy(() =>
  import('./modules/visual-qa/VisualQA.jsx'),
);
const LLMChat = lazy(() =>
  import('./modules/llm-chat/LLMChat.jsx'),
);
const ImageSegmentation = lazy(() =>
  import('./modules/image-segmentation/ImageSegmentation.jsx'),
);
const TextClassification = lazy(() =>
  import('./modules/text-classification/TextClassification.jsx'),
);
const FaqPage = lazy(() => import('./modules/faq/FaqPage.jsx'));

function LazyRoute({ children }) {
  return <Suspense fallback={<div>Lade Modul...</div>}>{children}</Suspense>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route
        path="/image-classification"
        element={
          <LazyRoute>
            <ImageClassification />
          </LazyRoute>
        }
      />
      <Route
        path="/object-detection"
        element={
          <LazyRoute>
            <ObjectDetection />
          </LazyRoute>
        }
      />
      <Route
        path="/visual-qa"
        element={
          <LazyRoute>
            <VisualQA />
          </LazyRoute>
        }
      />
      <Route
        path="/image-segmentation"
        element={
          <LazyRoute>
            <ImageSegmentation />
          </LazyRoute>
        }
      />
      <Route
        path="/llm-chat"
        element={
          <LazyRoute>
            <LLMChat />
          </LazyRoute>
        }
      />
      <Route
        path="/text-classification"
        element={
          <LazyRoute>
            <TextClassification />
          </LazyRoute>
        }
      />
      <Route
        path="/audio-classification"
        element={
          <LazyRoute>
            <AudioClassification />
          </LazyRoute>
        }
      />
      <Route
        path="/pose-estimation"
        element={
          <LazyRoute>
            <PoseEstimation />
          </LazyRoute>
        }
      />
      <Route
        path="/face-landmarks"
        element={
          <LazyRoute>
            <FaceLandmarks />
          </LazyRoute>
        }
      />
      <Route
        path="/gestenerkennung"
        element={
          <LazyRoute>
            <HandGestures />
          </LazyRoute>
        }
      />
      <Route
        path="/head-tracking"
        element={
          <LazyRoute>
            <HeadTracking />
          </LazyRoute>
        }
      />
      <Route
        path="/faq"
        element={
          <LazyRoute>
            <FaqPage />
          </LazyRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
