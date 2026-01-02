import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-wasm';

// Optional: configure WASM paths if needed.
// import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';

let initializationPromise = null;

export async function initTensorFlowBackend() {
  if (initializationPromise) return initializationPromise;

  initializationPromise = (async () => {
    const backendsToTry = ['webgpu', 'webgl', 'wasm', 'cpu'];

    for (const backend of backendsToTry) {
      try {
        if (!tf.findBackend(backend)) {
          console.warn(`Backend ${backend} not found or not supported.`);
          continue;
        }

        await tf.setBackend(backend);
        await tf.ready();

        console.log(`TensorFlow backend set to: ${tf.getBackend()}`);
        return tf.getBackend();
      } catch (err) {
        console.warn(`Failed to initialize ${backend}:`, err);
      }
    }

    throw new Error('No supported TensorFlow backend could be initialized.');
  })();

  return initializationPromise;
}
