import * as tf from '@tensorflow/tfjs';

class TokenAndPositionEmbedding extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.vocabSize = config.vocabSize;
    this.maxLen = config.maxLen;
    this.embedDim = config.embedDim;
  }

  build() {
    const init = tf.initializers.randomNormal({ mean: 0, stddev: 0.02 });
    this.tokenEmbeddings = this.addWeight(
      'tokenEmbeddings',
      [this.vocabSize, this.embedDim],
      'float32',
      init,
    );
    this.positionEmbeddings = this.addWeight(
      'positionEmbeddings',
      [this.maxLen, this.embedDim],
      'float32',
      init,
    );
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const seqLen = input.shape[1] ?? this.maxLen;
      const tokenEmb = tf.gather(this.tokenEmbeddings.read(), input);
      const positions = tf.range(0, seqLen, 1, 'int32');
      const positionEmb = tf
        .gather(this.positionEmbeddings.read(), positions)
        .expandDims(0);
      return tokenEmb.add(positionEmb);
    });
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], inputShape[1], this.embedDim];
  }

  getConfig() {
    const baseConfig = super.getConfig();
    return {
      ...baseConfig,
      vocabSize: this.vocabSize,
      maxLen: this.maxLen,
      embedDim: this.embedDim,
    };
  }

  static get className() {
    return 'TokenAndPositionEmbedding';
  }
}

class MultiHeadSelfAttention extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.embedDim = config.embedDim;
    this.numHeads = config.numHeads;
    this.headDim = Math.floor(this.embedDim / this.numHeads);
    if (this.embedDim % this.numHeads !== 0) {
      throw new Error('embedDim muss durch numHeads teilbar sein.');
    }
  }

  build() {
    const init = tf.initializers.glorotUniform();
    this.wq = this.addWeight('wq', [this.embedDim, this.embedDim], 'float32', init);
    this.wk = this.addWeight('wk', [this.embedDim, this.embedDim], 'float32', init);
    this.wv = this.addWeight('wv', [this.embedDim, this.embedDim], 'float32', init);
    this.wo = this.addWeight('wo', [this.embedDim, this.embedDim], 'float32', init);
    this.built = true;
  }

  call(inputs) {
    return tf.tidy(() => {
      const input = Array.isArray(inputs) ? inputs[0] : inputs;
      const batchSize = input.shape[0] ?? -1;
      const seqLen = input.shape[1];
      const project = (tensor, weight) => {
        const flat = tensor.reshape([-1, this.embedDim]);
        const projected = tf.matMul(flat, weight);
        return projected.reshape([batchSize, seqLen, this.embedDim]);
      };

      const q = project(input, this.wq.read());
      const k = project(input, this.wk.read());
      const v = project(input, this.wv.read());

      const reshapeHeads = (tensor) =>
        tensor
          .reshape([batchSize, seqLen, this.numHeads, this.headDim])
          .transpose([0, 2, 1, 3]);

      const qHeads = reshapeHeads(q);
      const kHeads = reshapeHeads(k);
      const vHeads = reshapeHeads(v);

      const scale = tf.scalar(1 / Math.sqrt(this.headDim));
      const scores = tf.mul(tf.matMul(qHeads, kHeads, false, true), scale);
      const mask = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0);
      const additiveMask = tf.mul(tf.sub(1, mask), tf.scalar(-1e9))
        .reshape([1, 1, seqLen, seqLen]);
      const maskedScores = scores.add(additiveMask);
      const weights = tf.softmax(maskedScores);
      const attention = tf.matMul(weights, vHeads)
        .transpose([0, 2, 1, 3])
        .reshape([batchSize, seqLen, this.embedDim]);

      return project(attention, this.wo.read());
    });
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], inputShape[1], this.embedDim];
  }

  getConfig() {
    const baseConfig = super.getConfig();
    return {
      ...baseConfig,
      embedDim: this.embedDim,
      numHeads: this.numHeads,
    };
  }

  static get className() {
    return 'MultiHeadSelfAttention';
  }
}

tf.serialization.registerClass(TokenAndPositionEmbedding);
tf.serialization.registerClass(MultiHeadSelfAttention);

function transformerBlock(x, config, blockIndex) {
  const attn = new MultiHeadSelfAttention({
    embedDim: config.embedDim,
    numHeads: config.numHeads,
    name: `encoder-attn-${blockIndex}`,
  }).apply(x);
  const attnOut = tf.layers
    .add({ name: `encoder-attn-add-${blockIndex}` })
    .apply([x, attn]);
  const attnNorm = tf.layers
    .layerNormalization({ epsilon: 1e-6, name: `encoder-attn-norm-${blockIndex}` })
    .apply(attnOut);

  const ffOut = tf.layers
    .dense({
      units: config.ffDim,
      activation: 'relu',
      name: `encoder-ffn-${blockIndex}-1`,
    })
    .apply(attnNorm);
  const ffProjected = tf.layers
    .dense({ units: config.embedDim, name: `encoder-ffn-${blockIndex}-2` })
    .apply(ffOut);
  const ffAdd = tf.layers
    .add({ name: `encoder-ffn-add-${blockIndex}` })
    .apply([attnNorm, ffProjected]);

  return tf.layers
    .layerNormalization({ epsilon: 1e-6, name: `encoder-ffn-norm-${blockIndex}` })
    .apply(ffAdd);
}

export function createNanoTransformer({
  vocabSize,
  contextWindow,
  embedDim = 64,
  numHeads = 4,
  numLayers = 2,
  ffDim = 128,
}) {
  const inputs = tf.input({
    shape: [contextWindow],
    dtype: 'int32',
    name: 'token-input',
  });

  let x = new TokenAndPositionEmbedding({
    vocabSize,
    maxLen: contextWindow,
    embedDim,
    name: 'token-pos-embedding',
  }).apply(inputs);

  for (let i = 0; i < numLayers; i += 1) {
    x = transformerBlock(x, { embedDim, numHeads, ffDim }, i + 1);
  }

  const outputs = tf.layers
    .dense({ units: vocabSize, activation: 'softmax', name: 'lm-head' })
    .apply(x);

  return tf.model({ inputs, outputs, name: 'nano-transformer' });
}

export function setEncoderTrainable(model, trainable) {
  model.layers.forEach((layer) => {
    if (layer.name === 'lm-head') {
      layer.trainable = true;
      return;
    }
    layer.trainable = trainable;
  });
}

export function createCharTokenizer(text, maxVocabSize = 500) {
  const counts = new Map();
  for (const char of text) {
    counts.set(char, (counts.get(char) ?? 0) + 1);
  }

  const vocab = Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, Math.max(1, maxVocabSize - 1))
    .map(([char]) => char);

  const idxToChar = ['<unk>', ...vocab];
  const charToIdx = new Map(idxToChar.map((char, index) => [char, index]));

  const encode = (input) => {
    const tokens = new Int32Array(input.length);
    let i = 0;
    for (const char of input) {
      tokens[i] = charToIdx.get(char) ?? 0;
      i += 1;
    }
    return tokens;
  };

  const decode = (tokens) =>
    Array.from(tokens, (token) => idxToChar[token] ?? '').join('');

  return {
    vocabSize: idxToChar.length,
    vocab: idxToChar,
    encode,
    decode,
  };
}

export function createBatch(tokens, contextWindow, batchSize) {
  const maxStart = tokens.length - contextWindow - 1;
  if (maxStart <= 0) {
    throw new Error('Text ist zu kurz fuer das Kontextfenster.');
  }

  const xs = new Int32Array(batchSize * contextWindow);
  const ys = new Int32Array(batchSize * contextWindow);

  for (let batch = 0; batch < batchSize; batch += 1) {
    const start = Math.floor(Math.random() * maxStart);
    const offset = batch * contextWindow;
    for (let step = 0; step < contextWindow; step += 1) {
      xs[offset + step] = tokens[start + step];
      ys[offset + step] = tokens[start + step + 1];
    }
  }

  return { xs, ys };
}

export async function warmupModel(model, contextWindow) {
  const output = tf.tidy(() => {
    const dummy = tf.zeros([1, contextWindow], 'int32');
    return model.predict(dummy);
  });
  await output.data();
  output.dispose();
}

function sparseCategoricalCrossentropyFromProbs(labels, probs, vocabSize) {
  const clipped = tf.clipByValue(probs, 1e-7, 1);
  const oneHot = tf.oneHot(labels, vocabSize);
  const logProbs = tf.log(clipped);
  const perExample = tf.sum(tf.mul(oneHot, logProbs), -1);
  return tf.neg(perExample);
}

function sampleFromProbs(probabilities, temperature = 1, topK = 0) {
  const epsilon = 1e-8;
  const entries = [];

  for (let i = 0; i < probabilities.length; i += 1) {
    entries.push({ index: i, value: Math.max(probabilities[i], epsilon) });
  }

  if (topK > 0 && topK < entries.length) {
    entries.sort((a, b) => b.value - a.value);
    entries.length = topK;
  }

  if (temperature !== 1) {
    const invTemp = 1 / Math.max(temperature, 1e-3);
    for (const entry of entries) {
      entry.value = Math.pow(entry.value, invTemp);
    }
  }

  let total = 0;
  for (const entry of entries) {
    total += entry.value;
  }

  let threshold = Math.random() * total;
  for (const entry of entries) {
    threshold -= entry.value;
    if (threshold <= 0) return entry.index;
  }

  return entries[entries.length - 1]?.index ?? 0;
}

export async function generateNanoText({
  model,
  tokenizer,
  seed,
  contextWindow,
  maxNewTokens = 120,
  temperature = 0.8,
  topK = 40,
  onToken,
  shouldStop,
}) {
  const tokens = Array.from(tokenizer.encode(seed));
  const vocabSize = tokenizer.vocabSize;

  for (let step = 0; step < maxNewTokens; step += 1) {
    if (shouldStop?.()) break;

    const contextTokens = tokens.slice(-contextWindow);
    const input = new Int32Array(contextWindow);
    input.set(contextTokens, contextWindow - contextTokens.length);

    const nextToken = tf.tidy(() => {
      const inputTensor = tf.tensor2d(input, [1, contextWindow], 'int32');
      const preds = model.predict(inputTensor);
      const last = preds
        .slice([0, contextWindow - 1, 0], [1, 1, vocabSize])
        .reshape([vocabSize]);
      const probs = last.dataSync();
      return sampleFromProbs(probs, temperature, topK);
    });

    tokens.push(nextToken);
    if (typeof onToken === 'function') {
      onToken(tokenizer.decode([nextToken]));
    }

    if ((step + 1) % 4 === 0) {
      await tf.nextFrame();
    }
  }

  return tokenizer.decode(tokens);
}

export async function trainNanoTransformer({
  model,
  tokens,
  vocabSize,
  contextWindow,
  batchSize,
  epochs,
  stepsPerEpoch,
  optimizer,
  onEpochEnd,
  onBatchEnd,
  shouldStop,
}) {
  const trainableVars = model.trainableWeights.map((weight) => weight.val);

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    if (shouldStop?.()) break;
    let epochLoss = 0;
    let stepsRun = 0;

    for (let step = 0; step < stepsPerEpoch; step += 1) {
      if (shouldStop?.()) break;

      const lossTensor = tf.tidy(() => {
        // Training step inside tf.tidy keeps memory stable on iPad.
        const { xs, ys } = createBatch(tokens, contextWindow, batchSize);
        const inputs = tf.tensor2d(xs, [batchSize, contextWindow], 'int32');
        const labels = tf.tensor2d(ys, [batchSize, contextWindow], 'int32');

        const loss = optimizer.minimize(() => {
          const predictions = model.apply(inputs, { training: true });
          const flatPreds = predictions.reshape([batchSize * contextWindow, vocabSize]);
          const flatLabels = labels.reshape([batchSize * contextWindow]);
          const batchLoss = sparseCategoricalCrossentropyFromProbs(
            flatLabels,
            flatPreds,
            vocabSize,
          );
          return tf.mean(batchLoss);
        }, true, trainableVars);

        inputs.dispose();
        labels.dispose();

        return loss;
      });

      const lossValue = (await lossTensor.data())[0];
      lossTensor.dispose();
      epochLoss += lossValue;
      stepsRun += 1;

      if (typeof onBatchEnd === 'function') {
        onBatchEnd({ epoch, step, loss: lossValue });
      }
    }

    const avgLoss = stepsRun ? epochLoss / stepsRun : 0;
    if (typeof onEpochEnd === 'function') {
      onEpochEnd({ epoch, loss: avgLoss });
    }

    await tf.nextFrame();
  }
}
