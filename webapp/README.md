# MNIST Inference Web App

Vite + React front end for the MNIST_MODEL project. **UI only** — the
inference pipeline is stubbed, not wired to a real model yet.

## Two modes

- **Draw / Upload** — draw a digit on the canvas or drop in an image. It's
  downsampled to 28×28 grayscale (784 values) and sent to `predict()`.
- **Benchmark 10,000** — runs the full MNIST test set and reports throughput
  (images/sec, ms/image), total time, and accuracy.

## Run it

```bash
cd webapp
npm install
npm run dev      # http://localhost:5173
npm run build    # production build in dist/
```

## Wiring up a real model

Everything fake lives in `src/lib/inference.js`. Replace two functions:

- `predict(pixels)` → returns `{ probs: number[10], predicted: number }`.
  `pixels` is a `Float32Array` of length 784, values 0..1.
- `runBatch(total, onProgress)` → returns
  `{ total, correct, accuracy, elapsedMs, imagesPerSec }`.

Point these at ONNX Runtime Web (there are `.onnx` files in the parent repo),
a WASM build, or a `fetch()` to a server. The UI reads only those return
shapes, so nothing else needs to change.
