// ─────────────────────────────────────────────────────────────
// STUB inference layer. Not wired to a real model yet.
// Swap these two functions out for real backend calls (ONNX
// runtime web, a fetch() to a server, WASM, etc).
// `pixels` is a Float32Array of length 784 (28*28), values 0..1.
// ─────────────────────────────────────────────────────────────

function softmaxRandom(seed = Math.random()) {
  const logits = Array.from({ length: 10 }, () => Math.random() * 4 - 1)
  // nudge one class up so the demo looks plausible
  logits[Math.floor(seed * 10) % 10] += 3
  const max = Math.max(...logits)
  const exps = logits.map((l) => Math.exp(l - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

// Single-image prediction. Returns { probs:number[10], predicted:number }.
export async function predict(pixels) {
  await new Promise((r) => setTimeout(r, 120)) // fake latency
  const probs = softmaxRandom()
  const predicted = probs.indexOf(Math.max(...probs))
  return { probs, predicted }
}

// Full test-set benchmark. `onProgress(done, total)` is called as it runs.
// Returns { total, correct, accuracy, elapsedMs, imagesPerSec }.
export async function runBatch(total = 10000, onProgress = () => {}) {
  const start = performance.now()
  let correct = 0
  const chunk = 250
  for (let done = 0; done < total; done += chunk) {
    // fake per-chunk compute
    await new Promise((r) => setTimeout(r, 12))
    const n = Math.min(chunk, total - done)
    // pretend ~98.7% accuracy
    for (let i = 0; i < n; i++) if (Math.random() < 0.987) correct++
    onProgress(Math.min(done + n, total), total)
  }
  const elapsedMs = performance.now() - start
  return {
    total,
    correct,
    accuracy: correct / total,
    elapsedMs,
    imagesPerSec: total / (elapsedMs / 1000),
  }
}
