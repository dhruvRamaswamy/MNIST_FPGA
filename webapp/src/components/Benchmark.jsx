import { useState } from 'react'
import { runBatch } from '../lib/inference.js'

const TOTAL = 10000

export default function Benchmark() {
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState(0)
  const [result, setResult] = useState(null)

  async function run() {
    setRunning(true)
    setResult(null)
    setDone(0)
    const res = await runBatch(TOTAL, (d) => setDone(d))
    setResult(res)
    setRunning(false)
  }

  const pct = ((done / TOTAL) * 100).toFixed(0)

  return (
    <div className="panel">
      <h3 className="section-title">Full test set · {TOTAL.toLocaleString()} images</h3>
      <p className="muted" style={{ marginTop: 0 }}>
        Runs the entire MNIST test set through the model and reports throughput and accuracy.
      </p>

      <div className="row">
        <button className="btn primary" onClick={run} disabled={running}>
          {running ? 'Running…' : 'Run benchmark'}
        </button>
        {running && (
          <span className="muted mono">
            {done.toLocaleString()} / {TOTAL.toLocaleString()}
          </span>
        )}
      </div>

      {(running || result) && (
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${result ? 100 : pct}%` }}
          />
        </div>
      )}

      {result && (
        <div className="stats">
          <div className="stat">
            <div className="num">{result.imagesPerSec.toFixed(0)}</div>
            <div className="lbl">images / sec</div>
          </div>
          <div className="stat">
            <div className="num">{(result.elapsedMs / 1000).toFixed(2)}s</div>
            <div className="lbl">total time</div>
          </div>
          <div className="stat">
            <div className="num">{(result.accuracy * 100).toFixed(2)}%</div>
            <div className="lbl">accuracy</div>
          </div>
          <div className="stat">
            <div className="num">{(result.elapsedMs / result.total).toFixed(3)}</div>
            <div className="lbl">ms / image</div>
          </div>
        </div>
      )}

      {result && (
        <p className="muted" style={{ marginTop: 16 }}>
          {result.correct.toLocaleString()} / {result.total.toLocaleString()} correct.
        </p>
      )}
    </div>
  )
}
