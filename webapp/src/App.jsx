import { useState } from 'react'
import SingleImage from './components/SingleImage.jsx'
import Benchmark from './components/Benchmark.jsx'

export default function App() {
  const [tab, setTab] = useState('single')

  return (
    <div className="app">
      <div className="header">
        <h1>
          MNIST Inference
          <span className="badge">UI ONLY · pipeline stubbed</span>
        </h1>
        <p>Draw or upload a 28×28 digit, or benchmark the full 10,000-image test set.</p>
      </div>

      <div className="tabs">
        <button
          className={'tab' + (tab === 'single' ? ' active' : '')}
          onClick={() => setTab('single')}
        >
          Draw / Upload
        </button>
        <button
          className={'tab' + (tab === 'batch' ? ' active' : '')}
          onClick={() => setTab('batch')}
        >
          Benchmark 10,000
        </button>
      </div>

      {tab === 'single' ? <SingleImage /> : <Benchmark />}

      <div className="note">
        Inference is not wired up yet. Predictions here are placeholder values from{' '}
        <code>src/lib/inference.js</code>. Replace <code>predict()</code> and{' '}
        <code>runBatch()</code> with real calls to your model backend.
      </div>
    </div>
  )
}
