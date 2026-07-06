import { useRef, useState, useEffect } from 'react'
import { predict } from '../lib/inference.js'

const CANVAS = 280 // draw canvas px
const GRID = 28

export default function SingleImage() {
  const drawRef = useRef(null)
  const previewRef = useRef(null)
  const [drawing, setDrawing] = useState(false)
  const [hasInput, setHasInput] = useState(false)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  // init draw canvas
  useEffect(() => {
    const ctx = drawRef.current.getContext('2d')
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, CANVAS, CANVAS)
  }, [])

  function pos(e) {
    const rect = drawRef.current.getBoundingClientRect()
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top
    return { x: (cx / rect.width) * CANVAS, y: (cy / rect.height) * CANVAS }
  }

  function start(e) {
    e.preventDefault()
    setDrawing(true)
    setResult(null)
    draw(e)
  }
  function draw(e) {
    if (e.type !== 'mousedown' && e.type !== 'touchstart' && !drawing) return
    const ctx = drawRef.current.getContext('2d')
    const { x, y } = pos(e)
    ctx.fillStyle = '#fff'
    ctx.beginPath()
    ctx.arc(x, y, 12, 0, Math.PI * 2)
    ctx.fill()
    setHasInput(true)
    renderPreview()
  }
  function stop() { setDrawing(false) }

  function clear() {
    const ctx = drawRef.current.getContext('2d')
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, CANVAS, CANVAS)
    const p = previewRef.current.getContext('2d')
    p.fillStyle = '#000'
    p.fillRect(0, 0, GRID, GRID)
    setHasInput(false)
    setResult(null)
  }

  // downsample draw canvas -> 28x28 preview
  function renderPreview() {
    const src = drawRef.current
    const dst = previewRef.current
    const dctx = dst.getContext('2d')
    dctx.imageSmoothingEnabled = true
    dctx.clearRect(0, 0, GRID, GRID)
    dctx.drawImage(src, 0, 0, GRID, GRID)
  }

  // pull 784 grayscale floats 0..1 from the 28x28 preview
  function getPixels() {
    const p = previewRef.current.getContext('2d')
    const { data } = p.getImageData(0, 0, GRID, GRID)
    const out = new Float32Array(GRID * GRID)
    for (let i = 0; i < GRID * GRID; i++) out[i] = data[i * 4] / 255
    return out
  }

  function handleFile(file) {
    if (!file) return
    const img = new Image()
    img.onload = () => {
      const ctx = drawRef.current.getContext('2d')
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, CANVAS, CANVAS)
      ctx.drawImage(img, 0, 0, CANVAS, CANVAS)
      setHasInput(true)
      setResult(null)
      renderPreview()
    }
    img.src = URL.createObjectURL(file)
  }

  async function run() {
    setBusy(true)
    const res = await predict(getPixels())
    setResult(res)
    setBusy(false)
  }

  return (
    <div className="panel">
      <div className="grid">
        <div>
          <h3 className="section-title">Input</h3>
          <canvas
            ref={drawRef}
            className="draw"
            width={CANVAS}
            height={CANVAS}
            onMouseDown={start}
            onMouseMove={draw}
            onMouseUp={stop}
            onMouseLeave={stop}
            onTouchStart={start}
            onTouchMove={draw}
            onTouchEnd={stop}
          />
          <div className="row" style={{ marginTop: 12 }}>
            <button className="btn" onClick={clear}>Clear</button>
            <button className="btn primary" onClick={run} disabled={!hasInput || busy}>
              {busy ? 'Predicting…' : 'Predict'}
            </button>
          </div>

          <div
            className={'dropzone' + (dragOver ? ' drag' : '')}
            style={{ marginTop: 14 }}
            onClick={() => document.getElementById('file-in').click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault(); setDragOver(false)
              handleFile(e.dataTransfer.files[0])
            }}
          >
            Drag an image here, or click to upload
            <input
              id="file-in"
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>
        </div>

        <div>
          <h3 className="section-title">Model sees (28×28)</h3>
          <div className="preview-box">
            <canvas ref={previewRef} className="preview" width={GRID} height={GRID} />
            <span className="preview-label">downsampled input · 784 grayscale values</span>
          </div>

          {result && (
            <div className="result">
              <div className="row" style={{ justifyContent: 'space-between' }}>
                <div>
                  <div className="pred-digit">{result.predicted}</div>
                  <div className="pred-conf">
                    {(result.probs[result.predicted] * 100).toFixed(1)}% confidence
                  </div>
                </div>
              </div>
              <div className="bars">
                {result.probs.map((p, i) => (
                  <div className="bar-row" key={i}>
                    <span className="lbl">{i}</span>
                    <span className="bar-track">
                      <span className="bar-fill" style={{ width: `${p * 100}%` }} />
                    </span>
                    <span className="val">{(p * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
