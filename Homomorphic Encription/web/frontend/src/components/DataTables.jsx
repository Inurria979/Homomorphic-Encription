import { useState } from 'react'
import { api } from '../api'

const TABLAS = [
  ['comparativa', 'Comparativa (plana vs homomórfica + derivadas)'],
  ['barrido_ckks', 'Barrido de parámetros CKKS (grado × escala)'],
  ['experimentos', 'Experimentos'],
  ['predicciones', 'Predicciones (todas las métricas)'],
  ['semillas', 'Entrenamientos por semilla'],
  ['metricas_clase', 'Métricas por clase'],
]

export default function DataTables() {
  return (
    <div className="fig" style={{ maxWidth: 'var(--fig)' }}>
      {TABLAS.map(([nombre, titulo]) => <Tabla key={nombre} nombre={nombre} titulo={titulo} />)}
    </div>
  )
}

function Tabla({ nombre, titulo }) {
  const [rows, setRows] = useState(null)

  const abrir = (e) => {
    if (e.target.open && rows == null) api.tabla(nombre).then(setRows)
  }

  return (
    <details className="data" onToggle={abrir}>
      <summary>
        <span>{titulo}</span>
        <span className="cnt">{rows ? `${rows.length} filas · CSV ⬇` : 'ver'}</span>
      </summary>
      <div className="tablescroll">
        {rows == null ? (
          <p style={{ fontFamily: 'var(--sans)', color: 'var(--muted)', padding: '4px 0 14px' }}>Cargando…</p>
        ) : rows.length === 0 ? (
          <p style={{ fontFamily: 'var(--sans)', color: 'var(--muted)' }}>Sin filas.</p>
        ) : (
          <>
            <button className="btn" style={{ margin: '4px 0 12px' }} onClick={() => descargarCSV(nombre, rows)}>
              ⬇ Descargar {nombre}.csv
            </button>
            <table className="grid">
              <thead>
                <tr>{Object.keys(rows[0]).map((k) => <th key={k}>{k}</th>)}</tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>{Object.keys(rows[0]).map((k) => <td key={k}>{fmt(r[k])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </details>
  )
}

function fmt(v) {
  if (v == null) return '—'
  if (typeof v === 'number' && !Number.isInteger(v)) return v.toFixed(3)
  return String(v)
}

function descargarCSV(nombre, rows) {
  const cols = Object.keys(rows[0])
  const esc = (v) => {
    if (v == null) return ''
    const s = String(v)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = `${nombre}.csv`; a.click()
  URL.revokeObjectURL(url)
}
