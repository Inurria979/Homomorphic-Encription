import { useEffect, useState } from 'react'
import { api } from '../api'
import EChart from '../EChart'
import { C } from '../theme'
import { matrizConfusion, metricasClase, semillas, polinomioVsRelu } from '../charts'

const METRICAS_CLASE = [
  ['f1_pct', 'F1 (%)'],
  ['precision_pct', 'Precisión (%)'],
  ['recall_pct', 'Recall (%)'],
  ['especificidad_pct', 'Especificidad (%)'],
]

const f2 = (v) => (v == null ? '—' : (+v).toFixed(2))
const miles = (v) => (v == null ? '—' : Math.round(v).toLocaleString('es-ES'))

export default function ExperimentExplorer({ comp }) {
  const [id, setId] = useState(comp[0]?.experimento_id)
  const [detalle, setDetalle] = useState(null)
  const [metricaClase, setMetricaClase] = useState('f1_pct')
  const [cargando, setCargando] = useState(false)
  // Índice de la configuración CKKS mostrada (los experimentos del barrido
  // tienen varias predicciones homomórficas, una por configuración).
  const [cfgIdx, setCfgIdx] = useState(0)

  useEffect(() => {
    let vivo = true
    setCargando(true)
    setCfgIdx(0)
    api.experimento(id).then((d) => { if (vivo) { setDetalle(d); setCargando(false) } })
    return () => { vivo = false }
  }, [id])

  const fila = comp.find((r) => r.experimento_id === id)
  if (!fila) return null

  const clases = detalle ? [...Array(detalle.info.num_clases).keys()] : []
  const plana = detalle?.predicciones?.plana
  const configs = detalle?.homomorficas || []
  // La configuración seleccionada manda; si no hay lista, la representativa.
  const homo = configs[cfgIdx] || detalle?.predicciones?.homomorfica
  const labelMc = METRICAS_CLASE.find(([k]) => k === metricaClase)[1]

  return (
    <div className="explorer">
      <div className="controls">
        <span className="ctl-label">Experimento</span>
        <select className="ui" value={id} onChange={(e) => setId(+e.target.value)}>
          {comp.map((r) => (
            <option key={r.experimento_id} value={r.experimento_id}>
              {r.nombre}
            </option>
          ))}
        </select>
      </div>

      {detalle && (
        <p className="caption" style={{ marginTop: -6, marginBottom: 14 }}>
          <b>{detalle.info.titulo}</b> — {detalle.info.subtitulo}
        </p>
      )}

      {/* Selector de configuración CKKS (solo si se barrieron varias) */}
      {configs.length > 1 && (
        <div className="controls">
          <span className="ctl-label">Configuración CKKS</span>
          <select className="ui" value={cfgIdx} onChange={(e) => setCfgIdx(+e.target.value)}>
            {configs.map((c, i) => (
              <option key={c.prediccion_id} value={i}>
                {c.config_ckks} · {f2(c.metricas.accuracy)}% · {f2(c.metricas.tiempo_wall_s)} s
              </option>
            ))}
          </select>
          <span className="caption" style={{ marginLeft: 10 }}>
            {configs.length} configuraciones probadas
          </span>
        </div>
      )}

      {/* Tarjetas resumen: reflejan la configuración CKKS seleccionada */}
      {(() => {
        const m = homo?.metricas
        const accH = m ? m.accuracy : fila.accuracy_homo
        const tH = m ? m.tiempo_wall_s : fila.tiempo_wall_s_homo
        const ramH = m ? m.ram_pico_proceso_gb : fila.ram_pico_proceso_gb_homo
        const delta = fila.accuracy_plana - accH
        const factor = fila.tiempo_wall_s_plana ? tH / fila.tiempo_wall_s_plana : null
        const overhead = ramH - fila.ram_pico_proceso_gb_plana
        return (
          <div className="exp-metrics">
            <Metric label="Accuracy plana" val={`${f2(fila.accuracy_plana)}%`} />
            <Metric
              label="Accuracy homomórfica" val={`${f2(accH)}%`}
              delta={delta > 0.01
                ? { txt: `−${f2(delta)} pp`, cls: 'down' }
                : { txt: 'sin pérdida', cls: 'flat' }} />
            <Metric label="Tiempo homomórfico" val={`${miles(tH)} s`}
              delta={factor ? { txt: `${miles(factor)}× más lento`, cls: 'down' } : null} />
            <Metric label="RAM pico homomórfica" val={`${f2(ramH)} GB`}
              delta={{ txt: `+${f2(overhead)} GB`, cls: 'down' }} />
          </div>
        )
      })()}

      {cargando && !detalle ? (
        <div className="state"><div className="spinner" />Cargando experimento…</div>
      ) : detalle && (
        <>
          {/* Matrices de confusión enfrentadas */}
          <h4 style={{ fontFamily: 'var(--sans)', textAlign: 'center', margin: '10px 0 4px' }}>
            Matrices de confusión
          </h4>
          <div className="two-col">
            <div className="chartcard">
              <Titulo t="Plana" color={C.plana} />
              <EChart height={320} option={matrizConfusion(plana?.matriz_confusion, clases, C.plana)} />
            </div>
            <div className="chartcard">
              <Titulo t="Homomórfica" color={C.homo} />
              <EChart height={320} option={matrizConfusion(homo?.matriz_confusion, clases, C.homo)} />
            </div>
          </div>

          {/* Métricas por clase */}
          <div className="controls" style={{ marginTop: 30 }}>
            <span className="ctl-label">Métricas por clase</span>
            <div className="segmented">
              {METRICAS_CLASE.map(([k, l]) => (
                <button key={k} className={k === metricaClase ? 'on' : ''} onClick={() => setMetricaClase(k)}>{l}</button>
              ))}
            </div>
          </div>
          <div className="chartcard">
            <EChart height={340} option={metricasClase(detalle.metricas_clase, metricaClase, labelMc)} />
          </div>

          {/* Semillas + parámetros CKKS */}
          <div className="two-col" style={{ marginTop: 22, alignItems: 'start' }}>
            <div className="chartcard">
              <Titulo t="Entrenamiento por semilla" />
              {detalle.semillas.length
                ? <EChart height={300} option={semillas(detalle.semillas)} />
                : <p style={{ padding: 20, color: 'var(--muted)' }}>Sin registro de semillas.</p>}
            </div>
            <ParamCard homo={homo} info={detalle.info} />
          </div>

          {/* Polinomio que sustituye a la ReLU dentro del cifrado */}
          {homo?.rango?.length === 2 && homo?.coeficientes?.length > 0 && (
            <div className="chartcard" style={{ marginTop: 22 }}>
              <Titulo t={`Aproximación de la ReLU · polinomio de grado ${homo.ckks.grado_taylor}`} />
              <EChart height={340}
                option={polinomioVsRelu(homo.rango, homo.coeficientes, +homo.ckks.grado_taylor)} />
              <p className="caption" style={{ padding: '4px 14px 12px' }}>
                CKKS solo suma y multiplica, así que la ReLU se sustituye por un polinomio.
                Se ajusta por mínimos cuadrados al <b>rango real</b> de las pre-activaciones
                (<span style={{ fontFamily: 'var(--mono)' }}>
                  [{f2(homo.rango[0])}, {f2(homo.rango[1])}]
                </span>), no a [−1, 1]: el fan-in y la estandarización llevan las
                pre-activaciones muy lejos de ese intervalo.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function Metric({ label, val, delta }) {
  return (
    <div className="metric-card">
      <div className="mlbl">{label}</div>
      <div className="mval">{val}</div>
      {delta && <div className={`mdelta ${delta.cls}`}>{delta.txt}</div>}
    </div>
  )
}

function Titulo({ t, color }) {
  return (
    <div style={{ fontFamily: 'var(--sans)', fontSize: 13, fontWeight: 600, textAlign: 'center', padding: '6px 0', color: color || 'var(--ink-soft)' }}>
      {color && <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: 2, background: color, marginRight: 6 }} />}
      {t}
    </div>
  )
}

function ParamCard({ homo, info }) {
  const k = homo?.ckks
  const rango = homo?.rango
  const coefs = homo?.coeficientes
  return (
    <div className="paramcard">
      <h4>Parámetros de cifrado (CKKS)</h4>
      {k ? (
        <table>
          <tbody>
            <Row k="poly_modulus_degree" v={Math.round(k.poly_modulus_degree)} />
            <Row k="global_scale (bits)" v={`2^${k.global_scale_bits}`} />
            <Row k="Grado del polinomio" v={k.grado_taylor} />
            {rango?.length === 2 && (
              <Row k="Rango de activación" v={`[${f2(rango[0])}, ${f2(rango[1])}]`} />
            )}
            <Row k="Batch efectivo" v={k.batch_size_efectivo} />
            <Row k="Workers efectivos" v={k.num_workers_efectivos} />
            <Row k="Tiempo de contexto" v={`${(+k.tiempo_contexto).toFixed(1)} s`} />
            <Row k="Clases" v={info.num_clases} />
          </tbody>
        </table>
      ) : <p style={{ color: 'var(--muted)' }}>Sin predicción homomórfica.</p>}
      {k && (
        <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--muted)', marginTop: 12, wordBreak: 'break-all' }}>
          <p>coeff_mod_bit_sizes: {k.coeff_mod_bit_sizes}</p>
          {coefs?.length > 0 && (
            <p style={{ marginTop: 6 }}>
              coeficientes: [{coefs.map((c) => (+c).toFixed(4)).join(', ')}]
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function Row({ k, v }) {
  return <tr><td>{k}</td><td>{v}</td></tr>
}
