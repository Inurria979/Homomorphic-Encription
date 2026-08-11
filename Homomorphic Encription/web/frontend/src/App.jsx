import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import EChart from './EChart'
import { Section, Head, Figure } from './ui'
import Toc from './components/Toc'
import ExperimentExplorer from './components/ExperimentExplorer'
import DataTables from './components/DataTables'
import {
  fidelidad, deltaPrecision, tiempo, ralentizacion, tiempoVsMuestras,
  ram, barrasComparativa, radar,
  barridoGrado, barridoCoste, barridoEscala,
} from './charts'

const METRICAS = [
  ['accuracy', 'Accuracy (%)', false],
  ['balanced_accuracy', 'Balanced accuracy (%)', false],
  ['f1_macro', 'F1 macro (%)', false],
  ['f1_weighted', 'F1 weighted (%)', false],
  ['precision_macro', 'Precisión macro (%)', false],
  ['recall_macro', 'Recall macro (%)', false],
  ['mcc', 'MCC (−1..1)', true],
  ['cohen_kappa', 'Cohen kappa (−1..1)', true],
]

const SECCIONES = [
  { id: 'resumen', label: 'Resumen' },
  { id: 'fidelidad', label: 'Fidelidad' },
  { id: 'tiempo', label: 'Coste temporal' },
  { id: 'recursos', label: 'Recursos' },
  { id: 'cifrado', label: 'Afinado del cifrado' },
  { id: 'metricas', label: 'Métrica a métrica' },
  { id: 'explorador', label: 'Explorador' },
  { id: 'metodos', label: 'Datos y métodos' },
]

const miles = (v) => Math.round(v).toLocaleString('es-ES')

export default function App() {
  const [comp, setComp] = useState(null)
  const [estado, setEstado] = useState(null)
  const [barrido, setBarrido] = useState([])
  const [error, setError] = useState(null)
  const [dsActivos, setDsActivos] = useState(null)
  const [metrica, setMetrica] = useState('accuracy')
  const [radarId, setRadarId] = useState(null)

  useEffect(() => {
    Promise.all([api.estado(), api.comparativa(), api.barrido()])
      .then(([est, c, b]) => {
        setEstado(est)
        setComp(c)
        setBarrido(b || [])
        setDsActivos(new Set(est.datasets))
        setRadarId(c[0]?.experimento_id)
      })
      .catch((e) => setError(e.message))
  }, [])

  const compF = useMemo(
    () => (comp && dsActivos ? comp.filter((r) => dsActivos.has(r.dataset_nombre)) : []),
    [comp, dsActivos],
  )

  if (error) return <Estado><b>No se pudieron cargar los datos.</b><br />{error}</Estado>
  if (!comp || !estado) return <Estado><div className="spinner" />Cargando resultados…</Estado>
  if (!estado.hay_datos) return <Estado>No hay base de datos de resultados todavía.<br />Ejecuta <code>ejecutar_experimentos.py</code>.</Estado>

  // --- Cifras de portada ---
  const nFieles = comp.filter((r) => Math.abs(r.delta_accuracy) <= 0.01).length
  const peorDelta = Math.max(...comp.map((r) => r.delta_accuracy))
  const maxRalent = Math.max(...comp.map((r) => r.factor_ralentizacion))
  const [mKey, mLabel, mFrac] = METRICAS.find(([k]) => k === metrica)
  const filaRadar = comp.find((r) => r.experimento_id === radarId) || comp[0]

  const toggleDs = (ds) => {
    const n = new Set(dsActivos)
    n.has(ds) ? n.delete(ds) : n.add(ds)
    if (n.size) setDsActivos(n)
  }

  return (
    <>
      <Toc items={SECCIONES} />

      <header className="topbar">
        <div className="brand">🔐 FHE <span className="dot">·</span> Coste de la privacidad</div>
        <nav>
          {SECCIONES.map((s) => <a key={s.id} href={`#${s.id}`}>{s.label}</a>)}
        </nav>
      </header>

      {/* ---------------- HERO ---------------- */}
      <section className="hero">
        <div className="wrap">
          <div className="eyebrow">Prueba de concepto · Cifrado homomórfico (CKKS / TenSEAL)</div>
          <h1>El coste de la privacidad</h1>
          <p className="lede">
            ¿Cuánto cuesta clasificar sin ver los datos? Evaluamos redes neuronales
            dos veces sobre el mismo test —<b> en claro</b> y sobre <b>datos cifrados</b>—
            y medimos la factura en precisión, tiempo y memoria.
          </p>
          <div className="byline">
            <b>{estado.n_experimentos}</b> experimentos · <b>{estado.datasets.length}</b> datasets
            {estado.n_configs_ckks > 0 && <> · <b>{estado.n_configs_ckks}</b> configuraciones de cifrado barridas</>}
          </div>
          <div className="actions">
            <a className="btn primary" href="#explorador">Explorar experimentos</a>
            <a className="btn" href="#fidelidad">Ver resultados</a>
          </div>
        </div>
      </section>

      {/* ---------------- BANDA DE CIFRAS ---------------- */}
      <section className="statband">
        <div className="wrap">
          <div className="statgrid">
            <Stat num={estado.n_experimentos} lbl="Experimentos" />
            <Stat num={`${nFieles}/${comp.length}`} lbl="Con fidelidad exacta" />
            <Stat num={`−${peorDelta.toFixed(1)} pp`} lbl="Peor pérdida de precisión" cost />
            <Stat num={`${miles(maxRalent)}×`} lbl="Ralentización máxima" cost />
          </div>
        </div>
      </section>

      {/* ---------------- RESUMEN / ABSTRACT ---------------- */}
      <Section id="resumen">
        <Head n={0} eyebrow="Resumen" title="La privacidad se paga en tiempo; a veces también en precisión">
        </Head>
        <div className="prose" style={{ marginTop: 18 }}>
          <p>
            El cifrado homomórfico <b>CKKS</b> permite calcular sobre datos cifrados sin
            descifrarlos, pero solo sabe <b>sumar y multiplicar</b>. La función de activación
            ReLU no es polinómica, así que en la red cifrada se sustituye por un
            <b> polinomio ajustado</b> por mínimos cuadrados al rango real de las
            pre-activaciones. Ese pequeño error residual se propaga capa a capa.
          </p>
          <p>
            El resultado es nítido: en redes <b>sin capas ocultas</b> no hay activación que
            aproximar y la inferencia cifrada da <b>exactamente</b> la misma precisión que la
            plana. En redes <b>profundas</b>, el error de la aproximación se amplifica y la
            precisión puede caer —hasta {peorDelta.toFixed(0)} puntos en el peor caso. El coste
            temporal, en cambio, es enorme <i>siempre</i>: entre miles y cientos de miles de
            veces más lento.
          </p>
        </div>

        <div className="findings">
          <Finding cls="f-plana" k={`${nFieles}/${comp.length}`} h="Fidelidad exacta">
            En redes de una sola capa la clase predicha cifrada coincide con la plana en todas
            las muestras: cifrar no cuesta ni un acierto.
          </Finding>
          <Finding cls="f-homo" k={`${miles(maxRalent)}×`} h="Más lento">
            La factura real es el tiempo: de centésimas de segundo en claro a decenas de minutos
            cifrado. El throughput cae de millones a unidades por segundo.
          </Finding>
          <Finding cls="f-cost" k={`−${peorDelta.toFixed(1)} pp`} h="Precisión perdida">
            En las redes profundas, la ReLU aproximada degrada la precisión. La profundidad —no el
            dataset— es lo que predice la caída.
          </Finding>
        </div>
      </Section>

      {/* ---------------- §1 FIDELIDAD ---------------- */}
      <Section id="fidelidad" alt>
        <Head n={1} eyebrow="Fidelidad" title="¿Cifrar cuesta precisión?">
          Cada punto es un experimento. En la diagonal, la precisión cifrada iguala a la plana;
          por debajo, se pierde. El tamaño del punto es la profundidad de la red.
        </Head>
        <DatasetFilter datasets={estado.datasets} activos={dsActivos} toggle={toggleDs} />
        <Figure caption={<>Precisión <b>plana</b> frente a <b>homomórfica</b>. Los puntos sobre la diagonal son fidelidad perfecta; cuanto más abajo, más precisión cuesta la privacidad.</>}>
          <EChart height={440} option={fidelidad(compF)} />
        </Figure>
        <Figure caption={<>Puntos porcentuales de <b>accuracy</b> perdidos al cifrar, coloreados por número de capas ocultas. Las barras altas son siempre redes profundas.</>}>
          <EChart height={420} option={deltaPrecision(compF)} />
        </Figure>
      </Section>

      {/* ---------------- §2 COSTE TEMPORAL ---------------- */}
      <Section id="tiempo">
        <Head n={2} eyebrow="Coste temporal" title="La factura se paga en segundos">
          La inferencia cifrada es entre 2.000 y 150.000 veces más lenta. Ojo a la escala
          logarítmica: cada línea de la cuadrícula multiplica por diez.
        </Head>
        <DatasetFilter datasets={estado.datasets} activos={dsActivos} toggle={toggleDs} />
        <div className="two-col" style={{ maxWidth: 'var(--fig)', margin: '0 auto' }}>
          <Figure caption={<>Tiempo de inferencia, <b>escala log</b>.</>}>
            <EChart height={420} option={tiempo(compF)} />
          </Figure>
          <Figure caption={<>Factor de ralentización frente a la inferencia en claro.</>}>
            <EChart height={420} option={ralentizacion(compF)} />
          </Figure>
        </div>
        <Figure caption={<>El tiempo crece con el <b>número de muestras</b> y con la <b>profundidad</b> (tamaño del punto): cada capa añade multiplicaciones cifradas por muestra.</>}>
          <EChart height={440} option={tiempoVsMuestras(compF)} />
        </Figure>
      </Section>

      {/* ---------------- §3 RECURSOS ---------------- */}
      <Section id="recursos" alt>
        <Head n={3} eyebrow="Recursos" title="La memoria también se multiplica">
          Cifrar un vector lo convierte en un polinomio enorme. La huella de RAM pasa de medio
          gigabyte a varios; por eso la batería aísla cada experimento en un subproceso.
        </Head>
        <DatasetFilter datasets={estado.datasets} activos={dsActivos} toggle={toggleDs} />
        <Figure caption={<>RAM pico del proceso: <b>plana</b> frente a <b>homomórfica</b>.</>}>
          <EChart height={420} option={ram(compF)} />
        </Figure>
      </Section>

      {/* ---------------- §4 AFINADO DEL CIFRADO ---------------- */}
      {barrido.length > 0 && (
        <Section id="cifrado">
          <Head n={4} eyebrow="Afinado del cifrado" title="No todos los parámetros del cifrado importan igual">
            CKKS solo suma y multiplica, así que la ReLU se sustituye por un polinomio ajustado
            al <b>rango real</b> de las pre-activaciones. Barriendo el grado del polinomio y la
            escala sobre {new Set(barrido.map((r) => r.nombre)).size} redes
            ({barrido.length} configuraciones de cifrado) se ve qué mueve la aguja.
          </Head>

          <Figure caption={<>Precisión cifrada según el <b>grado del polinomio</b>, con la precisión en claro (rombo) como referencia. Al ajustar el polinomio al rango real, subir el grado mejora la aproximación de la ReLU.</>}>
            <EChart height={430} option={barridoGrado(barrido)} />
          </Figure>

          <Figure caption={<>Lo que cuesta ese grado: cada activación de grado <i>d</i> consume ⌈log₂ d⌉ niveles multiplicativos, lo que alarga la cadena de módulos y el tiempo. <b>Escala logarítmica.</b></>}>
            <EChart height={420} option={barridoCoste(barrido)} />
          </Figure>

          <Figure caption={<>La <b>escala global</b>, en cambio, es indiferente para la precisión: las líneas son planas. Solo cambia el coste y la viabilidad del cifrado.</>}>
            <EChart height={420} option={barridoEscala(barrido)} />
          </Figure>

          <div className="method" style={{ marginTop: 10 }}>
            <div className="callout">
              <b>Por qué no basta con [−1, 1]:</b> aunque los pesos de la red rondan [−1, 1], la
              pre-activación <span style={{ fontFamily: 'var(--mono)' }}>z = Σwᵢxᵢ + b</span> suma
              sobre decenas de entradas estandarizadas, así que alcanza valores de ±8 a ±15.
              Un polinomio ajustado solo a [−1, 1] se evaluaría muy fuera de su zona de ajuste.
              En el <a href="#explorador">explorador</a> puedes ver, para cada configuración, el
              polinomio dibujado sobre la ReLU real y su error.
            </div>
          </div>
        </Section>
      )}

      {/* ---------------- §5 MÉTRICA A MÉTRICA ---------------- */}
      <Section id="metricas" alt>
        <Head n={5} eyebrow="Métrica a métrica" title="Elige la métrica y compara">
          Todas las métricas de clasificación, plana frente a homomórfica. ROC-AUC y log-loss
          no aparecen en la cifrada: solo se descifra la clase ganadora, nunca las probabilidades.
        </Head>
        <div className="controls">
          <span className="ctl-label">Métrica</span>
          <select className="ui" value={metrica} onChange={(e) => setMetrica(e.target.value)}>
            {METRICAS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </select>
        </div>
        <Figure caption={<>Comparativa de <b>{mLabel}</b> por experimento.</>}>
          <EChart height={440} option={barrasComparativa(compF, mKey, mLabel, mFrac)} />
        </Figure>

        <div className="controls" style={{ marginTop: 24 }}>
          <span className="ctl-label">Perfil (radar)</span>
          <select className="ui" value={radarId ?? ''} onChange={(e) => setRadarId(+e.target.value)}>
            {comp.map((r) => <option key={r.experimento_id} value={r.experimento_id}>{r.nombre}</option>)}
          </select>
        </div>
        <Figure caption={<>Perfil de métricas de <b>{filaRadar.nombre}</b>: cuanto más se encoge el naranja respecto al azul, más cuesta el cifrado.</>}>
          <EChart height={400} option={radar(filaRadar)} />
        </Figure>
      </Section>

      {/* ---------------- §5 EXPLORADOR ---------------- */}
      <Section id="explorador">
        <Head n={6} eyebrow="Explorador" title="Radiografía de un experimento">
          Elige cualquier experimento y compara sus matrices de confusión, métricas por clase,
          entrenamiento por semilla y parámetros de cifrado.
        </Head>
        <ExperimentExplorer comp={comp} />
      </Section>

      {/* ---------------- §6 DATOS Y MÉTODOS ---------------- */}
      <Section id="metodos">
        <Head n={7} eyebrow="Datos y métodos" title="Cómo se obtuvieron estos números">
        </Head>
        <div className="method">
          <p>
            Cada experimento entrena una red <code>ConfigurableNN</code> con validación cruzada
            Monte Carlo, se queda con el mejor modelo y lo evalúa sobre un test fijo
            dos veces: en claro y cifrado. El split de test usa una semilla maestra fija, así que
            ambas modalidades ven exactamente las mismas muestras.
          </p>
          <div className="callout">
            <b>Cifrado CKKS:</b> la cadena de módulos y el <span style={{ fontFamily: 'var(--mono)' }}>poly_modulus_degree</span> se
            derivan de la <b>profundidad multiplicativa</b> de cada red (una multiplicación por
            capa lineal más ⌈log₂ d⌉ por cada activación de grado <i>d</i>), eligiendo el menor
            módulo que mantenga 128 bits de seguridad. La ReLU se aproxima por un polinomio
            ajustado al rango real de las pre-activaciones, medido en una pasada en claro. La
            predicción homomórfica solo descifra la clase ganadora, por eso no expone ROC-AUC ni
            log-loss. Todas las cifras salen de la misma base de datos SQLite; esta web la lee en
            modo solo-lectura.
          </div>
        </div>
        <h4 style={{ fontFamily: 'var(--sans)', textAlign: 'center', margin: '34px 0 8px' }}>Tablas de datos</h4>
        <p className="caption" style={{ marginBottom: 18 }}>Despliega cualquier tabla para inspeccionarla o descargarla en CSV.</p>
        <DataTables />
      </Section>

      {/* ---------------- CITA / FOOTER ---------------- */}
      <section className="statband citeband">
        <div className="wrap">
          <div className="citeblock">
            <h3>Sobre este trabajo</h3>
            <p style={{ fontFamily: 'var(--sans)', fontSize: 14.5, lineHeight: 1.6 }}>
              Prueba de concepto académica sobre el coste de la inferencia con privacidad mediante
              cifrado homomórfico. Frontend en React + ECharts; backend en FastAPI leyendo la base
              de datos de resultados. No es una implementación lista para uso clínico.
            </p>
            <pre>{`@misc{fhe_coste_privacidad,
  title  = {El coste de la privacidad: inferencia neuronal
            con cifrado homomorfico (CKKS)},
  note   = {Prueba de concepto academica},
  year   = {2026}
}`}</pre>
            <div className="footnote">
              Datos leídos de <code>resultados/resultados.db</code> · {estado.n_experimentos} experimentos, {estado.n_predicciones} predicciones
            </div>
          </div>
        </div>
      </section>
    </>
  )
}

function Stat({ num, lbl, cost }) {
  return (
    <div>
      <div className={`num${cost ? ' cost' : ''}`}>{num}</div>
      <div className="lbl">{lbl}</div>
    </div>
  )
}

function Finding({ cls, k, h, children }) {
  return (
    <div className={`finding ${cls}`}>
      <div className="k">{k}</div>
      <h4>{h}</h4>
      <p>{children}</p>
    </div>
  )
}

function DatasetFilter({ datasets, activos, toggle }) {
  return (
    <div className="controls">
      <span className="ctl-label">Datasets</span>
      <div className="pills">
        {datasets.map((ds) => (
          <button key={ds} className={`pill${activos.has(ds) ? ' on' : ''}`} onClick={() => toggle(ds)}>{ds}</button>
        ))}
      </div>
    </div>
  )
}

function Estado({ children }) {
  return <div className="state">{children}</div>
}
