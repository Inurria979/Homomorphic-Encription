// Constructores de opciones ECharts. Cada función recibe datos ya cargados y
// devuelve un objeto `option`. Sin estado ni React: fáciles de razonar y probar.

import { C, base, ejeSans, colorProfundidad, colorGrado } from './theme'

const f1 = (v) => (v == null ? '—' : (+v).toFixed(1))
const f2 = (v) => (v == null ? '—' : (+v).toFixed(2))
const miles = (v) => (v == null ? '—' : Math.round(v).toLocaleString('es-ES'))

// ---------------------------------------------------------------- Fidelidad
export function fidelidad(comp) {
  const datasets = [...new Set(comp.map((d) => d.dataset_nombre))]
  const lo = Math.max(0, Math.min(...comp.map((d) => Math.min(d.accuracy_plana, d.accuracy_homo))) - 3)

  const series = datasets.map((ds, i) => ({
    name: ds,
    type: 'scatter',
    color: C.datasets[i % C.datasets.length],
    symbolSize: (val) => 12 + val[2] * 6,
    emphasis: { focus: 'series', scale: 1.25 },
    itemStyle: { opacity: 0.85, borderColor: '#fff', borderWidth: 1.5 },
    data: comp.filter((d) => d.dataset_nombre === ds).map((d) => ({
      value: [d.accuracy_plana, d.accuracy_homo, d.profundidad],
      dir: d.nombre, arq: d.arquitectura, delta: d.delta_accuracy,
    })),
  }))

  // Diagonal y=x (fidelidad perfecta).
  series.push({
    name: 'Fidelidad perfecta', type: 'line', silent: true, symbol: 'none',
    lineStyle: { color: C.muted, type: 'dashed', width: 1 },
    data: [[lo, lo], [100, 100]], tooltip: { show: false }, z: 0,
  })

  return {
    ...base(),
    legend: { ...base().legend, data: datasets },
    tooltip: {
      ...base().tooltip,
      formatter: (p) => {
        if (p.seriesName === 'Fidelidad perfecta') return ''
        const d = p.data
        return `<b>${d.dir}</b><br/>Arquitectura: ${d.arq}<br/>` +
          `Plana: <b>${f2(d.value[0])}%</b><br/>Homomórfica: <b>${f2(d.value[1])}%</b><br/>` +
          `<span style="color:${C.cost}">Pérdida: ${f2(d.delta)} pp</span>`
      },
    },
    xAxis: { ...ejeSans, type: 'value', name: 'Accuracy plana (%)', min: lo, max: 101, nameLocation: 'middle', nameGap: 30 },
    yAxis: { ...ejeSans, type: 'value', name: 'Accuracy homomórfica (%)', min: lo, max: 101, nameLocation: 'middle', nameGap: 42 },
    series,
  }
}

// ------------------------------------------------------- Pérdida por cifrado
export function deltaPrecision(comp) {
  const d = [...comp].sort((a, b) => b.delta_accuracy - a.delta_accuracy)
  return {
    ...base(),
    grid: { ...base().grid, bottom: 90 },
    tooltip: {
      ...base().tooltip, trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: (ps) => {
        const it = ps[0]; const r = d[it.dataIndex]
        return `<b>${r.nombre}</b><br/>Arquitectura: ${r.arquitectura}<br/>` +
          `Capas ocultas: ${r.profundidad}<br/>` +
          `<span style="color:${C.cost}">Precisión perdida: ${f2(r.delta_accuracy)} pp</span>`
      },
    },
    xAxis: { ...ejeSans, type: 'category', data: d.map((r) => r.nombre), axisLabel: { ...ejeSans.axisLabel, rotate: 40, interval: 0 } },
    yAxis: { ...ejeSans, type: 'value', name: 'Precisión perdida (pp)' },
    series: [{
      type: 'bar', barMaxWidth: 40,
      data: d.map((r) => ({ value: r.delta_accuracy, itemStyle: { color: colorProfundidad(r.profundidad), borderRadius: [4, 4, 0, 0] } })),
    }],
  }
}

// ----------------------------------------------- Grupo plana vs homomórfica
function agrupadas(comp, keyP, keyH, nombreEjeY, { log = false } = {}) {
  return {
    ...base(),
    grid: { ...base().grid, bottom: 90 },
    legend: { ...base().legend, data: ['Plana', 'Homomórfica'] },
    tooltip: { ...base().tooltip, trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { ...ejeSans, type: 'category', data: comp.map((r) => r.nombre), axisLabel: { ...ejeSans.axisLabel, rotate: 40, interval: 0 } },
    yAxis: { ...ejeSans, type: log ? 'log' : 'value', name: nombreEjeY, min: log ? undefined : 0 },
    series: [
      { name: 'Plana', type: 'bar', color: C.plana, barMaxWidth: 26, itemStyle: { borderRadius: [3, 3, 0, 0] }, data: comp.map((r) => r[keyP]) },
      { name: 'Homomórfica', type: 'bar', color: C.homo, barMaxWidth: 26, itemStyle: { borderRadius: [3, 3, 0, 0] }, data: comp.map((r) => r[keyH]) },
    ],
  }
}

export function tiempo(comp) {
  const o = agrupadas(comp, 'tiempo_wall_s_plana', 'tiempo_wall_s_homo', 'Tiempo (s, escala log)', { log: true })
  o.tooltip.formatter = (ps) => {
    const r = comp[ps[0].dataIndex]
    return `<b>${r.nombre}</b><br/>Plana: <b>${f2(r.tiempo_wall_s_plana)} s</b><br/>` +
      `Homomórfica: <b>${miles(r.tiempo_wall_s_homo)} s</b>`
  }
  return o
}

export function ram(comp) {
  const o = agrupadas(comp, 'ram_pico_proceso_gb_plana', 'ram_pico_proceso_gb_homo', 'RAM pico del proceso (GB)')
  o.tooltip.formatter = (ps) => {
    const r = comp[ps[0].dataIndex]
    return `<b>${r.nombre}</b><br/>Plana: <b>${f2(r.ram_pico_proceso_gb_plana)} GB</b><br/>` +
      `Homomórfica: <b>${f2(r.ram_pico_proceso_gb_homo)} GB</b>`
  }
  return o
}

export function barrasComparativa(comp, metric, label, esFraccion) {
  const dec = esFraccion ? f2 : f1
  const o = agrupadas(comp, `${metric}_plana`, `${metric}_homo`, label)
  o.tooltip.formatter = (ps) => {
    const r = comp[ps[0].dataIndex]
    const p = r[`${metric}_plana`], h = r[`${metric}_homo`]
    return `<b>${r.nombre}</b><br/>Plana: <b>${dec(p)}</b><br/>Homomórfica: <b>${dec(h)}</b>`
  }
  return o
}

// ------------------------------------------------------ Factor ralentización
export function ralentizacion(comp) {
  const d = [...comp].sort((a, b) => a.factor_ralentizacion - b.factor_ralentizacion)
  const max = Math.max(...d.map((r) => r.factor_ralentizacion))
  return {
    ...base(),
    grid: { ...base().grid, left: 8, right: 40, bottom: 40 },
    tooltip: {
      ...base().tooltip, trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: (ps) => `<b>${d[ps[0].dataIndex].nombre}</b><br/>` +
        `<span style="color:${C.cost}">${miles(ps[0].value)}× más lento</span>`,
    },
    xAxis: { ...ejeSans, type: 'log', name: 'Veces más lento que en claro (log)', nameLocation: 'middle', nameGap: 34 },
    yAxis: { ...ejeSans, type: 'category', data: d.map((r) => r.nombre), splitLine: { show: false } },
    series: [{
      type: 'bar', barMaxWidth: 18,
      data: d.map((r) => {
        const t = r.factor_ralentizacion / max
        const col = `rgba(192,57,43,${0.35 + 0.6 * t})`
        return { value: r.factor_ralentizacion, itemStyle: { color: col, borderRadius: [0, 4, 4, 0] } }
      }),
    }],
  }
}

// --------------------------------------------------- Tiempo vs nº de muestras
export function tiempoVsMuestras(comp) {
  const datasets = [...new Set(comp.map((d) => d.dataset_nombre))]
  return {
    ...base(),
    legend: { ...base().legend, data: datasets },
    tooltip: {
      ...base().tooltip,
      formatter: (p) => {
        const d = p.data
        return `<b>${d.dir}</b> · ${d.arq}<br/>Muestras: <b>${miles(d.value[0])}</b><br/>` +
          `Tiempo: <b>${miles(d.value[1])} s</b>`
      },
    },
    xAxis: { ...ejeSans, type: 'log', name: 'Muestras de test (log)', nameLocation: 'middle', nameGap: 30 },
    yAxis: { ...ejeSans, type: 'log', name: 'Tiempo homomórfico (s, log)', nameLocation: 'middle', nameGap: 46 },
    series: datasets.map((ds, i) => ({
      name: ds, type: 'scatter', color: C.datasets[i % C.datasets.length],
      symbolSize: (val) => 12 + val[2] * 6,
      itemStyle: { opacity: 0.85, borderColor: '#fff', borderWidth: 1.5 },
      emphasis: { focus: 'series', scale: 1.25 },
      data: comp.filter((d) => d.dataset_nombre === ds).map((d) => ({
        value: [d.n_muestras, d.tiempo_wall_s_homo, d.profundidad], dir: d.nombre, arq: d.arquitectura,
      })),
    })),
  }
}

// ------------------------------------------------------------------- Radar
export function radar(fila) {
  const ejes = [
    ['accuracy', 'Accuracy'], ['f1_macro', 'F1 macro'], ['precision_macro', 'Precision'],
    ['recall_macro', 'Recall'], ['balanced_accuracy', 'Bal. acc'],
  ]
  return {
    ...base(),
    tooltip: { ...base().tooltip },
    legend: { ...base().legend, data: ['Plana', 'Homomórfica'] },
    radar: {
      indicator: ejes.map(([, n]) => ({ name: n, max: 100 })),
      radius: '65%', splitNumber: 4,
      axisName: { color: C.soft, fontFamily: "'Inter',sans-serif", fontSize: 12 },
      splitLine: { lineStyle: { color: C.line } },
      splitArea: { areaStyle: { color: ['#fff', '#fafbfd'] } },
      axisLine: { lineStyle: { color: C.line } },
    },
    series: [{
      type: 'radar', symbolSize: 5,
      data: [
        { name: 'Plana', value: ejes.map(([k]) => fila[`${k}_plana`]), itemStyle: { color: C.plana }, areaStyle: { color: 'rgba(47,111,219,0.15)' } },
        { name: 'Homomórfica', value: ejes.map(([k]) => fila[`${k}_homo`]), itemStyle: { color: C.homo }, areaStyle: { color: 'rgba(232,131,58,0.18)' } },
      ],
    }],
  }
}

// -------------------------------------------------------- Matriz de confusión
export function matrizConfusion(matriz, clases, color) {
  if (!matriz) return null
  const data = []
  let max = 0
  for (let i = 0; i < matriz.length; i++) {
    for (let j = 0; j < matriz[i].length; j++) {
      data.push([j, i, matriz[i][j]])
      if (matriz[i][j] > max) max = matriz[i][j]
    }
  }
  const etq = clases.map((c) => String(c))
  return {
    ...base(),
    grid: { left: 44, right: 16, top: 30, bottom: 46, containLabel: true },
    tooltip: {
      ...base().tooltip,
      formatter: (p) => `Real <b>${p.data[1]}</b> → Predicho <b>${p.data[0]}</b><br/>${p.data[2]} muestras`,
    },
    xAxis: { type: 'category', data: etq, name: 'Predicho', nameLocation: 'middle', nameGap: 28, splitArea: { show: true }, axisLabel: { color: C.soft, fontFamily: "'Inter',sans-serif" }, axisLine: { lineStyle: { color: C.lineStrong } }, axisTick: { show: false } },
    yAxis: { type: 'category', data: etq, name: 'Real', inverse: true, splitArea: { show: true }, axisLabel: { color: C.soft, fontFamily: "'Inter',sans-serif" }, axisLine: { lineStyle: { color: C.lineStrong } }, axisTick: { show: false } },
    visualMap: { min: 0, max, show: false, inRange: { color: ['#ffffff', color] } },
    series: [{
      type: 'heatmap', data,
      label: { show: true, fontFamily: "'Inter',sans-serif", fontSize: 11, formatter: (p) => p.data[2], color: '#111' },
      itemStyle: { borderColor: '#fff', borderWidth: 2 },
      emphasis: { itemStyle: { shadowBlur: 6, shadowColor: 'rgba(0,0,0,0.2)' } },
    }],
  }
}

// ------------------------------------------------------- Métricas por clase
export function metricasClase(rows, metricKey, label) {
  const clases = [...new Set(rows.map((r) => r.clase))].sort((a, b) => a - b)
  const serie = (tipo, color, nombre) => ({
    name: nombre, type: 'bar', color, barMaxWidth: 26, itemStyle: { borderRadius: [3, 3, 0, 0] },
    data: clases.map((c) => {
      const r = rows.find((x) => x.clase === c && x.tipo === tipo)
      return r ? r[metricKey] : null
    }),
  })
  return {
    ...base(),
    legend: { ...base().legend, data: ['Plana', 'Homomórfica'] },
    tooltip: { ...base().tooltip, trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => (v == null ? '—' : `${f1(v)}`) },
    xAxis: { ...ejeSans, type: 'category', data: clases.map((c) => `Clase ${c}`) },
    yAxis: { ...ejeSans, type: 'value', name: label, max: 100 },
    series: [serie('plana', C.plana, 'Plana'), serie('homomorfica', C.homo, 'Homomórfica')],
  }
}

// -------------------------------------------------------------- Semillas
export function semillas(rows) {
  return {
    ...base(),
    legend: { ...base().legend, data: ['Val accuracy (%)', 'Val loss'] },
    tooltip: { ...base().tooltip, trigger: 'axis', axisPointer: { type: 'cross' } },
    xAxis: { ...ejeSans, type: 'category', data: rows.map((r) => `#${r.semilla}`) },
    yAxis: [
      { ...ejeSans, type: 'value', name: 'Val acc (%)' },
      { ...ejeSans, type: 'value', name: 'Val loss', splitLine: { show: false }, position: 'right' },
    ],
    series: [
      { name: 'Val accuracy (%)', type: 'bar', color: C.plana, barMaxWidth: 40, itemStyle: { borderRadius: [3, 3, 0, 0] }, data: rows.map((r) => r.val_accuracy) },
      { name: 'Val loss', type: 'line', yAxisIndex: 1, color: C.cost, symbolSize: 7, smooth: true, data: rows.map((r) => r.val_loss) },
    ],
  }
}

// =====================================================================
// BARRIDO DE PARÁMETROS CKKS
// Los experimentos barridos tienen VARIAS predicciones homomórficas, una por
// configuración de cifrado (grado del polinomio × escala). Estas gráficas
// responden: ¿qué parámetro del cifrado importa de verdad?
// =====================================================================

// Agrupa las filas del barrido por red (experimento).
function porRed(barrido) {
  const mapa = new Map()
  barrido.forEach((r) => {
    if (!mapa.has(r.nombre)) mapa.set(r.nombre, [])
    mapa.get(r.nombre).push(r)
  })
  return mapa
}

// ------------------------------ Precisión según el grado del polinomio
export function barridoGrado(barrido) {
  const redes = [...porRed(barrido).keys()]
  const grados = [...new Set(barrido.map((r) => r.grado))].sort((a, b) => a - b)

  // Una serie por grado: para cada red, la accuracy media de ese grado.
  const series = grados.map((g) => ({
    name: `Grado ${g}`,
    type: 'bar',
    color: colorGrado(g),
    barMaxWidth: 34,
    itemStyle: { borderRadius: [4, 4, 0, 0] },
    data: redes.map((red) => {
      const filas = barrido.filter((r) => r.nombre === red && r.grado === g)
      if (!filas.length) return null
      return +(filas.reduce((s, r) => s + r.accuracy, 0) / filas.length).toFixed(4)
    }),
  }))

  // Línea de referencia: la precisión SIN cifrar de cada red.
  series.push({
    name: 'Plana (sin cifrar)',
    type: 'line',
    color: C.cost,
    symbol: 'diamond',
    symbolSize: 11,
    lineStyle: { type: 'dashed', width: 1.5 },
    data: redes.map((red) => barrido.find((r) => r.nombre === red)?.accuracy_plana ?? null),
  })

  const min = Math.max(0, Math.min(...barrido.map((r) => r.accuracy)) - 3)
  return {
    ...base(),
    legend: { ...base().legend, data: [...grados.map((g) => `Grado ${g}`), 'Plana (sin cifrar)'] },
    grid: { ...base().grid, bottom: 80 },
    tooltip: { ...base().tooltip, trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { ...ejeSans, type: 'category', data: redes, axisLabel: { ...ejeSans.axisLabel, rotate: 22, interval: 0 } },
    yAxis: { ...ejeSans, type: 'value', name: 'Accuracy (%)', min },
    series,
  }
}

// ------------------------------ Coste temporal de subir el grado
export function barridoCoste(barrido) {
  const redes = [...porRed(barrido).keys()]
  const grados = [...new Set(barrido.map((r) => r.grado))].sort((a, b) => a - b)

  return {
    ...base(),
    legend: { ...base().legend, data: grados.map((g) => `Grado ${g}`) },
    // `left` amplio: el nombre del eje Y va rotado y con `left: 8` cae fuera del lienzo.
    grid: { ...base().grid, bottom: 80, left: 58 },
    tooltip: {
      ...base().tooltip, trigger: 'axis', axisPointer: { type: 'shadow' },
      valueFormatter: (v) => (v == null ? '—' : `${f1(v)} s`),
    },
    xAxis: { ...ejeSans, type: 'category', data: redes, axisLabel: { ...ejeSans.axisLabel, rotate: 22, interval: 0 } },
    // El nombre va centrado y rotado: si se deja arriba (por defecto) un texto
    // largo se sale del lienzo por la izquierda.
    yAxis: {
      ...ejeSans, type: 'log', name: 'Tiempo cifrado (s)',
      nameLocation: 'middle', nameRotate: 90, nameGap: 48,
    },
    series: grados.map((g) => ({
      name: `Grado ${g}`,
      type: 'bar',
      color: colorGrado(g),
      barMaxWidth: 34,
      itemStyle: { borderRadius: [4, 4, 0, 0] },
      data: redes.map((red) => {
        const filas = barrido.filter((r) => r.nombre === red && r.grado === g)
        if (!filas.length) return null
        return +(filas.reduce((s, r) => s + r.tiempo_wall_s, 0) / filas.length).toFixed(2)
      }),
    })),
  }
}

// ------------------------------ ¿Importa la escala? (spoiler: no, en precisión)
export function barridoEscala(barrido) {
  const escalas = [...new Set(barrido.map((r) => r.global_scale_bits))].sort((a, b) => a - b)
  const redes = [...porRed(barrido).keys()]

  // Para cada red+grado, una línea a lo largo de las escalas.
  const series = []
  redes.forEach((red) => {
    const grados = [...new Set(barrido.filter((r) => r.nombre === red).map((r) => r.grado))].sort()
    grados.forEach((g) => {
      const datos = escalas.map((s) => {
        const f = barrido.find((r) => r.nombre === red && r.grado === g && r.global_scale_bits === s)
        return f ? f.accuracy : null
      })
      if (datos.every((d) => d == null)) return
      series.push({
        // Nombre corto: con 8 series el nombre completo desborda la leyenda.
        name: `${red.replace(/^Cáncer /, '').replace(/ · barrido$/, '')} · g${g}`,
        type: 'line',
        color: colorGrado(g),
        symbolSize: 9,
        connectNulls: true,
        lineStyle: { width: 2 },
        data: datos,
      })
    })
  })

  const min = Math.max(0, Math.min(...barrido.map((r) => r.accuracy)) - 3)
  return {
    ...base(),
    legend: { ...base().legend, type: 'scroll', top: 0 },
    grid: { ...base().grid, top: 60 },
    tooltip: { ...base().tooltip, trigger: 'axis', valueFormatter: (v) => (v == null ? '—' : `${f2(v)}%`) },
    xAxis: { ...ejeSans, type: 'category', data: escalas.map((s) => `2^${s}`), name: 'Escala global', nameLocation: 'middle', nameGap: 30 },
    yAxis: { ...ejeSans, type: 'value', name: 'Accuracy (%)', min },
    series,
  }
}

// ------------------------------ El polinomio ajustado frente a la ReLU real
// Es el corazón del método: se dibuja la ReLU exacta y el polinomio que la
// aproxima, SOBRE EL RANGO REAL de las pre-activaciones (no [-1,1]).
export function polinomioVsRelu(rango, coeficientes, grado) {
  if (!rango || rango.length !== 2 || !coeficientes || !coeficientes.length) return null
  const [a, b] = rango
  const N = 160
  const xs = Array.from({ length: N }, (_, i) => a + ((b - a) * i) / (N - 1))
  // Horner sobre la base de potencias ascendente [c0, c1, ...] (igual que polyval).
  const evalPoly = (x) => coeficientes.reduceRight((acc, c) => acc * x + c, 0)

  const relu = xs.map((x) => [x, Math.max(x, 0)])
  const poly = xs.map((x) => [x, evalPoly(x)])
  const err = xs.map((x) => [x, Math.abs(evalPoly(x) - Math.max(x, 0))])

  return {
    ...base(),
    legend: { ...base().legend, data: ['ReLU exacta', `Polinomio grado ${grado}`, 'Error absoluto'] },
    // Hueco arriba para que la leyenda no pise los ejes, y a los lados para los
    // nombres rotados de los dos ejes Y (salida y error).
    grid: { ...base().grid, top: 62, bottom: 46, left: 54, right: 58 },
    tooltip: { ...base().tooltip, trigger: 'axis', axisPointer: { type: 'cross' }, valueFormatter: (v) => f2(v) },
    xAxis: {
      ...ejeSans, type: 'value', name: 'Pre-activación (rango real medido)',
      nameLocation: 'middle', nameGap: 30, min: a, max: b,
      // Sin formatear, los extremos del rango salen con 15 decimales.
      axisLabel: { ...ejeSans.axisLabel, formatter: (v) => (+v).toFixed(1) },
    },
    yAxis: [
      { ...ejeSans, type: 'value', name: 'Salida', nameLocation: 'middle', nameRotate: 90, nameGap: 38 },
      {
        ...ejeSans, type: 'value', name: 'Error', position: 'right', splitLine: { show: false },
        nameLocation: 'middle', nameRotate: 90, nameGap: 46,
        axisLabel: { ...ejeSans.axisLabel, formatter: (v) => (+v).toFixed(2) },
      },
    ],
    series: [
      { name: 'ReLU exacta', type: 'line', color: C.ink, symbol: 'none', lineStyle: { width: 2.5 }, data: relu },
      { name: `Polinomio grado ${grado}`, type: 'line', color: colorGrado(grado), symbol: 'none', lineStyle: { width: 2.5, type: 'dashed' }, data: poly },
      {
        name: 'Error absoluto', type: 'line', yAxisIndex: 1, color: C.cost, symbol: 'none',
        lineStyle: { width: 1 }, areaStyle: { opacity: 0.12, color: C.cost }, data: err,
      },
    ],
  }
}
