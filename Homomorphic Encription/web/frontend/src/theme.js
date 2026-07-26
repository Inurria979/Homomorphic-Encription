// Paleta y estilos base compartidos por todas las gráficas ECharts.
// Azul = plana (barato); naranja = homomórfica (el coste de la privacidad).

export const C = {
  plana: '#2f6fdb',
  homo: '#e8833a',
  cost: '#c0392b',
  ink: '#16202c',
  soft: '#47535f',
  muted: '#8a949e',
  line: '#e7ebf0',
  lineStrong: '#d4dae1',
  // Paleta cualitativa para datasets.
  datasets: ['#2f6fdb', '#e8833a', '#3a9e6f', '#9b59b6', '#e6b800'],
}

export const FONT = "'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif"

// Escala de color por profundidad de red (0 = claro, 3 = intenso): cuenta la
// historia de que la profundidad amplifica el error de la Taylor-ReLU.
export function colorProfundidad(d) {
  const escala = ['#9ec2f2', '#f0b27a', '#e8833a', '#c0392b']
  return escala[Math.min(d, escala.length - 1)]
}

// Color por grado del polinomio que aproxima la ReLU (3, 5, 7...): a mayor grado,
// mejor aproximación pero más profundidad multiplicativa (y más tiempo).
export const COLOR_GRADO = { 2: '#9b59b6', 3: '#2f6fdb', 5: '#3a9e6f', 7: '#e8833a' }
export function colorGrado(g) {
  return COLOR_GRADO[g] || C.muted
}

export function base() {
  return {
    textStyle: { fontFamily: FONT, color: C.soft },
    animationDuration: 600,
    animationEasing: 'cubicOut',
    grid: { left: 8, right: 20, top: 40, bottom: 30, containLabel: true },
    tooltip: {
      backgroundColor: 'rgba(255,255,255,0.98)',
      borderColor: C.line,
      borderWidth: 1,
      padding: [10, 14],
      textStyle: { color: C.ink, fontFamily: FONT, fontSize: 13 },
      extraCssText: 'box-shadow:0 6px 24px rgba(16,32,52,0.12);border-radius:10px;',
    },
    legend: {
      top: 4,
      left: 'center',
      icon: 'roundRect',
      itemWidth: 12,
      itemHeight: 12,
      textStyle: { fontFamily: FONT, color: C.soft, fontSize: 13 },
    },
  }
}

export const ejeSans = {
  axisLine: { lineStyle: { color: C.lineStrong } },
  axisTick: { show: false },
  axisLabel: { color: C.soft, fontFamily: FONT, fontSize: 12 },
  nameTextStyle: { color: C.muted, fontFamily: FONT, fontSize: 12.5, fontWeight: 600 },
  splitLine: { lineStyle: { color: C.line, type: 'dashed' } },
}
