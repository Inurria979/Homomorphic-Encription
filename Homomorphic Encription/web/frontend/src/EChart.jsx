import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

// Wrapper mínimo sobre ECharts para React 19: inicializa una vez, reaplica las
// opciones cuando cambian (con animación) y se redimensiona con su contenedor.
export default function EChart({ option, height = 380, style }) {
  const ref = useRef(null)
  const inst = useRef(null)

  useEffect(() => {
    inst.current = echarts.init(ref.current, null, { renderer: 'canvas' })
    const ro = new ResizeObserver(() => inst.current && inst.current.resize())
    ro.observe(ref.current)
    return () => { ro.disconnect(); inst.current.dispose() }
  }, [])

  useEffect(() => {
    if (inst.current && option) inst.current.setOption(option, { notMerge: true, lazyUpdate: true })
  }, [option])

  return <div ref={ref} className="chart" style={{ height, ...style }} />
}
