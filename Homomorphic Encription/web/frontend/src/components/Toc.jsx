import { useEffect, useState } from 'react'

// Índice lateral con resaltado de la sección visible (scroll-spy).
export default function Toc({ items }) {
  const [activo, setActivo] = useState(items[0]?.id)

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entradas) => {
        entradas.forEach((e) => { if (e.isIntersecting) setActivo(e.target.id) })
      },
      { rootMargin: '-45% 0px -50% 0px', threshold: 0 },
    )
    items.forEach((it) => {
      const el = document.getElementById(it.id)
      if (el) obs.observe(el)
    })
    return () => obs.disconnect()
  }, [items])

  return (
    <nav className="toc" aria-label="Índice">
      <ul>
        {items.map((it) => (
          <li key={it.id}>
            <a href={`#${it.id}`} className={activo === it.id ? 'active' : ''}>{it.label}</a>
          </li>
        ))}
      </ul>
    </nav>
  )
}
