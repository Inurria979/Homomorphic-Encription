// Piezas presentacionales reutilizables con la estética de artículo de investigación.

export function Section({ id, alt, children }) {
  return (
    <section id={id} className={alt ? 'alt' : undefined}>
      <div className="wrap">{children}</div>
    </section>
  )
}

export function Head({ n, eyebrow, title, children }) {
  return (
    <div className="sec-head">
      <div className="eyebrow">{n != null ? `§${n} · ` : ''}{eyebrow}</div>
      <h2>{title}</h2>
      {children && <p className="lead">{children}</p>}
    </div>
  )
}

export function Figure({ caption, children }) {
  return (
    <figure className="fig figure">
      <div className="chartcard">{children}</div>
      {caption && <figcaption className="caption">{caption}</figcaption>}
    </figure>
  )
}

export function Prose({ children }) {
  return <div className="prose">{children}</div>
}
