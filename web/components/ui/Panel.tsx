import type { ReactNode } from "react";

/**
 * Instrument-panel section: hairline frame, mono caption, optional right-aligned meta/actions.
 * `id` is used for the heading so the region is labelled for assistive tech.
 */
export function Panel({
  id,
  title,
  meta,
  children,
  className = "",
  bodyClassName = "p-4",
}: {
  id: string;
  title: ReactNode;
  meta?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  const headingId = `${id}-title`;
  return (
    <section id={id} aria-labelledby={headingId} className={`rounded-md border border-line bg-surface ${className}`}>
      <header className="flex min-h-10 flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-line px-4 py-2">
        <h2 id={headingId} className="caption">
          {title}
        </h2>
        {meta ? <div className="flex flex-wrap items-center gap-2">{meta}</div> : null}
      </header>
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}
