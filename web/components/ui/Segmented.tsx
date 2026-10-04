"use client";

import { useRef, type KeyboardEvent } from "react";

/**
 * Small exclusive choice (view / scale switches): WAI-ARIA radio group with a roving tab stop,
 * arrow keys move the selection. Same look as the playback-speed control.
 */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  className = "",
}: {
  label: string;
  value: T;
  options: readonly { value: T; label: string; title?: string }[];
  onChange: (value: T) => void;
  className?: string;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const current = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  );

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const n = options.length;
    let next = -1;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (current + 1) % n;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (current - 1 + n) % n;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = n - 1;
    if (next < 0) return;
    e.preventDefault();
    onChange(options[next].value);
    refs.current[next]?.focus();
  };

  return (
    <div role="radiogroup" aria-label={label} onKeyDown={onKeyDown} className={`inline-flex rounded-md border border-line p-0.5 ${className}`}>
      {options.map((o, i) => {
        const checked = i === current;
        return (
          <button
            key={o.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={checked}
            tabIndex={checked ? 0 : -1}
            title={o.title}
            onClick={() => onChange(o.value)}
            className="focus-ring rounded-[4px] px-2 py-0.5 text-xs whitespace-nowrap text-ink-2 transition-colors duration-150 hover:text-ink aria-checked:bg-ink aria-checked:text-on-ink"
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
