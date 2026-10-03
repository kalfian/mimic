import { useEffect, useState } from "react";

/** Wall clock that re-renders every `intervalMs` while `active`; null until the first tick (keeps render pure). */
export function useNow(intervalMs = 1000, active = true): number | null {
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    if (!active) return;
    const tick = () => setNow(Date.now());
    const first = window.setTimeout(tick, 0);
    const id = window.setInterval(tick, intervalMs);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(id);
    };
  }, [intervalMs, active]);
  return now;
}
