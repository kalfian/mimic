import { useNow } from "@/components/job/useNow";

/**
 * Whole seconds left until `untilMs` (epoch ms), ticking every second; 0 once passed, null when
 * there is no deadline. Set `untilMs` from an event handler (`Date.now() + retryAfterS * 1000`).
 */
export function useCountdown(untilMs: number | null): number | null {
  const now = useNow(1000, untilMs !== null);
  if (untilMs === null) return null;
  if (now === null) return null;
  return Math.max(0, Math.ceil((untilMs - now) / 1000));
}
