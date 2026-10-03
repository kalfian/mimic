/**
 * React hooks for backend/interpreter status on the upload page (client components only).
 *
 * - `useHealth()` fetches `GET /api/health` once on mount: drives whether the "Use AI labeling"
 *   toggle is enabled (`health?.interpreter?.available`) and which mode is configured.
 *   `interpreter` / `limits` are null without a full session (PLAN-auth A12).
 * - `useInterpreterCheck()` runs `POST /api/interpreter/check` on demand ("Test connection").
 *   Admin only: users get 403 `forbidden`, so only render the button for admins.
 *
 * Interpreter configuration (endpoint, model, API key) is server-side env only. The browser never
 * sees or sends credentials; these hooks only read status.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { checkInterpreter, getHealth, type InterpreterCheck } from "./api";
import { ApiError } from "./errors";
import type { Health } from "./types";

export interface UseHealthResult {
  health: Health | null;
  /** Request failed (API down → disable AI toggle, show "server unreachable"). */
  error: ApiError | null;
  isLoading: boolean;
  refresh: () => void;
}

export function useHealth(): UseHealthResult {
  const [attempt, setAttempt] = useState(0);
  const [snapshot, setSnapshot] = useState<{ attempt: number; health: Health | null; error: ApiError | null } | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal).then(
      (health) => {
        if (!controller.signal.aborted) setSnapshot({ attempt, health, error: null });
      },
      (err: unknown) => {
        if (!controller.signal.aborted) setSnapshot({ attempt, health: null, error: ApiError.from(err) });
      },
    );
    return () => controller.abort();
  }, [attempt]);

  const refresh = useCallback(() => setAttempt((n) => n + 1), []);
  const current = snapshot?.attempt === attempt ? snapshot : null;
  return { health: current?.health ?? null, error: current?.error ?? null, isLoading: current === null, refresh };
}

export interface UseInterpreterCheckResult {
  /** Last completed check (`ok: false` + `error` when the connection test failed). */
  check: InterpreterCheck | null;
  /** The check request itself failed (API unreachable, 5xx). */
  error: ApiError | null;
  isChecking: boolean;
  run: () => Promise<InterpreterCheck | null>;
}

export function useInterpreterCheck(): UseInterpreterCheckResult {
  const [check, setCheck] = useState<InterpreterCheck | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const run = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setIsChecking(true);
    setError(null);
    try {
      const result = await checkInterpreter(controller.signal);
      if (controller.signal.aborted) return null;
      setCheck(result);
      return result;
    } catch (err) {
      if (controller.signal.aborted) return null;
      setError(ApiError.from(err));
      return null;
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setIsChecking(false);
      }
    }
  }, []);

  return { check, error, isChecking, run };
}
