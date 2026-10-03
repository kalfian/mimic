"use client";

import { useState, type InputHTMLAttributes, type ReactNode, type Ref } from "react";

import { IconAlert, IconEye, IconEyeOff } from "@/components/icons";

/** Input look shared by every text field: 3 px radius (token), hairline, ink focus ring. */
export const INPUT_CLASS =
  "focus-ring block h-10 w-full min-w-0 rounded-sm border border-line-strong bg-surface px-3 text-base text-ink placeholder:text-ink-3 " +
  "transition-colors duration-150 hover:border-ink-3 read-only:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-45 " +
  "aria-invalid:border-danger aria-invalid:hover:border-danger";

export interface FieldProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "id"> {
  id: string;
  label: ReactNode;
  /** Shown under the input (and announced with it). Stays visible next to an error. */
  hint?: ReactNode;
  error?: string | null;
  /** Right-aligned text on the label row (e.g. a live character count). */
  aside?: ReactNode;
  ref?: Ref<HTMLInputElement>;
}

function describedBy(id: string, hint: unknown, error: unknown): string | undefined {
  const ids = [error ? `${id}-error` : null, hint ? `${id}-hint` : null].filter(Boolean);
  return ids.length ? ids.join(" ") : undefined;
}

function FieldFrame({
  id,
  label,
  hint,
  error,
  aside,
  children,
}: Pick<FieldProps, "id" | "label" | "hint" | "error" | "aside"> & { children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-sm font-medium text-ink">
          {label}
        </label>
        {aside ? <span className="text-xs text-ink-3">{aside}</span> : null}
      </div>
      {children}
      {error ? (
        <p id={`${id}-error`} className="flex items-start gap-1.5 text-xs text-danger">
          <IconAlert size={14} className="mt-px shrink-0" />
          {error}
        </p>
      ) : null}
      {hint ? (
        <p id={`${id}-hint`} className="text-xs text-ink-3">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

/** Labelled text input with hint + error wired to aria-describedby / aria-invalid. */
export function TextField({ id, label, hint, error, aside, className = "", ref, ...input }: FieldProps) {
  return (
    <FieldFrame id={id} label={label} hint={hint} error={error} aside={aside}>
      <input
        ref={ref}
        id={id}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy(id, hint, error)}
        className={`${INPUT_CLASS} ${className}`}
        {...input}
      />
    </FieldFrame>
  );
}

/** Password input with a show/hide toggle (a real button, so it is keyboard reachable). */
export function PasswordField({ id, label, hint, error, aside, className = "", ref, ...input }: FieldProps) {
  const [visible, setVisible] = useState(false);
  return (
    <FieldFrame id={id} label={label} hint={hint} error={error} aside={aside}>
      <div className="relative">
        <input
          ref={ref}
          id={id}
          type={visible ? "text" : "password"}
          spellCheck={false}
          autoCapitalize="none"
          autoCorrect="off"
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy(id, hint, error)}
          className={`${INPUT_CLASS} pr-11 font-mono text-sm tracking-tight ${className}`}
          {...input}
        />
        <button
          type="button"
          onClick={() => setVisible((v) => !v)}
          aria-pressed={visible}
          aria-label={visible ? "Hide password" : "Show password"}
          aria-controls={id}
          disabled={input.disabled}
          className="focus-ring absolute top-1 right-1 inline-flex size-8 items-center justify-center rounded-sm text-ink-3 transition-colors duration-150 hover:bg-surface-2 hover:text-ink disabled:opacity-45"
        >
          {visible ? <IconEyeOff /> : <IconEye />}
        </button>
      </div>
    </FieldFrame>
  );
}
