import type { ButtonHTMLAttributes, ReactNode, Ref } from "react";

import { IconSpinner } from "@/components/icons";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

const BASE =
  "focus-ring inline-flex shrink-0 items-center justify-center gap-2 rounded-md font-medium whitespace-nowrap select-none " +
  "transition-colors duration-150 ease-out disabled:cursor-not-allowed disabled:opacity-45 aria-disabled:cursor-not-allowed aria-disabled:opacity-45";

const VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-ink text-on-ink hover:bg-ink-2 active:bg-ink disabled:hover:bg-ink",
  secondary:
    "border border-line-strong bg-surface text-ink hover:border-ink-3 hover:bg-surface-2 active:bg-line disabled:hover:bg-surface disabled:hover:border-line-strong",
  ghost: "text-ink-2 hover:bg-surface-2 hover:text-ink active:bg-line disabled:hover:bg-transparent",
  /** Only for the confirming button of an irreversible action (delete, disable). */
  danger: "bg-danger text-on-ink hover:bg-danger/85 active:bg-danger disabled:hover:bg-danger",
};

const SIZES: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-sm",
  md: "h-10 px-4 text-base",
};

/** Class list for anything that should look like a button (e.g. a `<Link>`). */
export function buttonClass(variant: ButtonVariant = "secondary", size: ButtonSize = "md", extra = ""): string {
  return `${BASE} ${VARIANTS[variant]} ${SIZES[size]} ${extra}`.trim();
}

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /** Shows a spinner, sets aria-busy and blocks clicks while keeping the label. */
  loading?: boolean;
  icon?: ReactNode;
  /** React 19: `ref` is a plain prop on function components. */
  ref?: Ref<HTMLButtonElement>;
}

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  icon,
  className = "",
  disabled,
  children,
  type = "button",
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={buttonClass(variant, size, className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <IconSpinner /> : icon}
      {children}
    </button>
  );
}
