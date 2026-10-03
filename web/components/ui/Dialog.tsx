"use client";

import { useEffect, useId, useLayoutEffect, useRef, type ReactNode, type RefObject } from "react";

import { IconCross } from "@/components/icons";

export interface DialogProps {
  open: boolean;
  /** Esc and the close button call this (never while `busy`). */
  onClose: () => void;
  title: ReactNode;
  /** Mono caption above the title ("Delete job", "New account"). */
  caption?: ReactNode;
  /** Short description, linked with aria-describedby. */
  description?: ReactNode;
  children?: ReactNode;
  /** Footer buttons, right-aligned (primary action last). */
  footer?: ReactNode;
  /** A request is in flight: Esc and the close button do nothing. */
  busy?: boolean;
  /** Element to focus when the dialog opens (default: the first focusable control). */
  initialFocusRef?: RefObject<HTMLElement | null>;
  /**
   * Where focus goes when the dialog closes and the element that opened it is gone (e.g. the
   * row was deleted). Default: the opener.
   */
  returnFocusRef?: RefObject<HTMLElement | null>;
  /** "alertdialog" for confirmations of destructive actions. */
  role?: "dialog" | "alertdialog";
  width?: "sm" | "md";
}

/**
 * Modal dialog on the native `<dialog>` + `showModal()`: the rest of the page becomes inert (real
 * focus trap, including for screen readers), Esc fires `cancel`. On top of that: labelled +
 * described, initial focus, focus returned to the opener (or `returnFocusRef`), Esc blocked while
 * busy. A backdrop click does not close it, so typed input is never lost by a stray click.
 *
 * Mounted only while `open`; the entrance animation is in globals.css and is removed by
 * prefers-reduced-motion.
 */
export function Dialog(props: DialogProps) {
  if (!props.open) return null;
  return <DialogImpl {...props} />;
}

function DialogImpl({
  onClose,
  title,
  caption,
  description,
  children,
  footer,
  busy = false,
  initialFocusRef,
  returnFocusRef,
  role = "dialog",
  width = "sm",
}: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  const descId = useId();
  const onCloseRef = useRef(onClose);
  const busyRef = useRef(busy);

  useEffect(() => {
    onCloseRef.current = onClose;
    busyRef.current = busy;
  });

  useLayoutEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    const active = document.activeElement;
    const opener = active instanceof HTMLElement && active !== document.body ? active : null;
    const fallback = returnFocusRef;
    if (!dialog.open) dialog.showModal();
    // Initial focus: the requested element, else the first field of the body, else the first
    // footer button (Cancel for confirmations) — never the close button.
    const first =
      initialFocusRef?.current ??
      dialog.querySelector<HTMLElement>("[data-dialog-body] :is(input, select, textarea):not(:disabled)") ??
      dialog.querySelector<HTMLElement>("[data-dialog-footer] button:not(:disabled)");
    first?.focus();

    const onCancel = (e: Event) => {
      e.preventDefault();
      if (!busyRef.current) onCloseRef.current();
    };
    // Keep Tab inside the dialog (native modals otherwise let focus leave to the browser UI).
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const focusable = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          "button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex]:not([tabindex='-1'])",
        ),
      ).filter((el) => el.offsetParent !== null);
      if (focusable.length === 0) return;
      const head = focusable[0];
      const tail = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === head) {
        e.preventDefault();
        tail.focus();
      } else if (!e.shiftKey && document.activeElement === tail) {
        e.preventDefault();
        head.focus();
      }
    };
    dialog.addEventListener("cancel", onCancel);
    dialog.addEventListener("keydown", onKeyDown);
    return () => {
      dialog.removeEventListener("cancel", onCancel);
      dialog.removeEventListener("keydown", onKeyDown);
      if (dialog.open) dialog.close();
      // Restore focus after React has finished removing the dialog.
      queueMicrotask(() => {
        const back = fallback?.current ?? null;
        const target = opener?.isConnected ? opener : back?.isConnected ? back : null;
        target?.focus();
      });
    };
    // Open once per mount; refs carry the latest props.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const maxW = width === "md" ? "sm:max-w-[34rem]" : "sm:max-w-[28rem]";

  return (
    <dialog
      ref={ref}
      role={role}
      aria-modal="true"
      aria-labelledby={titleId}
      aria-describedby={description ? descId : undefined}
      aria-busy={busy || undefined}
      className={`mimic-dialog m-auto w-[calc(100vw-2rem)] ${maxW} max-h-[calc(100dvh-2rem)] overflow-y-auto rounded-md border border-line-strong bg-surface p-0 text-ink`}
    >
      <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
        <div className="min-w-0 space-y-1">
          {caption ? <p className="caption">{caption}</p> : null}
          <h2 id={titleId} className="text-md font-semibold tracking-tight break-words text-ink">
            {title}
          </h2>
        </div>
        <button
          type="button"
          onClick={() => {
            if (!busy) onClose();
          }}
          disabled={busy}
          aria-label="Close"
          className="focus-ring -mt-1 -mr-2 inline-flex size-8 shrink-0 items-center justify-center rounded-sm text-ink-3 transition-colors duration-150 hover:bg-surface-2 hover:text-ink disabled:opacity-45"
        >
          <IconCross />
        </button>
      </div>
      <div data-dialog-body className="space-y-4 px-5 py-4">
        {description ? (
          <div id={descId} className="text-sm text-ink-2">
            {description}
          </div>
        ) : null}
        {children}
      </div>
      {footer ? (
        <div data-dialog-footer className="flex flex-col-reverse gap-2 border-t border-line px-5 py-3 sm:flex-row sm:items-center sm:justify-end">
          {footer}
        </div>
      ) : null}
    </dialog>
  );
}
