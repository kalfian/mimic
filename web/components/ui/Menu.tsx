"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

export type MenuEntry =
  | {
      key: string;
      label: ReactNode;
      onSelect: () => void;
      icon?: ReactNode;
      /** Shown but not selectable; `reason` explains why (visible + aria-describedby). */
      disabled?: boolean;
      reason?: string;
      tone?: "default" | "danger";
    }
  | { key: string; separator: true };

type ActionEntry = Extract<MenuEntry, { onSelect: () => void }>;

const isAction = (e: MenuEntry): e is ActionEntry => !("separator" in e);

/**
 * Menu button (WAI-ARIA APG "menu button"): Enter/Space/↓ open on the first item, ↑ on the last;
 * ↑/↓/Home/End move; Esc closes and returns focus to the trigger; Tab closes. Disabled items stay
 * focusable so their reason is read out. The popover is `position: fixed` (placed below the
 * trigger, or above when there is no room), so it is never clipped by a scrolling table; it closes
 * on outside pointer-down, scroll and resize.
 */
export function Menu({
  label,
  trigger,
  triggerClassName,
  items,
  header,
  minWidth = "13rem",
}: {
  /** Accessible name of the trigger (required when the trigger is icon-only). */
  label: string;
  trigger: ReactNode;
  triggerClassName: string;
  items: MenuEntry[];
  /** Non-interactive block above the items (e.g. the signed-in account). */
  header?: ReactNode;
  minWidth?: string;
}) {
  const [open, setOpen] = useState(false);
  const [focusOn, setFocusOn] = useState<"first" | "last">("first");
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const menuId = useId();

  const itemEls = () => Array.from(popRef.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? []);

  const close = (returnFocus: boolean) => {
    setOpen(false);
    if (returnFocus) triggerRef.current?.focus();
  };

  // Position + initial focus once the popover is in the DOM (style writes only, no state).
  useLayoutEffect(() => {
    if (!open) return;
    const pop = popRef.current;
    const t = triggerRef.current;
    if (!pop || !t) return;
    const r = t.getBoundingClientRect();
    const h = pop.offsetHeight;
    const w = pop.offsetWidth;
    const below = r.bottom + 4 + h <= window.innerHeight - 8 || r.top < h + 12;
    pop.style.top = below ? `${r.bottom + 4}px` : `${Math.max(8, r.top - 4 - h)}px`;
    pop.style.left = `${Math.max(8, Math.min(r.right - w, window.innerWidth - w - 8))}px`;
    const els = itemEls();
    (focusOn === "last" ? els[els.length - 1] : els[0])?.focus();
  }, [open, focusOn]);

  useEffect(() => {
    if (!open) return;
    const onPointer = (e: PointerEvent) => {
      const target = e.target as Node;
      if (popRef.current?.contains(target) || triggerRef.current?.contains(target)) return;
      setOpen(false);
    };
    const onScroll = (e: Event) => {
      if (popRef.current && e.target instanceof Node && popRef.current.contains(e.target)) return;
      setOpen(false);
    };
    const onResize = () => setOpen(false);
    document.addEventListener("pointerdown", onPointer);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onResize);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onResize);
    };
  }, [open]);

  const openWith = (where: "first" | "last") => {
    setFocusOn(where);
    setOpen(true);
  };

  const onTriggerKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openWith("first");
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      openWith("last");
    }
  };

  const onMenuKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const els = itemEls();
    const i = els.indexOf(document.activeElement as HTMLElement);
    const move = (n: number) => {
      e.preventDefault();
      els[(n + els.length) % els.length]?.focus();
    };
    if (e.key === "ArrowDown") move(i + 1);
    else if (e.key === "ArrowUp") move(i - 1);
    else if (e.key === "Home") move(0);
    else if (e.key === "End") move(els.length - 1);
    else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      close(true);
    } else if (e.key === "Tab") {
      setOpen(false);
    }
  };

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-label={label}
        onClick={() => (open ? close(false) : openWith("first"))}
        onKeyDown={onTriggerKey}
        className={triggerClassName}
      >
        {trigger}
      </button>
      {open ? (
        <div
          ref={popRef}
          className="mimic-pop fixed z-40 rounded-md border border-line-strong bg-surface py-1 text-sm"
          style={{ minWidth, maxWidth: "calc(100vw - 1rem)", top: -9999, left: -9999 }}
          onKeyDown={onMenuKey}
        >
          {header ? <div className="border-b border-line px-3 pt-1.5 pb-2.5">{header}</div> : null}
          <div id={menuId} role="menu" aria-label={label} className={header ? "pt-1" : ""}>
            {items.map((entry) => {
              if (!isAction(entry)) return <div key={entry.key} role="separator" className="my-1 border-t border-line" />;
              const reasonId = `${menuId}-${entry.key}-reason`;
              const danger = entry.tone === "danger";
              return (
                <button
                  key={entry.key}
                  type="button"
                  role="menuitem"
                  tabIndex={-1}
                  aria-disabled={entry.disabled || undefined}
                  aria-describedby={entry.disabled && entry.reason ? reasonId : undefined}
                  onClick={() => {
                    if (entry.disabled) return;
                    // Focus goes back to the trigger first, so a dialog opened by the item
                    // returns focus there when it closes.
                    triggerRef.current?.focus();
                    setOpen(false);
                    entry.onSelect();
                  }}
                  className={`flex w-full items-start gap-2.5 px-3 py-2 text-left outline-none focus-visible:bg-surface-2 ${
                    entry.disabled ? "cursor-not-allowed text-ink-3" : `hover:bg-surface-2 ${danger ? "text-danger" : "text-ink"}`
                  }`}
                >
                  <span className={`mt-0.5 shrink-0 ${entry.disabled ? "opacity-60" : danger ? "" : "text-ink-3"}`}>{entry.icon}</span>
                  <span className="min-w-0">
                    <span className="block">{entry.label}</span>
                    {entry.disabled && entry.reason ? (
                      <span id={reasonId} className="mt-0.5 block max-w-[16rem] text-xs leading-4 text-ink-3">
                        {entry.reason}
                      </span>
                    ) : null}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </>
  );
}
