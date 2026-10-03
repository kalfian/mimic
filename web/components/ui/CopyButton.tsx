"use client";

import { useEffect, useRef, useState } from "react";

import { IconCheck, IconCopy, IconCross } from "@/components/icons";
import { Button, type ButtonSize, type ButtonVariant } from "@/components/ui/Button";

type CopyState = "idle" | "copied" | "failed";

async function writeClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // Insecure contexts (plain http on a LAN IP) have no async clipboard: fall back to execCommand.
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  const ok = document.execCommand("copy");
  area.remove();
  if (!ok) throw new Error("copy failed");
}

/** Copies `text`; reports the outcome visually and through a polite live region. */
export function CopyButton({
  text,
  label,
  variant = "secondary",
  size = "sm",
  className = "",
}: {
  text: string;
  label: string;
  variant?: ButtonVariant;
  size?: ButtonSize;
  className?: string;
}) {
  const [state, setState] = useState<CopyState>("idle");
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timer.current) window.clearTimeout(timer.current);
    },
    [],
  );

  const onClick = async () => {
    if (timer.current) window.clearTimeout(timer.current);
    let next: CopyState = "copied";
    try {
      await writeClipboard(text);
    } catch {
      next = "failed";
    }
    setState(next);
    timer.current = window.setTimeout(() => setState("idle"), next === "failed" ? 4000 : 1800);
  };

  const icon = state === "copied" ? <IconCheck /> : state === "failed" ? <IconCross /> : <IconCopy />;
  const shown = state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : label;

  return (
    <>
      <Button
        variant={variant}
        size={size}
        icon={icon}
        onClick={onClick}
        disabled={text.length === 0}
        className={`${state === "failed" ? "text-danger" : ""} ${className}`}
      >
        <span className="min-w-[5.5rem] text-left">{shown}</span>
      </Button>
      <span className="sr-only" role="status" aria-live="polite">
        {state === "copied" ? `${label}: copied to clipboard` : state === "failed" ? "Copy failed. Select the text and copy it manually." : ""}
      </span>
    </>
  );
}
