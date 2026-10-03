"use client";

import { useRef, useState, type KeyboardEvent } from "react";

import { CopyButton } from "@/components/ui/CopyButton";
import type { Outputs } from "@/lib/types";

type TabKey = keyof Outputs;

const TABS: { key: TabKey; label: string; copy: string; note: string; wrap: boolean }[] = [
  { key: "technical", label: "Technical", copy: "Copy spec", note: "Readable motion spec for a developer.", wrap: true },
  {
    key: "llm_prompt",
    label: "LLM Prompt",
    copy: "Copy prompt",
    note: "Paste into a coding LLM or agent, ideally with the recording attached.",
    wrap: true,
  },
  { key: "json", label: "JSON", copy: "Copy JSON", note: "Motion IR (schema 0.1), without per-frame samples.", wrap: false },
  {
    key: "css",
    label: "CSS",
    copy: "Copy CSS",
    note: "Suggested implementation. The original site may use different CSS, JS or an animation library.",
    wrap: false,
  },
];

/** WAI-ARIA tabs (automatic activation, roving tabindex). LLM Prompt is the default: it is the primary output. */
export function SpecTabs({ outputs, disclaimer }: { outputs: Outputs; disclaimer: string }) {
  const [active, setActive] = useState<TabKey>("llm_prompt");
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const tab = TABS.find((t) => t.key === active) ?? TABS[1];
  const text = outputs[active] ?? "";

  const onKeyDown = (e: KeyboardEvent) => {
    const i = TABS.findIndex((t) => t.key === active);
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    if (next < 0) return;
    e.preventDefault();
    setActive(TABS[next].key);
    refs.current[TABS[next].key]?.focus();
  };

  return (
    <section id="spec" aria-labelledby="spec-title" className="scroll-mt-6 rounded-md border border-line bg-surface">
      <header className="flex flex-wrap items-end justify-between gap-x-4 gap-y-2 border-b border-line px-4 pt-2">
        <div className="flex flex-wrap items-end gap-x-6">
          <h2 id="spec-title" className="caption pb-2.5">
            Motion specification
          </h2>
          <div role="tablist" aria-label="Output format" onKeyDown={onKeyDown} className="-mb-px flex max-w-full overflow-x-auto">
            {TABS.map((t) => {
              const selected = t.key === active;
              return (
                <button
                  key={t.key}
                  ref={(el) => {
                    refs.current[t.key] = el;
                  }}
                  type="button"
                  role="tab"
                  id={`tab-${t.key}`}
                  aria-selected={selected}
                  aria-controls={`panel-${t.key}`}
                  tabIndex={selected ? 0 : -1}
                  onClick={() => setActive(t.key)}
                  className={`focus-ring relative shrink-0 rounded-t-sm border-b-2 px-3 py-2 text-sm whitespace-nowrap transition-colors duration-150 ${
                    selected ? "border-ink font-medium text-ink" : "border-transparent text-ink-3 hover:border-line-strong hover:text-ink"
                  }`}
                >
                  {t.label}
                  {t.key === "llm_prompt" ? <span className="ml-1.5 hidden font-mono text-2xs text-signal sm:inline">primary</span> : null}
                </button>
              );
            })}
          </div>
        </div>
        <div className="pb-2">
          <CopyButton text={text} label={tab.copy} variant={active === "llm_prompt" ? "primary" : "secondary"} />
        </div>
      </header>

      <div role="tabpanel" id={`panel-${tab.key}`} aria-labelledby={`tab-${tab.key}`} tabIndex={0} className="focus-ring">
        <p className="border-b border-line px-4 py-2 text-xs text-ink-3">{tab.note}</p>
        {text.trim().length === 0 ? (
          <p className="px-4 py-8 text-sm text-ink-3">Nothing was generated for this format.</p>
        ) : (
          <pre
            className={`max-h-[32rem] overflow-auto px-4 py-4 font-mono text-[0.8125rem] leading-[1.6] text-ink ${
              tab.wrap ? "break-words whitespace-pre-wrap" : "whitespace-pre"
            }`}
          >
            {text}
          </pre>
        )}
        <p className="border-t border-line px-4 py-2 text-2xs text-ink-3">{disclaimer}</p>
      </div>
    </section>
  );
}
