"use client";

import { useRef, useState, type KeyboardEvent } from "react";

import { CopyButton } from "@/components/ui/CopyButton";
import { jsOutput } from "@/lib/spec";
import type { Outputs } from "@/lib/types";

type TabKey = keyof Outputs;

interface Tab {
  key: TabKey;
  label: string;
  copy: string;
  note: string;
  wrap: boolean;
}

function buildTabs({ schemaVersion, continuous, hasJs }: { schemaVersion: string; continuous: boolean; hasJs: boolean }): Tab[] {
  const tabs: Tab[] = [
    { key: "technical", label: "Technical", copy: "Copy spec", note: "Readable motion spec for a developer.", wrap: true },
    {
      key: "llm_prompt",
      label: "LLM Prompt",
      copy: "Copy prompt",
      note: "Paste into a coding LLM or agent, ideally with the recording attached.",
      wrap: true,
    },
    {
      key: "json",
      label: "JSON",
      copy: "Copy JSON",
      note: `Motion IR (schema ${schemaVersion}), without per-frame ${continuous ? "velocity samples" : "samples"}.`,
      wrap: false,
    },
    {
      key: "css",
      label: "CSS",
      copy: "Copy CSS",
      note: continuous
        ? "Suggested implementation: layout and, for autoplay-only scrollers, a CSS loop. Drag, momentum and resume need the JS tab."
        : "Suggested implementation. The original site may use different CSS, JS or an animation library.",
      wrap: false,
    },
  ];
  if (hasJs) {
    tabs.push({
      key: "js",
      label: "JS",
      copy: "Copy JS",
      note: "Suggested implementation: a small driver built from the measured values, not the original code. The site may use a carousel library instead.",
      wrap: false,
    });
  }
  return tabs;
}

/** WAI-ARIA tabs (automatic activation, roving tabindex). LLM Prompt is the default: it is the primary output. */
export function SpecTabs({
  outputs,
  disclaimer,
  schemaVersion,
  continuous = false,
}: {
  outputs: Outputs;
  disclaimer: string;
  /** `spec.schema_version` (stored 0.1 results keep saying 0.1). */
  schemaVersion: string;
  continuous?: boolean;
}) {
  const [active, setActive] = useState<TabKey>("llm_prompt");
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const js = jsOutput(outputs);
  // The JS tab exists only when the result carries a driver (continuous results).
  const tabs = buildTabs({ schemaVersion, continuous, hasJs: js != null });
  const tab = tabs.find((t) => t.key === active) ?? tabs[1];
  const text = (tab.key === "js" ? js : outputs[tab.key]) ?? "";

  const onKeyDown = (e: KeyboardEvent) => {
    const i = tabs.findIndex((t) => t.key === tab.key);
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % tabs.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = tabs.length - 1;
    if (next < 0) return;
    e.preventDefault();
    setActive(tabs[next].key);
    refs.current[tabs[next].key]?.focus();
  };

  return (
    <section id="spec" aria-labelledby="spec-title" className="scroll-mt-6 rounded-md border border-line bg-surface">
      <header className="flex flex-wrap items-end justify-between gap-x-4 gap-y-2 border-b border-line px-4 pt-2">
        <div className="flex flex-wrap items-end gap-x-6">
          <h2 id="spec-title" className="caption pb-2.5">
            Motion specification
          </h2>
          <div role="tablist" aria-label="Output format" onKeyDown={onKeyDown} className="-mb-px flex max-w-full overflow-x-auto">
            {tabs.map((t) => {
              const selected = t.key === tab.key;
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
          <CopyButton text={text} label={tab.copy} variant={tab.key === "llm_prompt" ? "primary" : "secondary"} />
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
