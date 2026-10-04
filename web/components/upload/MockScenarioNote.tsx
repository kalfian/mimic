"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";

import {
  MOCK_CONTINUOUS_SAMPLE_JOB_ID,
  MOCK_INTERPRETER_SCENARIOS,
  MOCK_PIPELINE_FAILURES,
  MOCK_RESULT_SCENARIOS,
  MOCK_SAMPLE_JOB_ID,
  MOCK_UPLOAD_FAILURES,
} from "@/lib/mock/mockApi";

/**
 * Dev-only helper (mock mode): shows which simulated scenario the next upload will run and links
 * to the others. The mock itself reads the same query params (lib/README.md).
 */

const RESULT: readonly string[] = MOCK_RESULT_SCENARIOS;
const PIPELINE: readonly string[] = MOCK_PIPELINE_FAILURES;
const UPLOAD: readonly string[] = MOCK_UPLOAD_FAILURES;
const INTERPRETER: readonly string[] = MOCK_INTERPRETER_SCENARIOS;

function Group({
  title,
  param,
  values,
  current,
  reload = false,
}: {
  title: string;
  param: string;
  values: readonly string[];
  current: string | null;
  /** Health is read once on mount, so interpreter switches need a full reload. */
  reload?: boolean;
}) {
  const cls =
    "focus-ring inline-block rounded-sm border border-line px-1.5 py-0.5 text-ink-2 hover:border-ink-3 hover:text-ink aria-[current=true]:border-ink aria-[current=true]:bg-ink aria-[current=true]:text-on-ink";
  return (
    <div>
      <p className="mb-1 text-ink-3">{title}</p>
      <ul className="flex flex-wrap gap-1">
        {values.map((v) => (
          <li key={v}>
            {reload ? (
              <button
                type="button"
                aria-current={current === v ? "true" : undefined}
                className={cls}
                onClick={() => (window.location.search = `?${param}=${v}`)}
              >
                {v}
              </button>
            ) : (
              <Link href={`/?${param}=${v}`} replace scroll={false} aria-current={current === v ? "true" : undefined} className={cls}>
                {v}
              </Link>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function MockScenarioNote() {
  const params = useSearchParams();
  const fail = params.get("fail");
  const scenario = params.get("scenario");
  const interpreter = params.get("interpreter");
  const active = fail ? `fail=${fail}` : scenario ? `scenario=${scenario}` : "success";

  return (
    <details className="rounded-md border border-dashed border-line-strong font-mono text-2xs">
      <summary className="focus-ring cursor-pointer rounded-md px-3 py-2 text-ink-2 select-none">
        Mock API · next upload runs <span className="text-ink">{active}</span>
        {interpreter ? (
          <>
            {" "}
            · interpreter <span className="text-ink">{interpreter}</span>
          </>
        ) : null}
      </summary>
      <div className="space-y-3 border-t border-line px-3 py-3">
        <Group title="Result scenarios (?scenario=)" param="scenario" values={RESULT} current={scenario ?? (fail ? null : "success")} />
        <Group title="Pipeline failures (?fail=)" param="fail" values={PIPELINE} current={fail} />
        <Group title="Upload rejections (?fail=)" param="fail" values={UPLOAD} current={fail} />
        <Group title="Interpreter status (?interpreter=)" param="interpreter" values={INTERPRETER} current={interpreter} reload />
        <p className="text-ink-3">
          Finished sample without upload:{" "}
          <Link href={`/jobs/${MOCK_SAMPLE_JOB_ID}`} className="focus-ring rounded-sm text-ink underline underline-offset-2">
            /jobs/{MOCK_SAMPLE_JOB_ID.slice(0, 8)}…
          </Link>{" "}
          · continuous:{" "}
          <Link href={`/jobs/${MOCK_CONTINUOUS_SAMPLE_JOB_ID}`} className="focus-ring rounded-sm text-ink underline underline-offset-2">
            /jobs/{MOCK_CONTINUOUS_SAMPLE_JOB_ID.slice(0, 8)}…
          </Link>
        </p>
      </div>
    </details>
  );
}
