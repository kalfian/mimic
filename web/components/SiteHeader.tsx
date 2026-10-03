import Link from "next/link";

import { IconMark } from "@/components/icons";
import { IS_MOCK_API } from "@/lib/api";

export function SiteHeader() {
  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto flex h-12 max-w-[1240px] items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <Link href="/" className="focus-ring -mx-1 flex items-center gap-2 rounded-sm px-1 text-ink">
          <IconMark size={20} className="text-signal" />
          <span className="font-mono text-sm font-semibold tracking-tight">mimic</span>
          <span className="hidden text-sm text-ink-3 sm:inline">/ UI motion reverse engineer</span>
        </Link>
        <div className="flex items-center gap-3">
          {IS_MOCK_API ? (
            <span
              className="rounded-sm border border-warn/40 bg-warn-soft px-1.5 py-0.5 font-mono text-2xs tracking-wide text-warn uppercase"
              title="NEXT_PUBLIC_API_MOCK=1: the API is simulated in the browser"
            >
              Mock API
            </span>
          ) : null}
        </div>
      </div>
    </header>
  );
}
