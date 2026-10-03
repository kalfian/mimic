import Link from "next/link";

import { IconLock } from "@/components/icons";
import { buttonClass } from "@/components/ui/Button";
import { USER_ROLE_LABELS } from "@/lib/format";
import { JOBS_PATH } from "@/lib/routes";
import type { Me } from "@/lib/types";

/**
 * 403 state for a signed-in account on an admin-only page. Deliberately not a redirect: the URL
 * stays, and the page says why instead of silently bouncing.
 */
export function Forbidden({ me }: { me: Me | null }) {
  return (
    <div className="mx-auto max-w-[1240px] px-4 py-10 sm:px-6 lg:px-8 lg:py-14">
      <section aria-labelledby="forbidden-title" className="max-w-xl space-y-4">
        <p className="caption">
          <span className="nums">403</span> · Admins only
        </p>
        <h1 id="forbidden-title" className="flex items-center gap-2 text-xl font-semibold tracking-tight text-ink">
          <IconLock size={20} className="shrink-0 text-ink-3" />
          You don’t have access to this page
        </h1>
        <p className="text-md text-ink-2">
          Managing accounts is limited to admins.
          {me ? (
            <>
              {" "}
              You’re signed in as <span className="font-mono text-ink">{me.username}</span> ({USER_ROLE_LABELS[me.role]}).
            </>
          ) : null}{" "}
          Ask an admin if you need an account created, reset or changed.
        </p>
        <div className="flex flex-wrap gap-3 pt-2">
          <Link href={JOBS_PATH} className={buttonClass("primary")}>
            Go to my jobs
          </Link>
          <Link href="/" className={buttonClass("secondary")}>
            Analyze a recording
          </Link>
        </div>
      </section>
    </div>
  );
}
