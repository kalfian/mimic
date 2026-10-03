"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { UserMenu } from "@/components/auth/UserMenu";
import { IconMark } from "@/components/icons";
import { IS_MOCK_API } from "@/lib/api";
import { ADMIN_USERS_PATH, JOBS_PATH, normalizePathname } from "@/lib/routes";
import { useSession } from "@/lib/session";

interface NavItem {
  href: string;
  label: string;
  active: boolean;
}

/**
 * Anonymous / loading: wordmark only. Signed in: Analyze · My jobs (All jobs for admins) · Users
 * (admins) + the account menu. During a forced password change only the account menu (sign out)
 * is offered, so the user can't wander into pages that would bounce them back.
 */
export function SiteHeader() {
  const session = useSession();
  const pathname = normalizePathname(usePathname());
  const me = session.status === "authenticated" ? session.me : null;
  const full = me !== null && !me.must_change_password;
  const isAdmin = full && me.role === "admin";

  const nav: NavItem[] = full
    ? [
        { href: "/", label: "Analyze", active: pathname === "/" },
        { href: JOBS_PATH, label: isAdmin ? "All jobs" : "My jobs", active: pathname === JOBS_PATH || pathname.startsWith(`${JOBS_PATH}/`) },
        ...(isAdmin ? [{ href: ADMIN_USERS_PATH, label: "Users", active: pathname.startsWith("/admin") }] : []),
      ]
    : [];

  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto flex h-12 max-w-[1240px] items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <div className="flex min-w-0 items-center gap-6 self-stretch">
          <Link href="/" className="focus-ring -mx-1 flex shrink-0 items-center gap-2 self-center rounded-sm px-1 text-ink">
            <IconMark size={20} className="text-signal" />
            <span className="font-mono text-sm font-semibold tracking-tight">mimic</span>
            {nav.length === 0 ? <span className="hidden text-sm text-ink-3 sm:inline">/ UI motion reverse engineer</span> : null}
          </Link>
          {nav.length > 0 ? <NavLinks items={nav} className="hidden md:flex" /> : null}
        </div>
        <div className="flex items-center gap-3">
          {IS_MOCK_API ? (
            <span
              className="rounded-sm border border-warn/40 bg-warn-soft px-1.5 py-0.5 font-mono text-2xs tracking-wide text-warn uppercase"
              title="NEXT_PUBLIC_API_MOCK=1: the API is simulated in the browser"
            >
              Mock API
            </span>
          ) : null}
          {me ? <UserMenu me={me} forced={me.must_change_password} pathname={pathname} /> : null}
        </div>
      </div>
      {nav.length > 0 ? (
        <div className="border-t border-line md:hidden">
          <NavLinks items={nav} className="flex h-10 overflow-x-auto px-4 sm:px-6" />
        </div>
      ) : null}
    </header>
  );
}

function NavLinks({ items, className }: { items: NavItem[]; className: string }) {
  return (
    <nav aria-label="Main" className={className}>
      <ul className="flex items-stretch gap-5">
        {items.map((item) => (
          <li key={item.href} className="flex">
            <Link
              href={item.href}
              aria-current={item.active ? "page" : undefined}
              className={`focus-ring relative flex items-center rounded-sm text-sm whitespace-nowrap transition-colors duration-150 after:absolute after:inset-x-0 after:-bottom-px after:h-0.5 after:rounded-full ${
                item.active ? "font-medium text-ink after:bg-ink" : "text-ink-3 after:bg-transparent hover:text-ink"
              }`}
            >
              {item.label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}
