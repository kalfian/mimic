"use client";

import { IconChevronDown } from "@/components/icons";
import { useAdminUsers } from "@/lib/useAdminUsers";

/** "" = everyone (incl. legacy jobs), "me", or a user id. Admins only (lists accounts). */
export function OwnerFilter({ value, onChange, meId }: { value: string; onChange: (owner: string) => void; meId: string }) {
  const { users, isLoading, error } = useAdminUsers();
  const others = users.filter((u) => u.id !== meId);
  const known = value === "" || value === "me" || users.some((u) => u.id === value);

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="owner-filter" className="text-sm text-ink-3">
        Owner
      </label>
      <div className="relative">
        <select
          id="owner-filter"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          aria-describedby={error ? "owner-filter-error" : undefined}
          className="focus-ring h-8 max-w-[14rem] appearance-none rounded-sm border border-line-strong bg-surface pr-8 pl-2.5 text-sm text-ink transition-colors duration-150 hover:border-ink-3"
        >
          <option value="">Everyone</option>
          <option value="me">Me</option>
          {others.map((u) => (
            <option key={u.id} value={u.id}>
              {u.username} ({u.job_count})
            </option>
          ))}
          {!known && !isLoading ? <option value={value}>Unknown account</option> : null}
          {isLoading ? <option disabled>Loading accounts…</option> : null}
        </select>
        <IconChevronDown size={14} className="pointer-events-none absolute top-1/2 right-2.5 -translate-y-1/2 text-ink-3" />
      </div>
      {error ? (
        <span id="owner-filter-error" className="text-xs text-ink-3">
          Accounts couldn’t be loaded
        </span>
      ) : null}
    </div>
  );
}
