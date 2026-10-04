/**
 * Mock mode: which job belongs to whom (PLAN-auth §8.3, U3), persisted via `mockStore`.
 *
 * Mock job ids encode their scenario + creation time (see mockApi.ts), so the lifecycle survives a
 * reload by itself. This registry adds what the id can't carry: the owner, the original filename,
 * and tombstones for deleted jobs (a deleted id must stay 404 even though it still decodes).
 *
 * Seeded on first use (and after `resetMockJobsRegistry()`), see `MOCK_SEED_JOBS`.
 */

import { MOCK_JOBS_STORAGE_KEY, readMockState, writeMockState } from "./mockStore";

/* ---------- seeded accounts (ids shared with mockAuth.ts) ---------- */

/** Seeded mock user ids (32 hex chars, same shape as real ids). */
export const MOCK_USER_IDS = {
  admin: "a0000000000000000000000000000001",
  user: "a0000000000000000000000000000002",
  newbie: "a0000000000000000000000000000003",
  disabled: "a0000000000000000000000000000004",
  alice: "a0000000000000000000000000000005",
} as const;

/** Job id of the contract fixture (`sample-result.json`); seeded as owned by `user`. */
export const FIXTURE_JOB_ID = "3f2b9c1d8e7a4b6c9d0e1f2a3b4c5d6e";

/** Job id of the continuous contract fixture (`sample-continuous-result.json`); owned by `user`. */
export const CONTINUOUS_FIXTURE_JOB_ID = "7c41e2a95b0d4f3e8a6c1b2d3e4f5a6b";

export interface MockJobEntry {
  id: string;
  /** null = legacy job from before accounts existed (admin-only, "Legacy" owner). */
  ownerId: string | null;
  filename: string;
  /** ms since epoch. */
  createdAt: number;
}

const t = (iso: string) => Date.parse(iso);
const mockId = (scenario: string, ai: 0 | 1, ratio: string, createdAt: number) => `mock-${scenario}-${ai}-${ratio}-${createdAt.toString(36)}`;

function seedJobs(): MockJobEntry[] {
  const rows: [string, 0 | 1, string, string, string | null, string][] = [
    // scenario, ai, pixel ratio, created, owner, filename
    ["success", 1, "2", "2026-09-29T14:05:00Z", MOCK_USER_IDS.alice, "dropdown-open.mp4"],
    ["no_motion_detected", 0, "auto", "2026-09-30T08:12:00Z", MOCK_USER_IDS.alice, "static-screen.mov"],
    ["low_confidence", 1, "auto", "2026-09-30T16:40:00Z", MOCK_USER_IDS.admin, "toast-dark.mov"],
    ["forward_only", 0, "auto", "2026-09-20T10:00:00Z", null, "modal-enter.mp4"],
    ["no_preview", 0, "auto", "2026-10-02T11:20:00Z", MOCK_USER_IDS.user, "button-press.webm"],
  ];
  const seeded: MockJobEntry[] = rows.map(([scenario, ai, ratio, iso, ownerId, filename]) => ({
    id: mockId(scenario, ai, ratio, t(iso)),
    ownerId,
    filename,
    createdAt: t(iso),
  }));
  seeded.push({ id: FIXTURE_JOB_ID, ownerId: MOCK_USER_IDS.user, filename: "card-hover.mov", createdAt: t("2026-10-01T09:30:00Z") });
  seeded.push({ id: CONTINUOUS_FIXTURE_JOB_ID, ownerId: MOCK_USER_IDS.user, filename: "carousel-drag.mov", createdAt: t("2026-10-03T10:00:00Z") });
  return seeded;
}

interface RegistryState {
  v: 1;
  entries: MockJobEntry[];
  deleted: string[];
}

function isRegistryState(value: unknown): value is RegistryState {
  const v = value as RegistryState | null;
  return typeof v === "object" && v !== null && v.v === 1 && Array.isArray(v.entries) && Array.isArray(v.deleted);
}

let state: RegistryState | null = null;

function load(): RegistryState {
  if (!state) state = readMockState(MOCK_JOBS_STORAGE_KEY, isRegistryState) ?? { v: 1, entries: seedJobs(), deleted: [] };
  return state;
}

function save(): void {
  if (state) writeMockState(MOCK_JOBS_STORAGE_KEY, state);
}

/** Back to the seeded jobs (tests, "reset mock data"). */
export function resetMockJobsRegistry(): void {
  state = { v: 1, entries: seedJobs(), deleted: [] };
  save();
}

export function registryEntry(id: string): MockJobEntry | null {
  return load().entries.find((e) => e.id === id) ?? null;
}

export function isDeletedJob(id: string): boolean {
  return load().deleted.includes(id);
}

export function registerJob(entry: MockJobEntry): void {
  const s = load();
  s.entries = [...s.entries.filter((e) => e.id !== entry.id), entry];
  s.deleted = s.deleted.filter((d) => d !== entry.id);
  save();
}

/** Remove + tombstone. */
export function unregisterJob(id: string): void {
  const s = load();
  s.entries = s.entries.filter((e) => e.id !== id);
  if (!s.deleted.includes(id)) s.deleted = [...s.deleted, id];
  save();
}

export function allJobEntries(): MockJobEntry[] {
  return [...load().entries];
}

export function jobCountOf(ownerId: string): number {
  return load().entries.filter((e) => e.ownerId === ownerId).length;
}

/** Delete (and tombstone) every job of `ownerId`; returns the removed ids (user deletion). */
export function deleteJobsOf(ownerId: string): string[] {
  const ids = load()
    .entries.filter((e) => e.ownerId === ownerId)
    .map((e) => e.id);
  for (const id of ids) unregisterJob(id);
  return ids;
}
