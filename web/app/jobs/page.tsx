import type { Metadata } from "next";

import { JobsScreen } from "@/components/jobs/JobsScreen";

export const metadata: Metadata = {
  title: "Jobs",
};

export default async function JobsPage(props: PageProps<"/jobs">) {
  const { owner } = await props.searchParams;
  return <JobsScreen initialOwner={typeof owner === "string" && owner ? owner : null} />;
}
