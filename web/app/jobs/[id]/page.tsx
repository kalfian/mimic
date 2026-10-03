import type { Metadata } from "next";

import { JobView } from "@/components/job/JobView";

export const metadata: Metadata = {
  title: "Analysis",
};

export default async function JobPage(props: PageProps<"/jobs/[id]">) {
  const { id } = await props.params;
  return <JobView id={decodeURIComponent(id)} />;
}
