import type { Metadata } from "next";

import { UsersScreen } from "@/components/admin/UsersScreen";

export const metadata: Metadata = {
  title: "Users",
};

export default function UsersPage() {
  return <UsersScreen />;
}
