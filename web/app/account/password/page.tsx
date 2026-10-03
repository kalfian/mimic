import type { Metadata } from "next";

import { ChangePasswordScreen } from "@/components/auth/ChangePasswordScreen";
import { safeNext } from "@/lib/routes";

export const metadata: Metadata = {
  title: "Password",
};

export default async function ChangePasswordPage(props: PageProps<"/account/password">) {
  const { next } = await props.searchParams;
  return <ChangePasswordScreen next={safeNext(next)} />;
}
