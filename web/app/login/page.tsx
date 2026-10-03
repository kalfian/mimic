import type { Metadata } from "next";

import { LoginScreen } from "@/components/auth/LoginScreen";
import { safeNext } from "@/lib/routes";

export const metadata: Metadata = {
  title: "Sign in",
};

export default async function LoginPage(props: PageProps<"/login">) {
  const { next } = await props.searchParams;
  return <LoginScreen next={safeNext(next)} />;
}
