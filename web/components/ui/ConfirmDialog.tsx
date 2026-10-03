"use client";

import type { ReactNode, RefObject } from "react";

import { Button, type ButtonVariant } from "@/components/ui/Button";
import { Dialog } from "@/components/ui/Dialog";
import { Notice } from "@/components/ui/Notice";
import { errorCopy } from "@/components/ui/errorCopy";
import type { ApiError } from "@/lib/errors";

/**
 * Yes/no confirmation. Initial focus is Cancel (the safe choice); the confirming button names the
 * action ("Disable account", never "OK"). A server refusal is shown inline, the dialog stays open.
 */
export function ConfirmDialog({
  open,
  caption,
  title,
  description,
  children,
  confirmLabel,
  busyLabel,
  confirmVariant = "primary",
  confirmIcon,
  confirmDisabled = false,
  busy = false,
  error = null,
  onCancel,
  onConfirm,
  returnFocusRef,
}: {
  open: boolean;
  caption?: ReactNode;
  title: ReactNode;
  description: ReactNode;
  children?: ReactNode;
  confirmLabel: string;
  busyLabel?: string;
  confirmVariant?: ButtonVariant;
  confirmIcon?: ReactNode;
  confirmDisabled?: boolean;
  busy?: boolean;
  error?: ApiError | null;
  onCancel: () => void;
  onConfirm: () => void;
  returnFocusRef?: RefObject<HTMLElement | null>;
}) {
  const d = error ? errorCopy(error) : null;
  return (
    <Dialog
      open={open}
      onClose={onCancel}
      role="alertdialog"
      caption={caption}
      title={title}
      description={description}
      busy={busy}
      returnFocusRef={returnFocusRef}
      footer={
        <>
          <Button onClick={onCancel} disabled={busy}>
            Cancel
          </Button>
          <Button variant={confirmVariant} icon={confirmIcon} loading={busy} disabled={confirmDisabled} onClick={onConfirm}>
            {busy && busyLabel ? busyLabel : confirmLabel}
          </Button>
        </>
      }
    >
      {children}
      {d ? (
        <Notice tone="danger" live="assertive" title={d.title} guidance={d.guidance}>
          {d.message}
        </Notice>
      ) : null}
    </Dialog>
  );
}
