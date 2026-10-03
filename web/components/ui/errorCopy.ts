import { describeApiError } from "@/lib/format";

/**
 * The copy every UI error surface shows for an `ApiError`: title, message and guidance, with
 * guidance lines the message already says removed. All of that lives in `describeApiError`; this
 * alias keeps the component imports stable.
 */
export const errorCopy = describeApiError;
