// Mirrors src/api/schemas.py's QueryResponse on the FastAPI backend.
export interface Citation {
  issue_number: number;
  url: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  refused: boolean;
  refusal_reason: string | null;
}

// `detail` is a plain string from our own handlers (auth/502/etc.), but
// FastAPI's built-in request-validation errors (422) return an array of
// {msg, loc, ...} objects instead -- handle both shapes safely.
export interface ApiErrorBody {
  detail?: string | { msg: string }[];
  error?: string;
}

export function extractErrorMessage(body: ApiErrorBody | null): string {
  if (!body) return "Something went wrong. Please try again.";
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail) && body.detail.length > 0) {
    return body.detail.map((d) => d.msg).join(" ");
  }
  if (body.error) return body.error;
  return "Something went wrong. Please try again.";
}
