"use client";

import { FormEvent, ReactNode, useState } from "react";
import { extractErrorMessage, type ApiErrorBody, type QueryResponse } from "@/lib/types";
import { ThemeToggle } from "./theme-toggle";

type Status = "idle" | "loading" | "success" | "error";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || status === "loading") return;

    setStatus("loading");
    setErrorMessage(null);
    setResult(null);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (response.status === 429) {
        setErrorMessage(
          "This demo is rate-limited to keep it free to run. Wait a moment and try again.",
        );
        setStatus("error");
        return;
      }

      if (!response.ok) {
        const body: ApiErrorBody | null = await response.json().catch(() => null);
        setErrorMessage(extractErrorMessage(body));
        setStatus("error");
        return;
      }

      const data: QueryResponse = await response.json();
      setResult(data);
      setStatus("success");
    } catch {
      setErrorMessage("Couldn't reach the server. Check your connection and try again.");
      setStatus("error");
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-xl flex-1 flex-col gap-10 px-6 py-14 sm:py-20">
      <header className="flex flex-col gap-3">
        <div className="flex items-start justify-between gap-4">
          <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
            ask scikit-learn
          </h1>
          <ThemeToggle />
        </div>
        <p className="max-w-[60ch] text-[15px] leading-relaxed text-ink/70">
          Every answer here comes from{" "}
          <a
            href="https://github.com/scikit-learn/scikit-learn/issues"
            target="_blank"
            rel="noopener noreferrer"
            className="text-prompt-blue underline underline-offset-2 hover:no-underline"
          >
            scikit-learn&rsquo;s GitHub issue tracker.
          </a>{" "}
          12,000+ bug reports, feature requests, and API debates
          where its maintainers do their actual work. Ask a real question and
          get a cited answer pulled from that history, or a rejection if
          the tracker doesn&apos;t have one.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <Cell label="In [ ]:" labelColor="text-prompt-blue">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Why does SGDRegressor raise an error about buffer dimensions?"
            rows={3}
            maxLength={2000}
            disabled={status === "loading"}
            className="w-full resize-none rounded-md border border-hairline bg-chip-bg p-3 font-mono text-sm leading-relaxed text-ink outline-none placeholder:text-ink/35 focus-visible:border-prompt-blue focus-visible:ring-2 focus-visible:ring-prompt-blue/20 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={!question.trim() || status === "loading"}
            // Some browser extensions (password managers, Grammarly, etc.)
            // inject/strip attributes on form buttons before hydration,
            // which trips React's hydration-mismatch warning even though
            // `disabled` here is a pure function of `question`/`status` with
            // no client-only branching. Confirmed via incognito (no
            // extensions) showing no warning.
            suppressHydrationWarning
            className="mt-2 self-start rounded-md bg-prompt-blue px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-prompt-blue/90 focus-visible:ring-2 focus-visible:ring-prompt-blue/40 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-40 dark:text-paper cursor-pointer"
          >
            Run cell
          </button>
        </Cell>
      </form>

      <div aria-live="polite">
        {status !== "idle" && (
          <div className="animate-cell-resolve">
            <Cell label="Out[ ]:" labelColor="text-output-rust">
              <OutContent status={status} result={result} errorMessage={errorMessage} />
            </Cell>
          </div>
        )}
      </div>
    </main>
  );
}

function Cell({
  label,
  labelColor,
  children,
}: {
  label: string;
  labelColor: string;
  children: ReactNode;
}) {
  return (
    <div className="grid grid-cols-[3.5rem_1fr] items-start gap-x-3 sm:grid-cols-[4.5rem_1fr]">
      <span className={`pt-2 font-mono text-xs sm:text-sm ${labelColor}`}>{label}</span>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

function OutContent({
  status,
  result,
  errorMessage,
}: {
  status: Status;
  result: QueryResponse | null;
  errorMessage: string | null;
}) {
  if (status === "loading") {
    return (
      <p className="font-mono text-sm text-ink/45">
        # retrieving related issues
        <span className="animate-pulse">▍</span>
      </p>
    );
  }

  if (status === "error") {
    return <p className="font-mono text-sm text-output-rust">{`# ${errorMessage}`}</p>;
  }

  if (!result) return null;

  if (result.refused) {
    return <p className="font-mono text-sm text-output-rust">{`# ${result.answer}`}</p>;
  }

  return (
    <div className="flex flex-col gap-3">
      <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-ink">
        {result.answer}
      </p>
      {result.citations.length > 0 && (
        <p className="font-mono text-xs text-ink/50">
          {"# sourced from "}
          {result.citations.map((citation, i) => (
            <span key={citation.issue_number}>
              <a
                href={citation.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-ink/70 underline underline-offset-2 hover:text-prompt-blue"
              >
                #{citation.issue_number}
              </a>
              {i < result.citations.length - 1 ? ", " : ""}
            </span>
          ))}
        </p>
      )}
    </div>
  );
}
