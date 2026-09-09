"use client";

import { FormEvent, useState } from "react";
import { extractErrorMessage, type ApiErrorBody, type QueryResponse } from "@/lib/types";

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
          "This demo is rate-limited to keep it free to run -- please wait a bit and try again.",
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
      setErrorMessage(
        "Couldn't reach the server. Check your connection and try again.",
      );
      setStatus("error");
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-8 px-4 py-12 sm:py-16">
      <header className="flex flex-col gap-3 text-center">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          Ask scikit-learn
        </h1>
        <p className="text-sm text-neutral-600 dark:text-neutral-400 sm:text-base">
          A retrieval-augmented Q&amp;A demo grounded in real{" "}
          <a
            href="https://github.com/scikit-learn/scikit-learn/issues"
            target="_blank"
            rel="noopener noreferrer"
            className="underline underline-offset-2 hover:no-underline"
          >
            scikit-learn GitHub issues
          </a>
          . Ask a question about a scikit-learn bug, API design decision, or
          feature request, and get a cited answer sourced from real issue
          discussions -- or a clear &ldquo;I don&apos;t know&rdquo; when the
          answer isn&apos;t in there.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. Why does SGDRegressor raise an error about buffer dimensions?"
          rows={3}
          maxLength={2000}
          disabled={status === "loading"}
          className="w-full resize-none rounded-lg border border-neutral-300 bg-white p-3 text-base outline-none focus:border-neutral-500 disabled:opacity-60 dark:border-neutral-700 dark:bg-neutral-900"
        />
        <button
          type="submit"
          disabled={!question.trim() || status === "loading"}
          className="self-end rounded-lg bg-neutral-900 px-5 py-2 text-sm font-medium text-white transition disabled:cursor-not-allowed disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
        >
          {status === "loading" ? "Thinking..." : "Ask"}
        </button>
      </form>

      {status === "error" && errorMessage && (
        <div
          role="alert"
          className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200"
        >
          {errorMessage}
        </div>
      )}

      {status === "success" && result && (
        <ResultCard result={result} />
      )}
    </main>
  );
}

function ResultCard({ result }: { result: QueryResponse }) {
  if (result.refused) {
    return (
      <div className="flex flex-col gap-2 rounded-lg border border-amber-300 bg-amber-50 p-4 dark:border-amber-900 dark:bg-amber-950">
        <p className="text-sm font-medium text-amber-900 dark:text-amber-200">
          No confident answer found
        </p>
        <p className="text-sm text-amber-800 dark:text-amber-300">
          {result.answer}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <p className="whitespace-pre-wrap text-sm leading-relaxed sm:text-base">
        {result.answer}
      </p>
      {result.citations.length > 0 && (
        <div className="flex flex-col gap-2 border-t border-neutral-200 pt-3 dark:border-neutral-800">
          <p className="text-xs font-medium uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
            Sources
          </p>
          <ul className="flex flex-wrap gap-2">
            {result.citations.map((citation) => (
              <li key={citation.issue_number}>
                <a
                  href={citation.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-block rounded-full border border-neutral-300 px-3 py-1 text-xs font-medium hover:bg-neutral-100 dark:border-neutral-700 dark:hover:bg-neutral-800"
                >
                  #{citation.issue_number}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
