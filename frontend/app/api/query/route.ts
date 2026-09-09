// Server-side proxy in front of the FastAPI backend's POST /query.
//
// The browser never talks to the backend directly -- it only ever calls this
// same-origin route. BACKEND_URL is a plain (non-NEXT_PUBLIC_) env var, so
// it's only readable here on the server and never ends up in the client
// bundle. This keeps the actual backend URL, and any future auth headers we
// might add, out of what a visitor's devtools can see.
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return NextResponse.json(
      { detail: "Server misconfiguration: BACKEND_URL is not set." },
      { status: 500 },
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body." }, { status: 400 });
  }

  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${backendUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    return NextResponse.json(
      { detail: "Could not reach the backend service. Please try again." },
      { status: 502 },
    );
  }

  const data = await backendResponse.json().catch(() => null);
  return NextResponse.json(data, { status: backendResponse.status });
}
