import type { NextConfig } from "next";
import path from "path";
import { fileURLToPath } from "url";

const dirname = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  // Pins Turbopack's root to this frontend/ directory. Without this, Next.js
  // walks up looking for a lockfile and can pick up an unrelated one further
  // up the filesystem (e.g. outside this repo entirely), which is harmless
  // but produces a confusing warning and could behave differently between
  // local dev and Vercel's build environment.
  turbopack: {
    root: dirname,
  },
};

export default nextConfig;
