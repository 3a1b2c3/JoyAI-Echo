import { defineConfig, loadEnv, type ProxyOptions } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import type { ServerResponse } from "node:http";

const REFUSED_LOG_THROTTLE_MS = 10_000;

function createProxyOptions(
  route: string,
  target: string,
  extra: ProxyOptions = {},
): ProxyOptions {
  let lastRefusedLog = 0;
  return {
    target,
    changeOrigin: true,
    ...extra,
    configure(proxy) {
      extra.configure?.(proxy, extra);
      proxy.on(
        "error",
        (
          error: NodeJS.ErrnoException,
          _req: unknown,
          res: Partial<ServerResponse> | undefined,
        ) => {
          if (error.code === "ECONNREFUSED") {
            const now = Date.now();
            if (now - lastRefusedLog > REFUSED_LOG_THROTTLE_MS) {
              lastRefusedLog = now;
              console.warn(
                `[vite] nanobot proxy target unavailable for ${route}: ${target}. ` +
                  "Start `nanobot gateway` with channels.websocket.enabled=true, " +
                  "or set NANOBOT_API_URL to the running websocket gateway.",
              );
            }
          } else {
            console.warn(
              `[vite] nanobot proxy error for ${route}: ${error.message}`,
            );
          }

          if (
            res &&
            !res.headersSent &&
            typeof res.writeHead === "function" &&
            typeof res.end === "function"
          ) {
            res.writeHead(502, { "Content-Type": "text/plain" });
            res.end("nanobot gateway unavailable");
          }
        },
      );
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.NANOBOT_API_URL ?? "http://127.0.0.1:8765";

  const isProd = mode === "production" || env.NODE_ENV === "production";
  const base = isProd ? "/webui/" : "";

  return {
    base,
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(import.meta.dirname, "./src"),
      },
    },
    optimizeDeps: {
      // Radix dialog was introduced mid-session for the mobile sidebar sheet.
      // When Vite re-optimizes it on a running dev server, the browser can race
      // and request stale chunk paths from `.vite/deps`. Excluding it keeps dev
      // reloads stable instead of rewriting those chunk filenames under us.
      exclude: ["@radix-ui/react-dialog"],
    },
    build: {
      outDir: path.resolve(import.meta.dirname, "../nanobot/web/dist"),
      emptyOutDir: true,
      sourcemap: false,
    },
    server: {
      host: "127.0.0.1",
      port: 5173,
      strictPort: true,
      proxy: {
        "/webui": createProxyOptions("/webui", target),
        "/api": createProxyOptions("/api", target),
        "/auth": createProxyOptions("/auth", target),
        "/prePrefix": createProxyOptions("/prePrefix", target),
        "/proPrefix": createProxyOptions("/proPrefix", target),
      },
    },
  };
});
