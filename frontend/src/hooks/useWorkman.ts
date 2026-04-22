import { useEffect, useRef, useState, useCallback } from "react";
import type { Issue, LogEntry, LogRange, Step, WsMessage } from "../types";

const MAX_LOGS = 10000;

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const DASHBOARD_TOKEN: string = import.meta.env.VITE_DASHBOARD_TOKEN ?? "";
const WS_BASE = API_URL.replace(/^https/, "wss").replace(/^http/, "ws") + "/ws";
const WS_URL = DASHBOARD_TOKEN
  ? `${WS_BASE}?token=${encodeURIComponent(DASHBOARD_TOKEN)}`
  : WS_BASE;

export function useWorkman() {
  const [issues, setIssues] = useState<Record<string, Issue>>({});
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [steps, setSteps] = useState<Step[]>([]);
  const [connected, setConnected] = useState(false);
  const [range, setRange] = useState<LogRange>("1h");
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const heartbeatRef = useRef<ReturnType<typeof setInterval>>(undefined);
  const connectRef = useRef<() => void>(() => {});

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      // Render's proxy drops idle connections after ~55s — ping every 30s
      heartbeatRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send("ping");
      }, 30000);
    };
    ws.onclose = () => {
      setConnected(false);
      clearInterval(heartbeatRef.current);
      retryRef.current = setTimeout(() => connectRef.current(), 3000);
    };
    ws.onerror = () => ws.close();

    ws.onmessage = (e) => {
      const msg: WsMessage = JSON.parse(e.data);
      if (msg.type === "init") {
        if (msg.steps?.length) setSteps(msg.steps);
        setIssues(Object.fromEntries(msg.issues.map((i) => [i.id, i])));
      } else if (msg.type === "issue_update") {
        setIssues((prev) => ({ ...prev, [msg.issue.id]: msg.issue }));
      } else if (msg.type === "log") {
        setLogs((prev) => {
          const next = [
            ...prev,
            { issue_id: msg.issue_id, message: msg.message, ts: msg.ts },
          ];
          return next.length > MAX_LOGS ? next.slice(-MAX_LOGS) : next;
        });
      }
    };
  }, []);

  useEffect(() => {
    connectRef.current = connect;
    connect();
    return () => {
      clearTimeout(retryRef.current);
      clearInterval(heartbeatRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  useEffect(() => {
    const controller = new AbortController();
    const headers: HeadersInit = DASHBOARD_TOKEN
      ? { Authorization: `Bearer ${DASHBOARD_TOKEN}` }
      : {};
    fetch(`${API_URL}/api/logs?range=${range}`, {
      headers,
      signal: controller.signal,
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((data: { logs: LogEntry[] }) => {
        const historical = data.logs ?? [];
        setLogs((prev) => {
          // Merge historical logs with any real-time logs received during fetch
          const seen = new Set(historical.map((l) => `${l.ts}-${l.issue_id}`));
          const newRealtime = prev.filter(
            (l) => !seen.has(`${l.ts}-${l.issue_id}`),
          );
          const merged = [...historical, ...newRealtime];
          merged.sort(
            (a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime(),
          );
          return merged.slice(-MAX_LOGS);
        });
      })
      .catch((err) => {
        if (err?.name !== "AbortError")
          console.error("Failed to fetch logs:", err);
      });
    return () => controller.abort();
  }, [range]);

  return { issues, logs, steps, connected, range, setRange };
}
