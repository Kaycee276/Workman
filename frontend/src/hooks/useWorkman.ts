import { useEffect, useRef, useState, useCallback } from "react";
import type { Issue, LogEntry, Step, WsMessage } from "../types";

const MAX_LOGS = 2000;

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const DASHBOARD_TOKEN: string = import.meta.env.VITE_DASHBOARD_TOKEN ?? "";
const WS_BASE = API_URL.replace(/^https/, "wss").replace(/^http/, "ws") + "/ws";
const WS_URL = DASHBOARD_TOKEN ? `${WS_BASE}?token=${encodeURIComponent(DASHBOARD_TOKEN)}` : WS_BASE;

export function useWorkman() {
  const [issues, setIssues] = useState<Record<string, Issue>>({});
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [steps, setSteps] = useState<Step[]>([]);
  const [connected, setConnected] = useState(false);
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
          const next = [...prev, { issue_id: msg.issue_id, message: msg.message, ts: msg.ts }];
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

  return { issues, logs, steps, connected };
}
