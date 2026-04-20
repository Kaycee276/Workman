export type Step =
  | "queued"
  | "detected"
  | "fetching"
  | "forking"
  | "cloning"
  | "setup"
  | "solving"
  | "pushing"
  | "done";

export interface Issue {
  id: string;
  title: string;
  step: Step;
  failed: boolean;
  pr_url: string | null;
  error: string | null;
  started_at: string;
  updated_at: string;
}

export interface LogEntry {
  issue_id: string | null;
  message: string;
  ts: string;
}

export type WsMessage =
  | { type: "init"; issues: Issue[]; steps: Step[] }
  | { type: "issue_update"; issue: Issue }
  | { type: "log"; issue_id: string | null; message: string; ts: string };
