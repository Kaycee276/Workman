import json
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Query, Security, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config
from src import state

_LOG_RANGES = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "3d": timedelta(days=3),
}

app = FastAPI(title="Workman API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["Authorization", "Content-Type"],
)

_bearer = HTTPBearer(auto_error=False)


def _check_header(credentials: HTTPAuthorizationCredentials | None) -> None:
    """Verify Bearer token from Authorization header. No-op if DASHBOARD_TOKEN is unset."""
    if not config.DASHBOARD_TOKEN:
        return
    if not credentials or credentials.credentials != config.DASHBOARD_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.api_route("/api/health", methods=["GET", "HEAD"])
async def health():
    return {"ok": True}


@app.get("/api/status")
async def api_status(credentials: HTTPAuthorizationCredentials | None = Security(_bearer)):
    _check_header(credentials)
    return {"issues": state.get_all(), "steps": state.STEPS}


@app.get("/api/logs")
async def api_logs(
    range: str = Query("1h"),
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
):
    _check_header(credentials)
    if range not in _LOG_RANGES:
        raise HTTPException(status_code=400, detail=f"range must be one of: {', '.join(_LOG_RANGES)}")
    since = (datetime.now(timezone.utc) - _LOG_RANGES[range]).isoformat()
    return {"logs": state.get_logs_since(since), "range": range}


# WebSocket auth uses a query parameter because browsers cannot set custom
# headers on WebSocket upgrade requests. The token is not logged by uvicorn
# at the default log level, but be aware it may appear in proxy / CDN access
# logs. Use DASHBOARD_TOKEN only in trusted network environments.
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, token: str = Query(default="")):
    await websocket.accept()
    if config.DASHBOARD_TOKEN and token != config.DASHBOARD_TOKEN:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    state.register_ws(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "init",
            "issues": state.get_all(),
            "steps": state.STEPS,
        }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        state.unregister_ws(websocket)
