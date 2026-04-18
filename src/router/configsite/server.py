"""Tiny FastAPI app for editing router.yaml + anchors.yaml in a browser.

Read/write happens directly against the on-disk files passed at startup.
Validation reuses the project's pydantic models so the form can never write
a config the router itself can't load.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router.config_loader import RouterConfig, load_anchors


# ----------------- request bodies -----------------

class ConfigUpdate(BaseModel):
    """Whatever the form posts; validated by RouterConfig before writing."""
    config: dict = Field(default_factory=dict)


class AnchorsUpdate(BaseModel):
    anchors: dict[str, list[str]] = Field(default_factory=dict)


# ----------------- file helpers -----------------

def _backup(path: Path) -> Path:
    if not path.exists():
        return path
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = path.with_suffix(path.suffix + f".bak-{ts}")
    shutil.copy2(path, bak)
    return bak


def _read_yaml(path: Path) -> dict | list:
    if not path.exists():
        raise HTTPException(404, f"file not found: {path}")
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, payload) -> None:
    text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    path.write_text(text)


# ----------------- app factory -----------------

def create_app(config_path: Path, anchors_path: Path) -> FastAPI:
    app = FastAPI(title="router-config", version="0.1.0")

    # Static index.html lives next to this module.
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = (static_dir / "index.html").read_text()
        return HTMLResponse(html)

    @app.get("/api/paths")
    def paths() -> dict:
        return {
            "config_path": str(config_path),
            "anchors_path": str(anchors_path),
        }

    @app.get("/api/config")
    def get_config() -> dict:
        return _read_yaml(config_path)

    @app.put("/api/config")
    def put_config(body: ConfigUpdate) -> dict:
        # Validate against the same pydantic schema the router uses.
        try:
            validated = RouterConfig.model_validate(body.config)
        except Exception as e:
            raise HTTPException(422, f"invalid config: {e}")
        # Refuse to write a config with no backends — that would brick the
        # router on its next call. The form should always send the full list.
        if not validated.backends:
            raise HTTPException(422, "config must define at least one backend")
        bak = _backup(config_path)
        _write_yaml(config_path, validated.model_dump(exclude_none=True))
        return {"ok": True, "backup": str(bak) if bak != config_path else None}

    @app.get("/api/anchors")
    def get_anchors() -> dict[str, list[str]]:
        return load_anchors(anchors_path)

    @app.put("/api/anchors")
    def put_anchors(body: AnchorsUpdate) -> dict:
        if not body.anchors:
            raise HTTPException(422, "anchors payload is empty")
        for label, exemplars in body.anchors.items():
            if not isinstance(exemplars, list) or not all(isinstance(e, str) for e in exemplars):
                raise HTTPException(422, f"anchors[{label}] must be list[str]")
        bak = _backup(anchors_path)
        _write_yaml(anchors_path, body.anchors)
        return {"ok": True, "backup": str(bak) if bak != anchors_path else None}

    @app.post("/api/rebuild-anchors")
    def rebuild_anchors() -> dict:
        # Rebuild the centroid cache so anchor edits actually take effect on
        # the next router invocation.
        from router.classifier.embed_anchors import EmbedAnchorsClassifier
        from router.config_loader import load_config

        cfg = load_config(config_path)
        clf = EmbedAnchorsClassifier(cfg.classifier)
        clf.rebuild()
        return {"ok": True}

    return app


# ----------------- CLI entrypoint -----------------

def _resolve_paths(config_arg: str | None, anchors_arg: str | None) -> tuple[Path, Path]:
    from router.orchestrator import find_default_config

    cfg_path = Path(config_arg) if config_arg else find_default_config()
    if anchors_arg:
        anc_path = Path(anchors_arg)
    else:
        # default: sibling of router.yaml, named anchors.yaml
        anc_path = cfg_path.parent / "anchors.yaml"
    return cfg_path.resolve(), anc_path.resolve()


def main() -> None:
    p = argparse.ArgumentParser(prog="router-config", description="Web UI for router.yaml + anchors.yaml.")
    p.add_argument("--config", default=None, help="Path to router.yaml (default: auto-detect).")
    p.add_argument("--anchors", default=None, help="Path to anchors.yaml (default: sibling of router.yaml).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write("uvicorn not installed. Run: pip install 'uvicorn[standard]' fastapi\n")
        sys.exit(2)

    cfg_path, anc_path = _resolve_paths(args.config, args.anchors)
    sys.stderr.write(f"[router-config] config={cfg_path}\n[router-config] anchors={anc_path}\n")
    sys.stderr.write(f"[router-config] http://{args.host}:{args.port}\n")
    app = create_app(cfg_path, anc_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
