from __future__ import annotations

from pathlib import Path


def create_app(fingerprint_dir: str = "results/fingerprints"):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="LLM Integrity Fingerprint Prototype", version="0.1.0")
    root = Path(fingerprint_dir)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/fingerprints")
    def list_fingerprints():
        return {"items": [path.stem for path in sorted(root.glob("*.json"))] if root.exists() else []}

    @app.get("/fingerprints/{name}")
    def get_fingerprint(name: str):
        path = root / f"{name}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Fingerprint not found")
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    return app


try:
    app = create_app()
except ImportError:  # FastAPI is an optional dependency for the prototype service.
    app = None
