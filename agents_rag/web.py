"""Lightweight HTTP server for the agents-rag knowledge base browser."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from . import db
from .config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


class Handler(SimpleHTTPRequestHandler):
    db_path: str = ""
    _conn = None

    @classmethod
    def get_conn(cls):
        if cls._conn is None:
            cls._conn = db.init_db(cls.db_path)
        return cls._conn

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/chunks":
            self._serve_chunks()
        elif self.path == "/api/stats":
            self._serve_stats()
        elif self.path.startswith("/api/queries"):
            self._serve_queries()
        else:
            self.send_error(404)

    def _json_response(self, data: object, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self) -> None:
        html_path = Path(__file__).parent / "static" / "index.html"
        if not html_path.exists():
            self.send_error(500, "index.html not found")
            return
        body = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_chunks(self) -> None:
        conn = self.get_conn()
        meta, _ = db.load_all_chunks(conn)
        if not meta:
            self._json_response({"error": "No index found. Run reindex first."}, 404)
            return
        for m in meta:
            text = m.get("text", "")
            m["preview"] = text[:200] if text else ""
            m.pop("text", None)
        self._json_response(meta)

    def _serve_stats(self) -> None:
        conn = self.get_conn()
        meta, _ = db.load_all_chunks(conn)
        deprecated = db.load_deprecated(conn)
        if not meta:
            self._json_response({"total": 0, "sources": {}, "tags": {}, "deprecated": 0})
            return
        source_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        for m in meta:
            source_counts[m.get("source", "unknown")] += 1
            for t in m.get("tags", []):
                tag_counts[t] += 1
        self._json_response({
            "total": len(meta),
            "sources": dict(source_counts.most_common()),
            "tags": dict(tag_counts.most_common(200)),
            "deprecated": len(deprecated),
        })

    def _serve_queries(self) -> None:
        conn = self.get_conn()
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        limit = int(qs.get("limit", ["50"])[0])
        entries = db.get_recent_queries(conn, limit=limit)
        self._json_response(entries)

    def log_message(self, format, *args):  # noqa: A002
        log.info(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="agents-rag web UI")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENTS_RAG_WEB_PORT", "8808")))
    args = parser.parse_args()

    config = load_config()
    cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/agents-rag"))
    Handler.db_path = os.path.join(cache_dir, "agents-rag.db")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    log.info("Serving on http://127.0.0.1:%d  (db=%s)", args.port, Handler.db_path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
