"""
Know-Surgery New UI - Flask Backend

启动: python new_ui/backend/app.py --port 5000
生产部署时用 Vite build 产出托管静态文件。
开发时前端 Vite dev server 通过 proxy 连接此后端。
"""

import argparse
from pathlib import Path

from flask import Flask, send_from_directory
from flask_cors import CORS

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIST), static_url_path="")
    CORS(app)

    from .api import config_bp, runner_bp, results_bp, skills_bp, agent_bp, data_bp
    app.register_blueprint(config_bp)
    app.register_blueprint(runner_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(skills_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(data_bp)

    @app.route("/")
    @app.route("/workshop")
    @app.route("/monitor")
    @app.route("/results")
    @app.route("/skills")
    def spa_entry():
        if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
            resp = send_from_directory(str(FRONTEND_DIST), "index.html")
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp
        return "<h3>Frontend not built yet. Run <code>cd new_ui/frontend && npm run build</code></h3>", 200

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=15000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=True)
