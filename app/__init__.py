import os

from flask import Flask
from dotenv import load_dotenv


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__, template_folder="../templates")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    upload_dir = os.path.join(base_dir, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-me")
    app.config["UPLOAD_FOLDER"] = upload_dir
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    app.config["STRIPE_SECRET_KEY"] = os.getenv("STRIPE_SECRET_KEY", "")
    app.config["STRIPE_WEBHOOK_SECRET"] = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    app.config["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    app.config["OPENAI_MODEL"] = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    app.config["APP_BASE_URL"] = os.getenv("APP_BASE_URL", "http://localhost:8000")

    from app.routes import main_bp

    app.register_blueprint(main_bp)
    return app
