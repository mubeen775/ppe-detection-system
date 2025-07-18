import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    # Fallback to SQLite for development
    database_url = "sqlite:///ppe_detection.db"

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the app with the extension
db.init_app(app)

# Import routes and models
with app.app_context():
    import models
    import routes
    db.create_all()
    
    # Create default admin user if none exists
    from models import Admin
    from werkzeug.security import generate_password_hash
    
    if not Admin.query.first():
        default_admin = Admin(
            username="admin",
            password_hash=generate_password_hash("admin123"),
            email="admin@ppe-system.com"
        )
        db.session.add(default_admin)
        db.session.commit()
        logging.info("Default admin user created: admin/admin123")
