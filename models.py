from app import db
from datetime import datetime

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Zone(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    cameras = db.relationship('Camera', backref='zone', lazy=True, cascade='all, delete-orphan')
    violations = db.relationship('Violation', backref='zone', lazy=True)

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    rtsp_url = db.Column(db.String(500), nullable=False)
    zone_id = db.Column(db.Integer, db.ForeignKey('zone.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    violations = db.relationship('Violation', backref='camera', lazy=True)

class Worker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(50), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    job_role = db.Column(db.String(100), nullable=False)
    whatsapp_number = db.Column(db.String(20))
    profile_picture = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    violations = db.relationship('Violation', backref='worker', lazy=True)

class Violation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    worker_id = db.Column(db.Integer, db.ForeignKey('worker.id'), nullable=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    zone_id = db.Column(db.Integer, db.ForeignKey('zone.id'), nullable=False)
    violation_type = db.Column(db.String(100), nullable=False)  # helmet, vest, gloves, etc.
    confidence = db.Column(db.Float, default=0.0)
    fine_amount = db.Column(db.Float, default=0.0)
    screenshot_path = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_resolved = db.Column(db.Boolean, default=False)

    notification_sent = db.Column(db.Boolean, default=False)





class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_name = db.Column(db.String(200), default="PPE Detection System")
    company_logo = db.Column(db.String(200))
    detection_confidence = db.Column(db.Float, default=0.7)
    # Fine amounts for different violations
    helmet_fine = db.Column(db.Float, default=50.0)
    vest_fine = db.Column(db.Float, default=30.0)
    gloves_fine = db.Column(db.Float, default=25.0)
    mask_fine = db.Column(db.Float, default=40.0)
    glasses_fine = db.Column(db.Float, default=20.0)
    boots_fine = db.Column(db.Float, default=35.0)
    # Currency setting
    currency = db.Column(db.String(10), default="USD")
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class WorkerFineTracking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    worker_id = db.Column(db.Integer, db.ForeignKey('worker.id'), nullable=False)
    violation_id = db.Column(db.Integer, db.ForeignKey('violation.id'), nullable=False)
    fine_amount = db.Column(db.Float, nullable=False)
    payment_status = db.Column(db.String(20), default='pending')  # pending, paid, waived, dispute
    payment_method = db.Column(db.String(50))  # cash, bank_transfer, payroll_deduction
    payment_date = db.Column(db.DateTime)
    due_date = db.Column(db.DateTime)
    late_fee = db.Column(db.Float, default=0.0)
    discount_applied = db.Column(db.Float, default=0.0)
    notes = db.Column(db.Text)
    processed_by = db.Column(db.String(100))  # Admin who processed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    worker = db.relationship('Worker', backref='fine_records')
    violation = db.relationship('Violation', backref='fine_tracking')
