import os
import uuid
from datetime import datetime, timedelta
from flask import render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from app import app, db
from models import Admin, Zone, Camera, Worker, Violation, Settings, WorkerFineTracking
from utils import generate_pdf_report, allowed_file, perform_ppe_detection, get_zone_ppe_requirements, get_detection_analytics

import json
import random

# Authentication decorator
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/')
def index():
    if 'admin_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Get settings for branding
    settings = Settings.query.first()
    if not settings:
        settings = Settings()
        db.session.add(settings)
        db.session.commit()
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        admin = Admin.query.filter_by(username=username).first()
        
        if admin and check_password_hash(admin.password_hash, password):
            session['admin_id'] = admin.id
            session['username'] = admin.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html', settings=settings)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get statistics
    total_cameras = Camera.query.count()
    total_zones = Zone.query.count()
    total_workers = Worker.query.count()
    total_violations = Violation.query.count()
    
    # Get recent violations (last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_violations = Violation.query.filter(Violation.timestamp >= yesterday).count()
    
    # Get violations by type for chart
    violation_types = db.session.query(
        Violation.violation_type,
        db.func.count(Violation.id).label('count')
    ).group_by(Violation.violation_type).all()
    
    # Get violations by zone for chart
    zone_violations = db.session.query(
        Zone.name,
        db.func.count(Violation.id).label('count')
    ).join(Violation).group_by(Zone.name).all()
    

    
    # Get camera status (simulate)
    cameras = Camera.query.all()
    active_cameras = len([c for c in cameras if c.is_active])
    
    # Get live detection analytics from database
    ai_analytics = get_detection_analytics()
    
    return render_template('dashboard.html',
                         total_cameras=total_cameras,
                         total_zones=total_zones,
                         total_workers=total_workers,
                         total_violations=total_violations,
                         recent_violations=recent_violations,
                         violation_types=violation_types,
                         zone_violations=zone_violations,
                         active_cameras=active_cameras,
                         ai_analytics=ai_analytics)

@app.route('/cameras')
@login_required
def cameras():
    cameras = Camera.query.join(Zone).all()
    zones = Zone.query.all()
    return render_template('cameras.html', cameras=cameras, zones=zones)

@app.route('/cameras/add', methods=['POST'])
@login_required
def add_camera():
    name = request.form['name']
    rtsp_url = request.form['rtsp_url']
    zone_id = request.form['zone_id']
    
    camera = Camera(name=name, rtsp_url=rtsp_url, zone_id=zone_id)
    db.session.add(camera)
    db.session.commit()
    
    flash('Camera added successfully!', 'success')
    return redirect(url_for('cameras'))

@app.route('/cameras/edit/<int:camera_id>', methods=['POST'])
@login_required
def edit_camera(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    camera.name = request.form['name']
    camera.rtsp_url = request.form['rtsp_url']
    camera.zone_id = request.form['zone_id']
    camera.is_active = 'is_active' in request.form
    
    db.session.commit()
    flash('Camera updated successfully!', 'success')
    return redirect(url_for('cameras'))

@app.route('/cameras/delete/<int:camera_id>')
@login_required
def delete_camera(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    db.session.delete(camera)
    db.session.commit()
    flash('Camera deleted successfully!', 'success')
    return redirect(url_for('cameras'))

@app.route('/zones')
@login_required
def zones():
    zones = Zone.query.all()
    return render_template('zones.html', zones=zones)

@app.route('/zones/add', methods=['POST'])
@login_required
def add_zone():
    name = request.form['name']
    description = request.form.get('description', '')
    
    zone = Zone(name=name, description=description)
    db.session.add(zone)
    db.session.commit()
    
    flash('Zone added successfully!', 'success')
    return redirect(url_for('zones'))

@app.route('/zones/edit/<int:zone_id>', methods=['POST'])
@login_required
def edit_zone(zone_id):
    zone = Zone.query.get_or_404(zone_id)
    zone.name = request.form['name']
    zone.description = request.form.get('description', '')
    
    db.session.commit()
    flash('Zone updated successfully!', 'success')
    return redirect(url_for('zones'))

@app.route('/zones/delete/<int:zone_id>')
@login_required
def delete_zone(zone_id):
    zone = Zone.query.get_or_404(zone_id)
    if zone.cameras:
        flash('Cannot delete zone with assigned cameras!', 'error')
    else:
        db.session.delete(zone)
        db.session.commit()
        flash('Zone deleted successfully!', 'success')
    return redirect(url_for('zones'))

@app.route('/workers')
@login_required
def workers():
    workers = Worker.query.all()
    return render_template('workers.html', workers=workers)

@app.route('/workers/add', methods=['POST'])
@login_required
def add_worker():
    employee_id = request.form['employee_id']
    full_name = request.form['full_name']
    department = request.form['department']
    job_role = request.form['job_role']
    whatsapp_number = request.form.get('whatsapp_number', '')
    
    # Handle file upload
    profile_picture = None
    if 'profile_picture' in request.files:
        file = request.files['profile_picture']
        if file and file.filename and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
            profile_picture = filename
    
    worker = Worker(
        employee_id=employee_id,
        full_name=full_name,
        department=department,
        job_role=job_role,
        whatsapp_number=whatsapp_number,
        profile_picture=profile_picture
    )
    
    try:
        db.session.add(worker)
        db.session.commit()
        flash('Worker added successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Employee ID already exists!', 'error')
    
    return redirect(url_for('workers'))

@app.route('/workers/edit/<int:worker_id>', methods=['POST'])
@login_required
def edit_worker(worker_id):
    worker = Worker.query.get_or_404(worker_id)
    worker.employee_id = request.form['employee_id']
    worker.full_name = request.form['full_name']
    worker.department = request.form['department']
    worker.job_role = request.form['job_role']
    worker.whatsapp_number = request.form.get('whatsapp_number', '')
    
    # Handle file upload
    if 'profile_picture' in request.files:
        file = request.files['profile_picture']
        if file and file.filename and allowed_file(file.filename):
            # Delete old file if exists
            if worker.profile_picture:
                old_path = os.path.join(app.config['UPLOAD_FOLDER'], worker.profile_picture)
                if os.path.exists(old_path):
                    os.remove(old_path)
            
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
            worker.profile_picture = filename
    
    try:
        db.session.commit()
        flash('Worker updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Employee ID already exists!', 'error')
    
    return redirect(url_for('workers'))

@app.route('/workers/delete/<int:worker_id>')
@login_required
def delete_worker(worker_id):
    worker = Worker.query.get_or_404(worker_id)
    
    # Delete profile picture if exists
    if worker.profile_picture:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], worker.profile_picture)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    db.session.delete(worker)
    db.session.commit()
    flash('Worker deleted successfully!', 'success')
    return redirect(url_for('workers'))

@app.route('/monitoring')
@login_required
def monitoring():
    cameras = Camera.query.join(Zone).all()
    return render_template('monitoring.html', cameras=cameras)

@app.route('/simulate_detection/<int:camera_id>')
@login_required
def simulate_detection(camera_id):
    """Perform AI-powered PPE detection on live camera feed"""
    camera = Camera.query.get_or_404(camera_id)
    
    # Get zone-specific PPE requirements
    zone_requirements = get_zone_ppe_requirements(camera.zone)
    
    # Perform AI-powered PPE detection
    detection_result = perform_ppe_detection(camera, zone_requirements)
    
    if detection_result['violation']:
        # Create violation record
        settings = Settings.query.first()
        if not settings:
            settings = Settings()
            db.session.add(settings)
            db.session.commit()
        

        

        
        violation = Violation(
            camera_id=camera.id,
            zone_id=camera.zone_id,
            violation_type=detection_result['violation_type'],
            confidence=detection_result['confidence']
        )
        
        # Worker identification through face recognition would be implemented here
        # For now, violations are created without worker assignment until facial recognition is configured
        
        db.session.add(violation)
        db.session.commit()
        
        # Automatically create fine record if worker is identified
        if detected_worker:
            from utils import create_fine_record_for_violation
            fine_record = create_fine_record_for_violation(violation)
            if fine_record:
                flash(f'Fine record automatically created for {detected_worker.full_name}: ${fine_record.fine_amount}', 'info')
        

        db.session.commit()
        
        # Log violation detection
        logging.info(f"PPE violation detected: {detection_result['violation_type']} in {camera.zone.name}")
    
    return jsonify(detection_result)

@app.route('/violations')
@login_required
def violations():
    page = request.args.get('page', 1, type=int)
    violations = Violation.query.join(Camera).join(Zone).order_by(Violation.timestamp.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template('violations.html', violations=violations)

@app.route('/violations/resolve/<int:violation_id>')
@login_required
def resolve_violation(violation_id):
    violation = Violation.query.get_or_404(violation_id)
    violation.is_resolved = True
    db.session.commit()
    flash('Violation marked as resolved!', 'success')
    return redirect(url_for('violations'))

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    settings = Settings.query.first()
    if not settings:
        settings = Settings()
        db.session.add(settings)
        db.session.commit()
    
    if request.method == 'POST':
        # Update company name
        settings.company_name = request.form.get('company_name', '').strip()
        
        # Update detection confidence
        try:
            confidence = float(request.form.get('detection_confidence', 0.7))
            if 0.1 <= confidence <= 1.0:
                settings.detection_confidence = confidence
        except ValueError:
            flash('Invalid confidence threshold value!', 'error')
            return redirect(url_for('settings'))
        

        
        # Handle logo upload
        if 'company_logo' in request.files:
            file = request.files['company_logo']
            if file and file.filename != '' and allowed_file(file.filename):
                # Create uploads directory if it doesn't exist
                upload_folder = app.config.get('UPLOAD_FOLDER', 'static/uploads')
                os.makedirs(upload_folder, exist_ok=True)
                
                # Generate secure filename
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                
                # Save file
                file_path = os.path.join(upload_folder, filename)
                file.save(file_path)
                
                # Update settings
                settings.company_logo = filename
        
        # Update timestamp
        settings.updated_at = datetime.utcnow()
        
        try:
            db.session.commit()
            flash('Company settings updated successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash('Error saving settings. Please try again.', 'error')
        
        return redirect(url_for('settings'))
    
    return render_template('settings.html', settings=settings)

@app.route('/change_admin_credentials', methods=['POST'])
@login_required
def change_admin_credentials():
    current_password = request.form.get('current_password')
    new_username = request.form.get('new_username', '').strip()
    new_password = request.form.get('new_password', '').strip()
    confirm_password = request.form.get('confirm_password', '').strip()
    
    # Get current admin
    admin = Admin.query.get(session['admin_id'])
    if not admin:
        flash('Admin account not found!', 'error')
        return redirect(url_for('settings'))
    
    # Verify current password
    if not check_password_hash(admin.password_hash, current_password):
        flash('Current password is incorrect!', 'error')
        return redirect(url_for('settings'))
    
    # Validate and update username if provided
    if new_username:
        if len(new_username) < 3:
            flash('Username must be at least 3 characters long!', 'error')
            return redirect(url_for('settings'))
        
        # Check if username already exists (excluding current admin)
        existing_admin = Admin.query.filter(Admin.username == new_username, Admin.id != admin.id).first()
        if existing_admin:
            flash('Username already exists!', 'error')
            return redirect(url_for('settings'))
        
        admin.username = new_username
        session['username'] = new_username
    
    # Validate and update password if provided
    if new_password:
        if len(new_password) < 6:
            flash('New password must be at least 6 characters long!', 'error')
            return redirect(url_for('settings'))
        
        if new_password != confirm_password:
            flash('New passwords do not match!', 'error')
            return redirect(url_for('settings'))
        
        admin.password_hash = generate_password_hash(new_password)
    
    # Check if any changes were made
    if not new_username and not new_password:
        flash('No changes were made!', 'info')
        return redirect(url_for('settings'))
    
    try:
        db.session.commit()
        
        # Build success message
        changes = []
        if new_username:
            changes.append('username')
        if new_password:
            changes.append('password')
        
        flash(f'Admin {" and ".join(changes)} updated successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash('Error updating admin credentials. Please try again.', 'error')
    
    return redirect(url_for('settings'))

@app.route('/api/check_database')
@login_required
def check_database():
    """API endpoint to check database connection status"""
    try:
        # Try to execute a simple query
        db.session.execute(db.text('SELECT 1'))
        db.session.commit()
        return jsonify({
            'connected': True,
            'status': 'Connected',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'connected': False,
            'status': 'Disconnected',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })



@app.route('/reports')
@login_required
def reports():
    # Get statistics for the reports page
    total_violations = Violation.query.count()
    total_workers = Worker.query.count()
    total_zones = Zone.query.count()

    zones = Zone.query.all()
    
    return render_template('reports.html',
                         total_violations=total_violations,
                         total_workers=total_workers,
                         total_zones=total_zones,
                         zones=zones)

@app.route('/generate_report')
@login_required
def generate_report():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    department = request.args.get('department')
    zone_id = request.args.get('zone_id')
    
    # Build query
    query = Violation.query.join(Camera).join(Zone)
    
    if start_date:
        query = query.filter(Violation.timestamp >= datetime.strptime(start_date, '%Y-%m-%d'))
    if end_date:
        query = query.filter(Violation.timestamp <= datetime.strptime(end_date + ' 23:59:59', '%Y-%m-%d %H:%M:%S'))
    if zone_id:
        query = query.filter(Zone.id == zone_id)
    if department:
        query = query.join(Worker).filter(Worker.department == department)
    
    violations = query.all()
    
    # Generate PDF
    pdf_path = generate_pdf_report(violations, start_date, end_date, department)
    
    return send_file(pdf_path, as_attachment=True, download_name='ppe_violations_report.pdf')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))





@app.route('/fine-tracking')
@login_required
def fine_tracking():
    """Intelligent Worker Fine Tracking Dashboard"""
    # Get filter parameters
    filter_status = request.args.get('status', 'all')
    filter_worker = request.args.get('worker', 'all')
    filter_period = request.args.get('period', '30')  # days
    
    # Base query for fine tracking records
    query = WorkerFineTracking.query
    
    # Apply filters
    if filter_status != 'all':
        query = query.filter(WorkerFineTracking.payment_status == filter_status)
    
    if filter_worker != 'all':
        query = query.filter(WorkerFineTracking.worker_id == int(filter_worker))
    
    # Apply time filter
    if filter_period != 'all':
        days_ago = datetime.utcnow() - timedelta(days=int(filter_period))
        query = query.filter(WorkerFineTracking.created_at >= days_ago)
    
    # Get paginated results
    page = request.args.get('page', 1, type=int)
    fine_records = query.order_by(WorkerFineTracking.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    
    # Calculate analytics
    analytics = calculate_fine_analytics(filter_period)
    
    # Get all workers for filter dropdown
    workers = Worker.query.all()
    
    return render_template('fine_tracking.html',
                         fine_records=fine_records,
                         analytics=analytics,
                         workers=workers,
                         current_filters={
                             'status': filter_status,
                             'worker': filter_worker,
                             'period': filter_period
                         },
                         moment=datetime,
                         datetime=datetime)

@app.route('/fine-tracking/add', methods=['GET', 'POST'])
@login_required
def add_fine_record():
    """Add new fine record"""
    if request.method == 'POST':
        try:
            # Get form data
            worker_id = request.form.get('worker_id')
            violation_id = request.form.get('violation_id')
            fine_amount = float(request.form.get('fine_amount', 0))
            payment_status = request.form.get('payment_status', 'pending')
            due_date_str = request.form.get('due_date')
            notes = request.form.get('notes', '')
            
            # Parse due date
            due_date = None
            if due_date_str:
                due_date = datetime.strptime(due_date_str, '%Y-%m-%d')
            
            # Create fine record
            fine_record = WorkerFineTracking(
                worker_id=worker_id,
                violation_id=violation_id,
                fine_amount=fine_amount,
                payment_status=payment_status,
                due_date=due_date,
                notes=notes,
                processed_by=session.get('admin_username', 'admin')
            )
            
            db.session.add(fine_record)
            db.session.commit()
            
            flash('Fine record added successfully!', 'success')
            return redirect(url_for('fine_tracking'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding fine record: {str(e)}', 'error')
    
    # Get workers and unprocessed violations
    workers = Worker.query.all()
    unprocessed_violations = Violation.query.filter(
        ~Violation.id.in_(db.session.query(WorkerFineTracking.violation_id))
    ).all()
    
    return render_template('add_fine_record.html',
                         workers=workers,
                         violations=unprocessed_violations,
                         now=datetime.utcnow,
                         timedelta=timedelta)

@app.route('/fine-tracking/update/<int:fine_id>', methods=['POST'])
@login_required
def update_fine_status():
    """Update fine payment status"""
    fine_id = request.view_args['fine_id']
    fine_record = WorkerFineTracking.query.get_or_404(fine_id)
    
    try:
        # Update payment status
        fine_record.payment_status = request.form.get('payment_status')
        fine_record.payment_method = request.form.get('payment_method')
        fine_record.notes = request.form.get('notes', fine_record.notes)
        
        # Set payment date if marked as paid
        if fine_record.payment_status == 'paid':
            fine_record.payment_date = datetime.utcnow()
        
        fine_record.processed_by = session.get('admin_username', 'admin')
        fine_record.updated_at = datetime.utcnow()
        
        db.session.commit()
        flash('Fine record updated successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating fine record: {str(e)}', 'error')
    
    return redirect(url_for('fine_tracking'))

@app.route('/fine-tracking/worker-summary/<int:worker_id>')
@login_required
def worker_fine_summary():
    """Detailed fine summary for specific worker"""
    worker = Worker.query.get_or_404(worker_id)
    
    # Get all fine records for this worker
    fine_records = WorkerFineTracking.query.filter_by(worker_id=worker_id).order_by(
        WorkerFineTracking.created_at.desc()
    ).all()
    
    # Calculate worker-specific analytics
    total_fines = sum(record.fine_amount for record in fine_records)
    paid_fines = sum(record.fine_amount for record in fine_records if record.payment_status == 'paid')
    pending_fines = sum(record.fine_amount for record in fine_records if record.payment_status == 'pending')
    
    # Get violation history
    violations = Violation.query.filter_by(worker_id=worker_id).order_by(
        Violation.timestamp.desc()
    ).limit(20).all()
    
    worker_analytics = {
        'total_fines': total_fines,
        'paid_fines': paid_fines,
        'pending_fines': pending_fines,
        'total_violations': len(violations),
        'payment_rate': (paid_fines / total_fines * 100) if total_fines > 0 else 0
    }
    
    return render_template('worker_fine_summary.html',
                         worker=worker,
                         fine_records=fine_records,
                         violations=violations,
                         analytics=worker_analytics,
                         moment=datetime)

def calculate_fine_analytics(period_days='30'):
    """Calculate comprehensive fine tracking analytics"""
    try:
        # Time filter
        if period_days != 'all':
            days_ago = datetime.utcnow() - timedelta(days=int(period_days))
            date_filter = WorkerFineTracking.created_at >= days_ago
        else:
            date_filter = True
        
        # Basic totals
        total_fines = db.session.query(db.func.sum(WorkerFineTracking.fine_amount)).filter(date_filter).scalar() or 0
        total_records = WorkerFineTracking.query.filter(date_filter).count()
        
        # Payment status breakdown
        status_breakdown = db.session.query(
            WorkerFineTracking.payment_status,
            db.func.count(WorkerFineTracking.id),
            db.func.sum(WorkerFineTracking.fine_amount)
        ).filter(date_filter).group_by(WorkerFineTracking.payment_status).all()
        
        # Top violators (workers with most fines)
        top_violators = db.session.query(
            Worker.id.label('worker_id'),
            Worker.employee_id,
            Worker.full_name,
            db.func.count(WorkerFineTracking.id).label('violation_count'),
            db.func.sum(WorkerFineTracking.fine_amount).label('total_amount')
        ).join(WorkerFineTracking).filter(date_filter).group_by(
            Worker.id, Worker.employee_id, Worker.full_name
        ).order_by(db.desc('total_amount')).limit(10).all()
        
        # Monthly trends (last 12 months)
        monthly_trends = []
        for i in range(12):
            month_start = (datetime.utcnow().replace(day=1) - timedelta(days=30*i)).replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            monthly_data = db.session.query(
                db.func.count(WorkerFineTracking.id),
                db.func.sum(WorkerFineTracking.fine_amount)
            ).filter(
                WorkerFineTracking.created_at >= month_start,
                WorkerFineTracking.created_at <= month_end
            ).first()
            
            monthly_trends.append({
                'month': month_start.strftime('%Y-%m'),
                'count': monthly_data[0] or 0,
                'amount': float(monthly_data[1] or 0)
            })
        
        monthly_trends.reverse()
        
        # Violation type analysis
        violation_types = db.session.query(
            Violation.violation_type,
            db.func.count(WorkerFineTracking.id),
            db.func.sum(WorkerFineTracking.fine_amount)
        ).join(WorkerFineTracking).filter(date_filter).group_by(
            Violation.violation_type
        ).all()
        
        return {
            'total_fines': float(total_fines),
            'total_records': total_records,
            'status_breakdown': [
                {'status': status, 'count': count, 'amount': float(amount or 0)}
                for status, count, amount in status_breakdown
            ],
            'top_violators': [
                {
                    'worker_id': worker_id,
                    'employee_id': emp_id,
                    'name': name,
                    'violation_count': count,
                    'total_amount': float(amount)
                }
                for worker_id, emp_id, name, count, amount in top_violators
            ],
            'monthly_trends': monthly_trends,
            'violation_types': [
                {'type': vtype, 'count': count, 'amount': float(amount or 0)}
                for vtype, count, amount in violation_types
            ]
        }
        
    except Exception as e:
        logging.error(f"Error calculating fine analytics: {e}")
        return {
            'total_fines': 0,
            'total_records': 0,
            'status_breakdown': [],
            'top_violators': [],
            'monthly_trends': [],
            'violation_types': []
        }

# Seed data function removed - system now uses live data only
