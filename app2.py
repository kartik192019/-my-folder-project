from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
import os
from datetime import datetime
import requests
import json
import logging
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import chardet
import aspose.words as aw
import tiktoken
from supabase import create_client, Client
from config import Config

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configure logging
logger = logging.getLogger(__name__)

# Add custom filter for JSON parsing in templates
@app.template_filter('from_json')
def from_json_filter(value):
    """Custom filter to parse JSON in templates"""
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value

# Database configuration
DB_CONFIG = Config.get_db_config()

# Supabase configuration
supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

# Backend AI analyzer URL
AI_ANALYZER_URL = 'http://localhost:5001/analyze'
CV_ANALYZER_URL = 'http://localhost:5002/analyze_resumes'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

# Token counter functions
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def validate_token_limit(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> bool:
    """Check if the text exceeds the maximum token limit."""
    return count_tokens(text, model) <= max_tokens

def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to stay within the specified token limit."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_TOKENS'] = 400000  # Default max tokens for analysis

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_doc_to_docx(file_obj, filename):
    """Convert .doc file to .docx format in memory"""
    try:
        # Reset file pointer to beginning
        file_obj.seek(0)
        
        # Create a temporary file path for the input
        import tempfile
        import os
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as temp_doc:
            temp_doc.write(file_obj.read())
            temp_doc_path = temp_doc.name
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_docx:
            temp_docx_path = temp_docx.name
        
        try:
            # Load the .doc document
            doc = aw.Document(temp_doc_path)
            
            # Save the document as .docx
            doc.save(temp_docx_path)
            
            # Read the converted .docx file
            with open(temp_docx_path, 'rb') as converted_file:
                converted_content = converted_file.read()
            
            return converted_content
        
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_doc_path)
                os.unlink(temp_docx_path)
            except:
                pass
    
    except Exception as e:
        print(f"Error converting .doc to .docx: {e}")
        raise Exception(f"Failed to convert .doc file: {str(e)}")

def upload_to_supabase_storage(file, filename):
    """Upload file to Supabase storage and return the public URL"""
    try:
        print(f"Starting Supabase upload for {filename}")
        
        # Check if it's a .doc file and convert to .docx
        file_extension = os.path.splitext(filename)[1].lower()
        original_filename = filename
        
        if file_extension == '.doc':
            print(f"Converting .doc file to .docx format...")
            try:
                # Convert .doc to .docx
                converted_content = convert_doc_to_docx(file, filename)
                
                # Update filename to .docx
                filename = os.path.splitext(filename)[0] + '.docx'
                print(f"Converted {original_filename} to {filename}")
                
                # Use converted content for upload
                file_content = converted_content
                content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            except Exception as e:
                print(f"Failed to convert .doc to .docx: {e}")
                # Fallback to original file
                file.seek(0)
                file_content = file.read()
                file.seek(0)
                content_type = file.content_type
        else:
            # Read original file content
            file_content = file.read()
            file.seek(0)  # Reset file pointer for later use
            content_type = file.content_type
        
        # Upload to Supabase storage - use resume bucket
        bucket_name = Config.SUPABASE_RESUME_BUCKET
        file_path = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        
        print(f"Uploading to bucket: {bucket_name}, path: {file_path}")
        
        # Upload the file
        result = supabase.storage.from_(bucket_name).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": content_type}
        )
        
        print(f"Upload result: {result}")
        
        if result:
            # Get the public URL
            public_url = supabase.storage.from_(bucket_name).get_public_url(file_path)
            print(f"Generated public URL: {public_url}")
            return public_url, file_path
        else:
            raise Exception("Failed to upload file to Supabase storage")
    
    except Exception as e:
        print(f"Supabase upload error: {e}")
        # Fallback to local storage if Supabase fails
        print("Falling back to local storage...")
        
        # Handle .doc conversion for local storage too
        if file_extension == '.doc':
            try:
                converted_content = convert_doc_to_docx(file, filename)
                local_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                local_path = os.path.join(app.config['UPLOAD_FOLDER'], local_filename)
                
                # Save converted content
                with open(local_path, 'wb') as f:
                    f.write(converted_content)
                
                print(f"Saved converted .docx to local path: {local_path}")
                return None, local_filename
            except Exception as conv_e:
                print(f"Failed to convert .doc for local storage: {conv_e}")
                # Fallback to original file
                file.seek(0)
                local_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_filename}"
                local_path = os.path.join(app.config['UPLOAD_FOLDER'], local_filename)
                file.save(local_path)
                print(f"Saved original .doc to local path: {local_path}")
                return None, local_filename
        else:
            local_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            local_path = os.path.join(app.config['UPLOAD_FOLDER'], local_filename)
            file.save(local_path)
            print(f"Saved to local path: {local_path}")
            return None, local_filename

def clean_text(text):
    """Clean text to remove null characters and problematic content"""
    if not text:
        return ""
    
    # Remove null characters and other problematic control characters
    text = text.replace('\x00', '')
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Normalize whitespace but preserve line breaks
    lines = text.split('\n')
    cleaned_lines = [' '.join(line.split()) for line in lines]
    text = '\n'.join(line for line in cleaned_lines if line.strip())
    
    return text

def get_db_connection():
    """Get database connection with better error handling"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

@app.route('/')
def index():
    """Main page with job description upload form"""
    return redirect(url_for('render_upload_page'))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            return jsonify({
                'status': 'healthy',
                'service': 'resume_upload',
                'database': 'connected'
            })
        else:
            return jsonify({
                'status': 'unhealthy',
                'service': 'resume_upload',
                'database': 'disconnected'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'resume_upload',
            'error': str(e)
        }), 500

@app.route('/upload', methods=['GET'])
def render_upload_page():
    """Render the resume upload form with JD and criteria selection (GET request)"""
    # Get available job descriptions for dropdown
    conn = get_db_connection()
    job_descriptions = []
    criteria_list = []
    
    if conn:
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Fetch job descriptions
            cur.execute("""
                SELECT jd_id, title, company_id, user_id, created_at, jd_file, file_id
                FROM job_descriptions
                ORDER BY created_at DESC
                LIMIT 50
            """)
            job_descriptions = cur.fetchall()
            
            # Fetch criteria
            cur.execute("""
                SELECT criteria_id, criteria_name, parameter, weightage, created_at, company_id
                FROM criteria
                ORDER BY created_at DESC
                LIMIT 50
            """)
            criteria_list = cur.fetchall()
            
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error fetching data: {e}")
    
    return render_template('upload_resumes.html', job_descriptions=job_descriptions, criteria_list=criteria_list)

@app.route('/upload', methods=['POST'])
def upload_resumes():
    """Handle multiple resume uploads and processing"""
    try:
        if 'files' not in request.files:
            flash('No files provided')
            return redirect(request.url)
        
        files = request.files.getlist('files')
        jd_id = request.form.get('jd_id')
        criteria_id = request.form.get('criteria_id')  # Get selected criteria
        
        if not jd_id:
            flash('Please select a job description')
            return redirect(request.url)
        
        if not files or all(file.filename == '' for file in files):
            flash('No files selected')
            return redirect(request.url)
        
        # Validate JD exists and get its details
        conn = get_db_connection()
        if not conn:
            flash('Database connection failed')
            return redirect(request.url)
        
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Validate JD exists and get its details
            cur.execute("""
                SELECT jd_id, title, jd_file, file_id
                FROM job_descriptions
                WHERE jd_id = %s
            """, (jd_id,))
            
            jd_info = cur.fetchone()
            if not jd_info:
                flash('Selected job description not found')
                return redirect(request.url)
            
            # Validate criteria if provided
            criteria_info = None
            if criteria_id:
                cur.execute("""
                    SELECT criteria_id, criteria_name, parameter, weightage
                    FROM criteria
                    WHERE criteria_id = %s
                """, (criteria_id,))
                
                criteria_info = cur.fetchone()
                if not criteria_info:
                    flash('Selected criteria not found')
                    return redirect(request.url)
            
            cur.close()
            conn.close()
        except Exception as e:
            flash(f'Error validating data: {str(e)}')
            return redirect(request.url)
        
        # Process each uploaded file with deduplication
        uploaded_files = []
        failed_files = []
        processed_filenames = set()  # Track processed filenames to prevent duplicates
        
        # First pass: collect unique files only
        unique_files = []
        for file in files:
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                if filename not in processed_filenames:
                    processed_filenames.add(filename)
                    unique_files.append(file)
                else:
                    print(f"⚠️ Skipping duplicate file: {filename}")
            else:
                failed_files.append({
                    'filename': file.filename,
                    'error': 'Invalid file type or empty filename'
                })
        
        print(f"📊 Total files received: {len(files)}")
        print(f"📊 Unique files to upload: {len(unique_files)}")
        print(f"📊 Duplicate files skipped: {len(files) - len(unique_files)}")
        
        # Second pass: upload only unique files
        for file in unique_files:
            try:
                filename = secure_filename(file.filename)
                
                # Upload to Supabase storage
                print(f"Uploading resume: {filename}")
                
                # Read file content for token counting
                file_content = file.read().decode('utf-8', errors='replace')
                token_count = count_tokens(file_content)
                
                # Reset file pointer after reading
                file.seek(0)
                
                supabase_url, file_path = upload_to_supabase_storage(file, filename)
                
                if supabase_url:
                    uploaded_files.append({
                        'filename': filename,
                        'url': supabase_url,
                        'path': file_path,
                        'token_count': token_count
                    })
                    print(f"Successfully uploaded to Supabase: {filename} (Tokens: {token_count})")
                else:
                    uploaded_files.append({
                        'filename': filename,
                        'url': None,
                        'path': file_path,
                        'token_count': token_count
                    })
                    print(f"Saved to local storage: {filename} (Tokens: {token_count})")
                
            except Exception as e:
                print(f"Failed to upload {file.filename}: {e}")
                failed_files.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        if not uploaded_files:
            flash('No files were successfully uploaded')
            return redirect(request.url)
        
        # Check for existing analyses to avoid duplicates
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get existing resume filenames for this JD
            cur.execute("""
                SELECT candidate_name as resume_filename 
                FROM assessment_reports 
                WHERE resolved_jd_id = %s
            """, (jd_id,))
            
            existing_filenames = {row['resume_filename'] for row in cur.fetchall()}
            cur.close()
            conn.close()
            
            print(f"📊 Existing resumes for JD {jd_id}: {existing_filenames}")
            
            # Filter out files that already exist
            new_files = []
            skipped_files = []
            
            for file in unique_files:
                filename = secure_filename(file.filename)
                if filename in existing_filenames:
                    print(f"⚠️ Skipping existing file: {filename}")
                    skipped_files.append(filename)
                else:
                    new_files.append(file)
            
            if not new_files:
                flash('All uploaded files already exist for this job description')
                return redirect(request.url)
            
            print(f"📊 New files to process: {len(new_files)}")
            print(f"📊 Skipped existing files: {len(skipped_files)}")
            
            # Update unique_files to only include new files
            unique_files = new_files
            
        except Exception as e:
            print(f"❌ Warning: Could not check existing analyses: {e}")
            # Continue with all files if check fails
        
        # Call CV analyzer with all uploaded files
        try:
            # Prepare payload for CV analyzer - prioritize Supabase URLs, fallback to local files
            resume_urls = []
            resume_files = []
            
            for file in uploaded_files:
                if file['url']:
                    # File was uploaded to Supabase successfully
                    resume_urls.append(file['url'])
                else:
                    # File was saved locally (fallback)
                    resume_files.append(file['path'])
            
            payload = {
                'jd_id': jd_id,
                'criteria_id': criteria_id,  # Include selected criteria
                'resume_urls': resume_urls,
                'resume_files': resume_files
            }
            
            print(f"📊 Payload to CV analyzer:")
            print(f"📊 Resume URLs: {resume_urls}")
            print(f"📊 Resume Files: {resume_files}")
            
            # If criteria is selected, we'll do additional scoring after CV analysis
            scoring_results = []
            if criteria_id:
                logger.info(f"Criteria selected: {criteria_id}, will perform additional scoring")
            
            print(f"Sending analysis request to: {CV_ANALYZER_URL}")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(CV_ANALYZER_URL, json=payload, timeout=120)
            
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Check how many records were actually created
                conn = get_db_connection()
                cur = conn.cursor()
                
                # Get the count of records that were just created (new records only)
                # We can estimate this by counting records created in the last few minutes
                cur.execute("""
                    SELECT COUNT(*) FROM assessment_reports 
                    WHERE resolved_jd_id = %s 
                    AND created_at >= NOW() - INTERVAL '5 minutes'
                """, (jd_id,))
                records_created = cur.fetchone()[0]
                
                # Alternative: count records that match the uploaded filenames
                if records_created == 0 and uploaded_files:
                    filename_list = [file['filename'] for file in uploaded_files]
                    placeholders = ','.join(['%s'] * len(filename_list))
                    cur.execute(f"""
                        SELECT COUNT(*) FROM assessment_reports 
                        WHERE resolved_jd_id = %s 
                        AND candidate_name IN ({placeholders})
                    """, (jd_id, *filename_list))
                    records_created = cur.fetchone()[0]
                
                cur.close()
                conn.close()
                
                print(f"📊 Records created for JD {jd_id}: {records_created}")
                print(f"📊 Files uploaded: {len(uploaded_files)}")
                print(f"📊 CV Analyzer response: {result}")
                print(f"📊 Results array length: {len(result.get('results', []))}")
                print(f"📊 Result type: {type(result)}")
                print(f"📊 Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                
                # Scoring is now handled by cv_analyzer.py
                scoring_results = []
                if criteria_id:
                    logger.info(f"Scoring will be performed by cv_analyzer service")
                    # Extract scoring results from the analysis response
                    for analysis_result in result.get('results', []):
                        if 'scoring' in analysis_result and analysis_result['scoring']:
                            scoring_results.append({
                                'filename': analysis_result['filename'],
                                'score_id': analysis_result['scoring'].get('score_id'),
                                'final_score': analysis_result['scoring'].get('scoring_result', {}).get('final_score', 0),
                                'recommendation': analysis_result['scoring'].get('scoring_result', {}).get('recommendation', 'Review further')
                            })
                
                logger.info(f"Final scoring results: {scoring_results}")
                flash(f'Successfully processed {len(uploaded_files)} new resumes!')
                
                print(f"📊 Final result variable before response creation:")
                print(f"📊 Result: {result}")
                print(f"📊 Result type: {type(result)}")
                print(f"📊 Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                print(f"📊 Result.get('results'): {result.get('results', [])}")
                
                # Prepare token info for each uploaded resume
                resume_token_info = [
                    {
                        'filename': file_info['filename'],
                        'token_count': file_info['token_count'],
                        'is_over_limit': file_info['token_count'] > app.config['MAX_TOKENS'],
                        'is_near_limit': file_info['token_count'] > (app.config['MAX_TOKENS'] * 0.85)
                    }
                    for file_info in uploaded_files
                ]
                
                response_data = {
                    'status': 'success',
                    'message': f'Successfully processed {len(uploaded_files)} new resumes',
                    'jd_id': jd_id,
                    'criteria_id': criteria_id,
                    'uploaded_count': len(uploaded_files),
                    'failed_count': len(failed_files),
                    'analysis_result': {'results': result.get('results', [])},
                    'records_created': records_created,
                    'scoring_results': scoring_results,
                    'resume_token_info': resume_token_info
                }
                
                print(f"📊 Response data being sent to frontend:")
                print(f"📊 Analysis result: {response_data['analysis_result']}")
                print(f"📊 Analysis result type: {type(response_data['analysis_result'])}")
                print(f"📊 Results array: {response_data['analysis_result']['results']}")
                print(f"📊 Results array type: {type(response_data['analysis_result']['results'])}")
                print(f"📊 Results array length: {len(response_data['analysis_result']['results'])}")
                
                return jsonify(response_data)
            else:
                error_msg = f'Analysis failed: {response.text}'
                print(f"Analysis error: {error_msg}")
                flash(error_msg)
                return jsonify({
                    'status': 'partial_success',
                    'message': f'Files uploaded but analysis failed',
                    'jd_id': jd_id,
                    'criteria_id': criteria_id,
                    'uploaded_count': len(uploaded_files),
                    'failed_count': len(failed_files),
                    'error_details': response.text
                })
        
        except requests.exceptions.RequestException as e:
            error_msg = f'Backend connection failed: {str(e)}'
            print(f"Connection error: {error_msg}")
            flash(error_msg)
            return jsonify({
                'status': 'partial_success',
                'message': f'Files uploaded but analysis failed',
                'jd_id': jd_id,
                'criteria_id': criteria_id,
                'uploaded_count': len(uploaded_files),
                'failed_count': len(failed_files),
                'error_details': str(e)
            })
    
    except Exception as e:
        flash(f'Upload error: {str(e)}')
        return redirect(request.url)

@app.route('/job_descriptions')
def list_job_descriptions():
    """List all job descriptions"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection failed')
        return render_template('list_resumes.html', job_descriptions=[])
    
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT jd_id, title, company_id, user_id, created_at, updated_at, file_id, jd_file
            FROM job_descriptions
            ORDER BY created_at DESC
            LIMIT 100
        """)
        job_descriptions = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return render_template('list_resumes.html', job_descriptions=job_descriptions)
    except Exception as e:
        flash(f'Error fetching job descriptions: {str(e)}')
        return render_template('list_resumes.html', job_descriptions=[])

@app.route('/api/job_descriptions')
def api_list_job_descriptions():
    """API endpoint to list all job descriptions as JSON"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT jd_id, title, company_id, user_id, created_at, updated_at, file_id, jd_file
            FROM job_descriptions
            ORDER BY created_at DESC
            LIMIT 100
        """)
        job_descriptions = cur.fetchall()
        
        # Convert to list of dictionaries for JSON serialization
        result = []
        for jd in job_descriptions:
            jd_dict = dict(jd)
            # Convert UUID to string for JSON serialization
            jd_dict['jd_id'] = str(jd_dict['jd_id'])
            result.append(jd_dict)
        
        cur.close()
        conn.close()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error fetching job descriptions: {str(e)}'}), 500

@app.route('/job_description/<uuid:jd_id>')
def view_job_description(jd_id):
    """View specific job description and its analysis"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection failed')
        return redirect(url_for('list_job_descriptions'))
    
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""SELECT * FROM job_descriptions WHERE jd_id = %s""", (jd_id,))
        jd = cur.fetchone()
        
        if not jd:
            flash('Job description not found')
            return redirect(url_for('list_job_descriptions'))
        
        cur.execute("""
            SELECT * FROM resolved_jd
            WHERE referenced_jd = %s OR referenced_jd = %s OR referenced_jd = %s
            ORDER BY created_at DESC
        """, (jd['file_id'], jd['jd_file'], str(jd['jd_id'])))
        
        resolved_data = cur.fetchall()
        
        # Calculate token information for the job description
        jd_text = jd.get('text_content', '') or jd.get('description', '')
        token_count = count_tokens(jd_text)
        max_tokens = app.config.get('MAX_TOKENS', 4000)
        token_percentage = min(round((token_count / max_tokens) * 100), 100)  # Cap at 100%
        
        token_info = {
            'count': token_count,
            'max': max_tokens,
            'percentage': token_percentage,
            'is_over_limit': token_count > max_tokens,
            'is_near_limit': token_count > (max_tokens * 0.85)  # 85% of max
        }
        
        cur.close()
        conn.close()
        
        return render_template('view_resumes.html', 
                            job_description=jd, 
                            resolved_data=resolved_data,
                            token_info=token_info)
    except Exception as e:
        flash(f'Error fetching job description: {str(e)}')
        return redirect(url_for('list_job_descriptions'))

@app.route('/job_description/<uuid:jd_id>/resumes')
def view_job_description_resumes(jd_id):
    """View job description analysis results page"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection failed')
        return redirect(url_for('list_job_descriptions'))
    
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""SELECT * FROM job_descriptions WHERE jd_id = %s""", (jd_id,))
        jd = cur.fetchone()
        
        if not jd:
            flash('Job description not found')
            return redirect(url_for('list_job_descriptions'))
        
        # Get resume analysis results
        cur.execute("""
            SELECT id as analysis_id, candidate_name as resume_filename, resume_url, scoring as analysis_data, status, created_at
            FROM assessment_reports
            WHERE resolved_jd_id = %s
            ORDER BY created_at DESC
        """, (jd_id,))
        
        resume_analyses = cur.fetchall()
        
        # Get scoring results if available
        cur.execute("""
            SELECT id as score_id, candidate_name as resume_filename, overall_score as final_score, recommendation, created_at
            FROM assessment_reports
            WHERE resolved_jd_id = %s AND overall_score IS NOT NULL
            ORDER BY overall_score DESC, created_at DESC
        """, (jd_id,))
        
        scoring_results = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return render_template('resume_analysis_results.html', 
                             job_description=jd, 
                             resume_analyses=resume_analyses,
                             scoring_results=scoring_results)
    except Exception as e:
        flash(f'Error fetching job description: {str(e)}')
        return redirect(url_for('list_job_descriptions'))

@app.route('/scoring/<uuid:jd_id>')
def view_scoring_details(jd_id):
    """View scoring details for a specific job description"""
    conn = get_db_connection()
    if not conn:
        flash('Database connection failed')
        return redirect(url_for('list_job_descriptions'))

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get job description info
        cur.execute("""
            SELECT jd_id, title, company_id, created_at, jd_file, file_id
            FROM job_descriptions 
            WHERE jd_id = %s
        """, (jd_id,))
        
        jd = cur.fetchone()
        if not jd:
            flash('Job description not found')
            return redirect(url_for('list_job_descriptions'))
        
        # Get all scoring results for this JD
        cur.execute("""
            SELECT 
                ar.id as score_id, ar.id as analysis_id, ar.candidate_name as resume_filename, 
                ar.overall_score as final_score, ar.recommendation, ar.scoring->>'consideration' as consideration,
                ar.scoring as parameter_scores, ar.created_at, ar.token_count,
                ar.cumulative_token_count, ar.upload_count,
                c.criteria_name,
                ar.scoring as analysis_data
            FROM assessment_reports ar
            LEFT JOIN criteria c ON ar.criteria_id = c.criteria_id
            WHERE ar.resolved_jd_id = %s AND ar.overall_score IS NOT NULL
            ORDER BY ar.overall_score DESC, ar.created_at DESC
        """, (jd_id,))
        
        scoring_results = cur.fetchall()
        
        # Get statistics
        cur.execute("""
            SELECT 
                COUNT(*) as total_resumes,
                AVG(overall_score) as avg_score,
                MAX(overall_score) as max_score,
                MIN(overall_score) as min_score,
                COUNT(CASE WHEN recommendation = 'To be interviewed' THEN 1 END) as to_interview,
                COUNT(CASE WHEN recommendation = 'Candidature rejected' THEN 1 END) as rejected,
                COUNT(CASE WHEN recommendation = 'Review further' THEN 1 END) as review_further
            FROM assessment_reports
            WHERE resolved_jd_id = %s AND overall_score IS NOT NULL
        """, (jd_id,))
        
        stats = cur.fetchone()
        
        cur.close()
        conn.close()
        
        return render_template('scoring_details.html', 
                             job_description=jd, 
                             scoring_results=scoring_results,
                             stats=stats)
                             
    except Exception as e:
        logger.error(f'Error fetching scoring details: {str(e)}')
        flash(f'Error fetching scoring details: {str(e)}')
        return redirect(url_for('list_job_descriptions'))

@app.route('/scoring_details/<uuid:score_id>')
def get_scoring_details(score_id):
    """Get detailed scoring information for a specific score ID"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'success': False, 'error': 'Database connection failed'})

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get the scoring result
        cur.execute("""
            SELECT 
                ar.id as score_id, ar.id as analysis_id, ar.candidate_name as resume_filename, 
                ar.overall_score as final_score, ar.recommendation, ar.scoring->>'consideration' as consideration,
                ar.scoring as parameter_scores, ar.created_at, ar.token_count,
                ar.cumulative_token_count, ar.upload_count,
                c.criteria_name, c.weightage
            FROM assessment_reports ar
            LEFT JOIN criteria c ON ar.criteria_id = c.criteria_id
            WHERE ar.id = %s
        """, (str(score_id),))
        
        score_result = cur.fetchone()
        if not score_result:
            return jsonify({'success': False, 'error': 'Scoring result not found'})
        
        # Parse parameter scores
        parameter_scores = {}
        if score_result['parameter_scores']:
            try:
                # If it's already a dict, use it directly
                if isinstance(score_result['parameter_scores'], dict):
                    parameter_scores = score_result['parameter_scores']
                else:
                    parameter_scores = json.loads(score_result['parameter_scores'])
                logger.info(f"Parsed parameter_scores: {parameter_scores}")
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error parsing parameter_scores: {e}")
                parameter_scores = {}
        
        # Prepare detailed scoring breakdown
        details = []
        for criteria_name, score_info in parameter_scores.items():
            details.append({
                'criteria_name': criteria_name,
                'score': score_info.get('score'),
                'rating': score_info.get('rating'),
                'weightage': score_info.get('weightage')
            })
        # Prepare response
        response_data = {
            'success': True,
            'score': {
                'final_score': score_result['final_score'],
                'recommendation': score_result['recommendation'],
                'consideration': score_result['consideration'],
                'created_at': score_result['created_at'].isoformat() if score_result['created_at'] else None,
                'token_count': score_result.get('token_count'),
                'cumulative_token_count': score_result.get('cumulative_token_count'),
                'upload_count': score_result.get('upload_count')
            },
            'details': details
        }
        
        cur.close()
        conn.close()
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f'Error fetching scoring details: {str(e)}')
        return jsonify({'success': False, 'error': f'Error fetching scoring details: {str(e)}'})

@app.route('/test_modal')
def test_modal():
    """Test modal functionality"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Modal Test</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>Modal Test</h1>
            <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#testModal">
                Test Modal
            </button>
            
            <!-- Modal -->
            <div class="modal fade" id="testModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Test Scoring Modal</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <p>Test content for modal</p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("Starting Resume Upload Application on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000) 