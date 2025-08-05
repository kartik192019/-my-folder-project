# Backend Setup Guide for React Integration

This guide explains how to set up all four Python backend services to work with the React frontend.

## Backend Services Overview

Your application consists of **4 separate Python Flask services**:

1. **`app.py`** (Port 5004) - Job Description Upload Service
2. **`app2.py`** (Port 5000) - Resume Upload Service  
3. **`ai_analyzer.py`** (Port 5001) - AI Analysis Service
4. **`cv_analyzer.py`** (Port 5002) - CV Analysis & Scoring Service

## Prerequisites

- Python 3.7+
- PostgreSQL database
- Supabase account (for file storage)
- AI API keys (OpenAI or Gemini)

## Step 1: Install Dependencies

```bash
pip install flask flask-cors psycopg2-binary supabase tiktoken
pip install PyPDF2 python-docx chardet aspose-words
pip install google-generativeai openai requests
```

## Step 2: Configure CORS for All Services

### For `app.py` (Job Description Upload Service)

Add these lines at the top of `app.py`:

```python
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS  # Add this line
import psycopg2
# ... rest of your imports

app = Flask(__name__)
CORS(app)  # Add this line
app.secret_key = 'your-secret-key-here'
```

### For `app2.py` (Resume Upload Service)

Add these lines at the top of `app2.py`:

```python
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS  # Add this line
import psycopg2
# ... rest of your imports

app = Flask(__name__)
CORS(app)  # Add this line
app.secret_key = 'your-secret-key-here'
```

### For `ai_analyzer.py` (AI Analysis Service)

Add these lines at the top of `ai_analyzer.py`:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this line
import psycopg2
# ... rest of your imports

app = Flask(__name__)
CORS(app)  # Add this line
```

### For `cv_analyzer.py` (CV Analysis Service)

Add these lines at the top of `cv_analyzer.py`:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this line
import psycopg2
# ... rest of your imports

app = Flask(__name__)
CORS(app)  # Add this line
```

## Step 3: Start All Backend Services

Open **4 separate terminal windows** and run each service:

### Terminal 1 - Job Description Upload Service
```bash
cd /path/to/your/project
python app.py
# Service will start on http://localhost:5004
```

### Terminal 2 - Resume Upload Service
```bash
cd /path/to/your/project
python app2.py
# Service will start on http://localhost:5000
```

### Terminal 3 - AI Analysis Service
```bash
cd /path/to/your/project
python ai_analyzer.py
# Service will start on http://localhost:5001
```

### Terminal 4 - CV Analysis Service
```bash
cd /path/to/your/project
python cv_analyzer.py
# Service will start on http://localhost:5002
```

## Step 4: Verify Services Are Running

You can verify each service is running by visiting:

- **Job Description Upload**: http://localhost:5004
- **Resume Upload**: http://localhost:5000
- **AI Analysis**: http://localhost:5001/health
- **CV Analysis**: http://localhost:5002/health

## Step 5: Start React Frontend

```bash
cd react-example
npm install
npm start
```

The React app will start on http://localhost:3000

## Service Communication Flow

```
React Frontend (Port 3000)
    ↓
├── app.py (Port 5004) - Job Description Upload
├── app2.py (Port 5000) - Resume Upload  
├── ai_analyzer.py (Port 5001) - AI Analysis
└── cv_analyzer.py (Port 5002) - CV Analysis & Scoring
```

## Troubleshooting Common Issues

### 1. CORS Errors
**Problem**: Browser shows CORS errors when React tries to call Python backends.

**Solution**: Ensure `flask-cors` is installed and `CORS(app)` is added to all Flask applications.

```bash
pip install flask-cors
```

### 2. Port Already in Use
**Problem**: "Address already in use" error when starting services.

**Solution**: Kill existing processes or use different ports:

```bash
# Find processes using the port
lsof -i :5000
lsof -i :5001
lsof -i :5002
lsof -i :5004

# Kill the process
kill -9 <PID>
```

### 3. Database Connection Errors
**Problem**: Services can't connect to PostgreSQL.

**Solution**: 
- Verify PostgreSQL is running
- Check database credentials in `config.py`
- Ensure database tables exist

### 4. AI API Key Issues
**Problem**: AI analysis fails due to missing or invalid API keys.

**Solution**:
- Set environment variables: `OPENAI_API_KEY` or `GEMINI_API_KEY`
- Or configure keys in `config.py`
- Test API keys manually

### 5. File Upload Issues
**Problem**: Resume or JD uploads fail.

**Solution**:
- Check Supabase configuration in `config.py`
- Verify file size limits (16MB max)
- Ensure supported file formats: PDF, DOC, DOCX, TXT

## Service-Specific Configuration

### app.py (Job Description Upload)
- **Port**: 5004
- **Purpose**: Upload and manage job descriptions
- **Key Endpoints**: `/upload`, `/job_descriptions`, `/job_description/<id>`

### app2.py (Resume Upload)
- **Port**: 5000
- **Purpose**: Upload and process resumes
- **Key Endpoints**: `/upload`, `/job_description/<id>/resumes`, `/scoring/<id>`

### ai_analyzer.py (AI Analysis)
- **Port**: 5001
- **Purpose**: AI-powered analysis and scoring
- **Key Endpoints**: `/analyze`, `/score_resume`, `/health`

### cv_analyzer.py (CV Analysis)
- **Port**: 5002
- **Purpose**: CV analysis and criteria management
- **Key Endpoints**: `/analyze_resumes`, `/calculate_resume_score`, `/criteria`

## Testing the Integration

1. **Start all 4 Python services**
2. **Start React frontend**
3. **Visit http://localhost:3000**
4. **Check Service Health tab** to verify all services are online
5. **Test each functionality**:
   - Upload a job description
   - Upload resumes
   - Create scoring criteria
   - Score resumes

## Production Deployment

For production deployment:

1. **Use a process manager** like PM2 or Supervisor
2. **Set up reverse proxy** (Nginx) to route requests
3. **Configure environment variables** for API keys
4. **Set up SSL certificates** for HTTPS
5. **Use production database** (not local PostgreSQL)
6. **Configure proper CORS origins** for your domain

## Monitoring

Monitor your services using:

- **Health check endpoints**: `/health` on each service
- **Logs**: Check terminal output for each service
- **React dashboard**: Service Health tab shows real-time status
- **Database**: Monitor PostgreSQL connections and queries

## Support

If you encounter issues:

1. Check the **Service Health** tab in React dashboard
2. Review terminal logs for each Python service
3. Verify database connections
4. Test API endpoints directly using curl or Postman
5. Check browser console for CORS or network errors 