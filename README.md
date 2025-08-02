# Resume Screener - AI-Powered Job Description & Resume Analyzer

This application analyzes job descriptions and resumes using AI to extract key information like requirements, skills, and qualifications. It supports both job description analysis and resume screening against job requirements.

## Features

### Job Description Analysis
- **PDF Analysis**: Upload and analyze PDF job descriptions
- **AI-Powered Extraction**: Uses Gemini AI or OpenAI to extract structured information
- **Multiple File Formats**: Supports PDF, DOCX, DOC, and TXT files
- **Real Analysis**: No mock responses - actual AI analysis of uploaded content
- **Structured Output**: Extracts personal info, technical skills, experience requirements, and more

### Resume Screening
- **Multiple Resume Upload**: Upload up to 15 resumes at once
- **JD Selection**: Choose from existing job descriptions via dropdown
- **Criteria-Based Scoring**: Select scoring criteria for detailed evaluation
- **AI-Powered Analysis**: Each resume is analyzed against job requirements
- **Comprehensive Results**: Extract personal information, educational qualifications, job history, and skills
- **Modern UI**: Beautiful interface with progress tracking and detailed results

### Criteria Management
- **Create Criteria**: Define custom scoring criteria with parameters and weightage
- **Criteria Grid**: Set up detailed evaluation grids for different job types
- **Bulk Operations**: Create multiple criteria at once
- **Validation**: Built-in validation for criteria data
- **Statistics**: Get insights about criteria usage and performance

### Scoring System
- **Criteria-Based Scoring**: Score resumes against specific criteria
- **Parameter Scoring**: Individual parameter evaluation with weightage
- **Final Recommendations**: Get interview recommendations based on scores
- **Score Statistics**: View scoring statistics and trends
- **Detailed Assessment**: Comprehensive evaluation reports

## Prerequisites

- Python 3.8 or higher
- Valid AI API key (Gemini or OpenAI)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd resume_screener
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   Edit `config.py` and set your API keys:
   ```python
   GEMINI_API_KEY = 'your-gemini-api-key-here'
   OPENAI_API_KEY = 'your-openai-api-key-here'  # Optional fallback
   ```

## Running the Application

### Option 1: Job Description Analysis Only

1. **Start the AI Analyzer Backend** (Terminal 1):
   ```bash
   python ai_analyzer.py
   ```
   This starts the AI analysis service on http://localhost:5001

2. **Start the Web Application** (Terminal 2):
   ```bash
   python app.py
   ```
   This starts the web interface on http://localhost:5000

### Option 2: Resume Screening System (Recommended)

1. **Start the CV Analyzer Backend** (Terminal 1):
   ```bash
   python cv_analyzer.py
   ```
   This starts the resume analysis service on http://localhost:5002
   - Includes criteria management endpoints
   - Includes scoring system endpoints
   - Handles resume analysis and scoring

2. **Start the Resume Upload App** (Terminal 2):
   ```bash
   python app2.py
   ```
   This starts the resume upload interface on http://localhost:5003

### Option 3: Full System (All Services)

Run all services for complete functionality:
```bash
# Terminal 1: AI Analyzer (for job descriptions)
python ai_analyzer.py

# Terminal 2: Main App (for job descriptions)
python app.py

# Terminal 3: CV Analyzer (for resumes, criteria, and scoring)
python cv_analyzer.py

# Terminal 4: Resume App (for resumes)
python app2.py
```

### Option 4: Test the System

Run the test script to verify the AI analyzer is working:
```bash
python test_ai_analyzer.py
```

## API Endpoints

### CV Analyzer (Port 5002) - Integrated Services

#### Resume Analysis
- `POST /analyze_resumes` - Analyze multiple resumes
- `POST /calculate_resume_score` - Calculate resume scores
- `GET /scoring_details/<score_id>` - Get scoring details

#### Criteria Management
- `POST /criteria` - Create new criteria
- `GET /criteria/<criteria_id>` - Get specific criteria
- `PUT /criteria/<criteria_id>` - Update criteria
- `DELETE /criteria/<criteria_id>` - Delete criteria
- `GET /criteria` - List all criteria
- `GET /criteria/stats` - Get criteria statistics
- `POST /criteria/bulk` - Create multiple criteria
- `POST /criteria/validate` - Validate criteria data

#### Scoring System
- `GET /scoring/scores` - Get resume scores
- `GET /scoring/stats` - Get scoring statistics

#### Health & Info
- `GET /health` - Health check
- `GET /` - Service information and endpoints

## Usage

### Job Description Analysis

1. **Open your browser** and go to http://localhost:5000

2. **Upload a job description**:
   - Choose a PDF, DOCX, DOC, or TXT file
   - Or paste the job description text directly
   - Fill in optional company and user information

3. **View the analysis results**:
   - The system will extract text from your file
   - Send it to the AI analyzer for processing
   - Display structured results including:
     - Personal requirements (location, age, gender)
     - Job history and experience requirements
     - Technical and functional skills
     - Educational qualifications
     - Soft skills

### Resume Screening

1. **Open your browser** and go to http://localhost:5003

2. **Select a job description**:
   - Choose from the dropdown of existing job descriptions
   - The system will use the analyzed requirements for comparison

3. **Upload multiple resumes**:
   - Select up to 15 resume files (PDF, DOCX, DOC, TXT)
   - Click "Upload and Analyze Resumes"

4. **View screening results**:
   - Match percentage for each resume
   - Personal information extraction (name, email, phone, city, birthdate, age, gender)
- Educational qualifications analysis
- Job history extraction
- Skills categorization (technical, functional, soft skills)
   - Skills matching and missing skills
   - Overall recommendation (Strongly Recommend/Recommend/Consider/Not Recommended)

## API Endpoints

### AI Analyzer Backend (Port 5001)

- `GET /health` - Health check
- `POST /analyze` - Analyze job description
- `POST /test` - Test analysis with sample text

### Web Application (Port 5000)

- `GET /` - Upload form
- `POST /upload` - Upload and analyze job description
- `GET /job_descriptions` - List all job descriptions
- `GET /job_description/<id>` - View specific job description

### CV Analyzer Backend (Port 5002)

- `GET /health` - Health check
- `POST /analyze_resumes` - Analyze multiple resumes against job description
- `GET /` - Home endpoint

### Resume Upload App (Port 5003)

- `GET /` - Resume upload form
- `POST /upload_resumes` - Upload and analyze resumes
- `GET /job_descriptions` - List job descriptions for selection

## Troubleshooting

### Common Issues

1. **"No valid AI API keys found"**
   - Check your API keys in `config.py`
   - Ensure the keys are valid and have sufficient credits

2. **"Could not extract text from PDF"**
   - The PDF might be image-based or corrupted
   - Try converting to text format or copying content manually

3. **"AI analysis failed"**
   - Check the server logs for detailed error messages
   - Verify your internet connection
   - Ensure API keys have sufficient quota

### Testing

Run the test script to verify everything is working:
```bash
python test_ai_analyzer.py
```

This will test:
- Health endpoint connectivity
- AI provider setup
- Job description analysis
- Response format validation

## Configuration

### Environment Variables

You can set these environment variables instead of editing `config.py`:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export DB_HOST="your-database-host"
export DB_PASSWORD="your-database-password"
```

### Database Configuration

The application uses PostgreSQL. Update the database configuration in `config.py`:

```python
DB_HOST = 'your-database-host'
DB_PORT = '5432'
DB_NAME = 'your-database-name'
DB_USER = 'your-database-user'
DB_PASSWORD = 'your-database-password'
```

### Database Tables

The system uses these tables:

- `job_descriptions` - Stores uploaded job descriptions
- `resolved_jd` - Stores analyzed job requirements
- `resume_analyses` - Stores resume analysis results

**Note**: The `resume_analyses` table should be created in your Supabase database using the SQL editor with this schema:

```sql
CREATE TABLE IF NOT EXISTS resume_analyses (
  analysis_id uuid not null default gen_random_uuid (),
  jd_id uuid not null,
  resume_filename text not null,
  resume_url text null,
  analysis_data jsonb not null,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  status character varying(50) null default 'active'::character varying,
  constraint resume_analyses_pkey primary key (analysis_id)
);
```

## File Format Support

- **PDF**: Text extraction using PyPDF2
- **DOCX/DOC**: Text extraction using python-docx
- **TXT**: Direct text processing with encoding detection

## System Architecture

```
resume_screener/
├── app.py                 # Job description upload frontend (Port 5000)
├── ai_analyzer.py         # Job description analysis backend (Port 5001)
├── app2.py               # Resume upload frontend (Port 5003)
├── cv_analyzer.py        # Resume analysis backend (Port 5002)
├── config.py             # Configuration settings

├── templates/
│   ├── upload.html       # Job description upload UI
│   ├── list.html         # Job description list
│   ├── view.html         # Job description view
│   ├── upload_resumes.html    # Resume upload UI
│   ├── list_resumes.html      # JD list for resumes
│   └── view_resumes.html      # JD view for resumes
└── uploads/              # Local file storage
```

## AI Providers

The system supports two AI providers:

1. **Gemini AI** (Primary): Google's Gemini model for analysis
2. **OpenAI** (Fallback): GPT-3.5-turbo as backup option

The system automatically tests API keys and falls back to the alternative provider if needed.

## Security Notes

- API keys are stored in configuration files
- Consider using environment variables for production
- File uploads are validated for allowed extensions
- Maximum file size is 16MB

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. 