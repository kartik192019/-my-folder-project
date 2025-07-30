# Resume Screener - AI-Powered Job Description Analyzer

This application analyzes job descriptions from PDF files and other formats using AI to extract key information like requirements, skills, and qualifications.

## Features

- **PDF Analysis**: Upload and analyze PDF job descriptions
- **AI-Powered Extraction**: Uses Gemini AI or OpenAI to extract structured information
- **Multiple File Formats**: Supports PDF, DOCX, DOC, and TXT files
- **Real Analysis**: No mock responses - actual AI analysis of uploaded content
- **Structured Output**: Extracts personal info, technical skills, experience requirements, and more

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

### Option 1: Run Both Services (Recommended)

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

### Option 2: Test the AI Analyzer

Run the test script to verify the AI analyzer is working:
```bash
python test_ai_analyzer.py
```

## Usage

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

## File Format Support

- **PDF**: Text extraction using PyPDF2
- **DOCX/DOC**: Text extraction using python-docx
- **TXT**: Direct text processing with encoding detection

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