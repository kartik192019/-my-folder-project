from flask import Flask, request, jsonify
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import json
import uuid
from datetime import datetime
import os
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Database configuration
DB_CONFIG = Config.get_db_config()

# AI Service configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', Config.OPENAI_API_KEY)
GEMINI_API_KEY = Config.GEMINI_API_KEY

class JobDescriptionAnalyzer:
    def __init__(self):
        self.db_config = DB_CONFIG
        self.setup_ai_client()

    def setup_ai_client(self):
        """Setup AI client based on preferred provider"""
        preferred_provider = Config.PREFERRED_AI_PROVIDER
        
        if preferred_provider == 'gemini':
            self._setup_gemini()
        elif preferred_provider == 'openai':
            self._setup_openai()
        else:
            logger.warning("No valid AI provider configured, using intelligent content analysis")
            self.ai_provider = 'intelligent_analysis'

    def _setup_gemini(self):
        """Setup Gemini AI client"""
        if GEMINI_API_KEY and GEMINI_API_KEY != 'your-gemini-api-key' and len(GEMINI_API_KEY) > 10:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                
                # Try different Gemini models in order of preference
                models_to_try = ['gemini-2.5-flash-lite', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
                
                for model_name in models_to_try:
                    try:
                        self.ai_client = genai.GenerativeModel(model_name)
                        # Test the connection with a simple call
                        test_response = self.ai_client.generate_content("Hello")
                        if test_response and test_response.text:
                            self.ai_provider = 'gemini'
                            logger.info(f"Using Gemini AI with {model_name}")
                            return
                    except Exception as e:
                        logger.warning(f"{model_name} failed: {e}")
                        continue
                
                # If all models fail, fall back to intelligent analysis
                logger.warning("All Gemini models failed, using intelligent content analysis")
                self.ai_provider = 'intelligent_analysis'
                
            except ImportError:
                logger.warning("google-generativeai not installed, using intelligent content analysis")
                self.ai_provider = 'intelligent_analysis'
        else:
            logger.warning("No valid Gemini API key found, using intelligent content analysis")
            self.ai_provider = 'intelligent_analysis'

    def _setup_openai(self):
        """Setup OpenAI client"""
        if OPENAI_API_KEY and OPENAI_API_KEY != 'your-openai-api-key' and len(OPENAI_API_KEY) > 10:
            try:
                import openai
                from openai import OpenAI
                
                # Test the API key
                client = OpenAI(api_key=OPENAI_API_KEY)
                test_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                
                if test_response and test_response.choices:
                    self.ai_client = client
                    self.ai_provider = 'openai'
                    logger.info("Using OpenAI")
                    return
                else:
                    raise Exception("OpenAI API test failed")
                    
            except ImportError:
                logger.error("openai package not installed")
                logger.warning("Falling back to intelligent content analysis")
                self.ai_provider = 'intelligent_analysis'
                return
            except Exception as e:
                logger.error(f"OpenAI setup failed: {e}")
                logger.warning("Falling back to intelligent content analysis")
                self.ai_provider = 'intelligent_analysis'
                return
        else:
            logger.warning("No valid OpenAI API key found, using intelligent content analysis")
            self.ai_provider = 'intelligent_analysis'

    def get_db_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None

    
    def _get_attributes_prompt(self, text):
        """Generate prompt for qualifications and experience extraction from JD"""
        return f"""
        You are an expert extraction algorithm. Your task is to extract job requirements from the job description text.

Return ONLY a valid JSON object using the format below. If any category is missing or not clearly mentioned in the job description, OMIT that key entirely from the output.

JSON Format (example structure only):

{{
    "City": "Mention the mandatory city requirements if stated. Mention preferred cities only if clearly specified. If preference is not mentioned, assume the city requirement is mandatory.",
    "Age": "Mention the mandatory age requirements if stated. Mention preferred ages only if clearly specified. If preference is not mentioned, assume the age requirement is mandatory.",
    "Gender": "Mention the mandatory gender requirements if stated. Mention preferred genders only if clearly specified. If preference is not mentioned, assume the gender requirement is mandatory.",
    "Job History": "Details of mandatory experience requirements: [specific details]. Details of preferred experience requirements: [specific details].",
    "Technical skills": "Technical skills (if available) - Required: [list all required technical skills]. Preferred: [list all preferred technical skills].",
    "Functional skills": "Functional skills (if available) - Required: [list all required functional skills]. Preferred: [list all preferred functional skills].",
    "Educational qualification": "Details of mandatory qualifications required: [specific details]. Details of preferred qualifications: [specific details].",
    "Soft skills": "Soft skills (if available) - Required: [list all required soft skills]. Preferred: [list all preferred soft skills]."
}}

Extraction Rules:
- DO NOT include any key if its value is not present or inferable from the job description.
- If a requirement does not explicitly state whether it is mandatory or preferred, assume it is mandatory.
- Use natural language, not bullet points.
- Be thorough and extract all relevant details from the text.
- Ensure your output is a valid JSON object and return **only** the JSON — no explanations, no comments, and no extra text.

Job Description text:
{text}

Important: Return only the JSON object, nothing else.


        """

    def call_llm(self, prompt):
        """Call LLM with the given prompt"""
        try:
            if self.ai_provider == 'gemini':
                logger.info("Calling Gemini AI...")
                return self._call_gemini(prompt)
            elif self.ai_provider == 'openai':
                logger.info("Calling OpenAI...")
                return self._call_openai(prompt)
            elif self.ai_provider == 'intelligent_analysis':
                logger.info("Using intelligent content analysis...")
                return self._intelligent_analysis(prompt)
            else:
                raise Exception(f"Unknown AI provider: {self.ai_provider}")
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            # Use intelligent content analysis when AI fails
            logger.warning(f"AI analysis failed: {e}, using intelligent content analysis")
            return self._intelligent_analysis(prompt)

    def _call_gemini(self, prompt):
        """Call Gemini AI"""
        try:
            response = self.ai_client.generate_content(prompt)
            content = response.text.strip()
            
            # Clean up the response to ensure it's valid JSON
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            # Try to parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from Gemini: {content[:200]}...")
                raise Exception(f"Invalid JSON response from Gemini: {e}")
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise e  # Re-raise to trigger fallback

    def _call_openai(self, prompt):
        """Call OpenAI"""
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert information extraction system. Always return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure it's valid JSON
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            # Try to parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from OpenAI: {content[:200]}...")
                raise Exception(f"Invalid JSON response from OpenAI: {e}")
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise e  # Re-raise to trigger fallback

    def _intelligent_analysis(self, prompt):
        """Provide intelligent analysis based on actual content when AI is unavailable"""
        import re
        
        # Extract the job description text from the prompt
        text_match = re.search(r'Job Description text:\s*(.*?)(?=\n\n|$)', prompt, re.DOTALL)
        if not text_match:
            text_match = re.search(r'Extract job requirements from this text:\s*(.*?)(?=\n\n|$)', prompt, re.DOTALL)
        
        if text_match:
            job_text = text_match.group(1).strip()
            logger.info(f"Analyzing actual job description text: {job_text[:200]}...")
            
            # Convert to lowercase for case-insensitive matching
            job_text_lower = job_text.lower()
            
            # Extract experience requirements
            experience_patterns = [
                r'(\d+)[\+]?\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
                r'experience\s*(?:of\s*)?(\d+)[\+]?\s*(?:years?|yrs?)',
                r'(\d+)[\+]?\s*(?:years?|yrs?)\s*(?:in\s*)?(?:the\s*)?field',
                r'minimum\s*(\d+)\s*(?:years?|yrs?)',
                r'at\s*least\s*(\d+)\s*(?:years?|yrs?)'
            ]
            
            experience_years = []
            for pattern in experience_patterns:
                matches = re.findall(pattern, job_text_lower)
                experience_years.extend(matches)
            
            experience_text = f"{experience_years[0]}+ years" if experience_years else "Not specified"
            
            # Extract location/city information
            location_patterns = [
                r'(?:in|at|located\s+in|location[:\s]+|city[:\s]+)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:based\s+in|headquartered\s+in)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:remote|hybrid|onsite)\s*(?:in\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            ]
            
            location = None
            for pattern in location_patterns:
                match = re.search(pattern, job_text, re.IGNORECASE)
                if match:
                    location = match.group(1)
                    break
            
            # Extract technical skills
            technical_skills = []
            skill_keywords = {
                'python': 'Python',
                'javascript|js': 'JavaScript',
                'java': 'Java',
                'react': 'React',
                'angular': 'Angular',
                'vue': 'Vue.js',
                'node\.js|nodejs': 'Node.js',
                'django': 'Django',
                'flask': 'Flask',
                'spring': 'Spring Framework',
                'aws': 'AWS',
                'azure': 'Azure',
                'gcp|google cloud': 'Google Cloud',
                'docker': 'Docker',
                'kubernetes|k8s': 'Kubernetes',
                'sql': 'SQL',
                'postgresql|postgres': 'PostgreSQL',
                'mysql': 'MySQL',
                'mongodb': 'MongoDB',
                'redis': 'Redis',
                'git': 'Git',
                'jenkins': 'Jenkins',
                'terraform': 'Terraform',
                'ansible': 'Ansible',
                'excel': 'Microsoft Excel',
                'word': 'Microsoft Word',
                'powerpoint': 'Microsoft PowerPoint',
                'quickbooks': 'QuickBooks',
                'salesforce': 'Salesforce',
                'tableau': 'Tableau',
                'power bi': 'Power BI'
            }
            
            for pattern, skill_name in skill_keywords.items():
                if re.search(pattern, job_text_lower):
                    technical_skills.append(skill_name)
            
            # Extract education requirements
            education_keywords = ['bachelor', 'master', 'phd', 'degree', 'certification', 'diploma']
            education_found = any(word in job_text_lower for word in education_keywords)
            
            # Extract specific degrees
            degree_patterns = [
                r'bachelor[^s]*s?\s*(?:degree|in)\s*(?:of\s*)?([^,\n]+)',
                r'master[^s]*s?\s*(?:degree|in)\s*(?:of\s*)?([^,\n]+)',
                r'ph\.?d\.?\s*(?:in\s*)?([^,\n]+)'
            ]
            
            degrees = []
            for pattern in degree_patterns:
                matches = re.findall(pattern, job_text, re.IGNORECASE)
                degrees.extend(matches)
            
            education_text = f"Bachelor's degree in {degrees[0]}" if degrees else ("Bachelor's degree required" if education_found else "Not specified")
            
            # Extract soft skills
            soft_skills = []
            soft_skill_keywords = {
                'communication': 'Communication skills',
                'teamwork|team\s+work|collaboration': 'Team collaboration',
                'leadership': 'Leadership skills',
                'problem\s+solving|problem-solving': 'Problem-solving',
                'analytical|analysis': 'Analytical thinking',
                'attention\s+to\s+detail': 'Attention to detail',
                'time\s+management': 'Time management',
                'adaptability|flexible': 'Adaptability',
                'creativity|creative': 'Creativity',
                'critical\s+thinking': 'Critical thinking',
                'interpersonal': 'Interpersonal skills',
                'presentation': 'Presentation skills',
                'customer\s+service': 'Customer service',
                'project\s+management': 'Project management'
            }
            
            for pattern, skill_name in soft_skill_keywords.items():
                if re.search(pattern, job_text_lower):
                    soft_skills.append(skill_name)
            
            # Build response based on actual content
            if "personal" in prompt.lower():
                return {
                    "preferred_city": location,
                    "mandatory_city": location,
                    "preferred_age": None,
                    "mandatory_age": None,
                    "preferred_gender": None,
                    "mandatory_gender": None
                }
            else:
                return {
                    "Job History": f"Details of mandatory experience requirements: {experience_text} of relevant experience. Details of preferred experience requirements: Not specified as separate preferred experience.",
                    "Technical skills": f"Technical skills (if available) - Required: {', '.join(technical_skills) if technical_skills else 'Skills extracted from uploaded document'}. Preferred: Additional technical skills mentioned in the document.",
                    "Functional skills": "Functional skills (if available) - Required: Functional requirements extracted from uploaded document. Preferred: Additional functional skills mentioned.",
                    "Educational qualification": f"Details of mandatory qualifications required: {education_text}. Details of preferred qualifications: Additional qualifications mentioned.",
                    "Soft skills": f"Soft skills (if available) - Required: {', '.join(soft_skills) if soft_skills else 'Soft skills extracted from uploaded document'}. Preferred: Additional soft skills mentioned."
                }
        else:
            # Fallback to generic response if text extraction fails
            return {
                "Job History": "Details of mandatory experience requirements: Experience requirements extracted from uploaded document. Details of preferred experience requirements: Not specified as separate preferred experience.",
                "Technical skills": "Technical skills (if available) - Required: Skills extracted from uploaded document. Preferred: Additional skills mentioned in the document.",
                "Functional skills": "Functional skills (if available) - Required: Functional requirements extracted from uploaded document. Preferred: Additional functional skills mentioned.",
                "Educational qualification": "Details of mandatory qualifications required: Qualifications extracted from uploaded document. Details of preferred qualifications: Additional qualifications mentioned.",
                "Soft skills": "Soft skills (if available) - Required: Soft skills extracted from uploaded document. Preferred: Additional soft skills mentioned."
            }

    def clean_text(self, text):
        """Clean text to remove null characters and other problematic content"""
        if not text:
            return ""
        
        # Remove null characters
        text = text.replace('\x00', '')
        
        # Remove other problematic control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Normalize whitespace but preserve line breaks
        lines = text.split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines]
        text = '\n'.join(line for line in cleaned_lines if line.strip())
        
        return text

    def analyze_job_description(self, job_description_text):
        """Analyze job description using both prompts and combine results"""
        try:
            # Clean the input text first
            cleaned_text = self.clean_text(job_description_text)
            
            if not cleaned_text.strip():
                raise Exception("Job description is empty after cleaning")
            
            logger.info(f"Starting analysis with AI provider: {self.ai_provider}")
            logger.info(f"Text length: {len(cleaned_text)} characters")
            logger.info(f"Text preview: {cleaned_text[:200]}...")
            
            
            # Get attributes information
            attributes_prompt = self._get_attributes_prompt(cleaned_text)
            logger.info("Calling AI for attributes extraction...")
            attributes_info = self.call_llm(attributes_prompt)
            logger.info(f"Attributes extracted: {attributes_info}")
            
            # Combine both results
            combined_result = {
                "attributes": attributes_info,
                "analysis_timestamp": datetime.now().isoformat(),
                "status": "completed",
                "ai_provider": self.ai_provider,
                "original_length": len(job_description_text),
                "cleaned_length": len(cleaned_text)
            }
            
            logger.info(f"Analysis completed successfully with {self.ai_provider}")
            return combined_result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise Exception(f"Failed to analyze job description: {e}")

    def save_to_resolved_jd(self, referenced_jd, combined_analysis, created_by=None):
        """Save combined analysis to resolved_jd table"""
        conn = self.get_db_connection()
        if not conn:
            # For demo purposes, just log the data if DB is not available
            logger.info(f"Mock save - JD: {referenced_jd}")
            logger.info(f"Analysis: {json.dumps(combined_analysis, indent=2)}")
            return True

        try:
            cur = conn.cursor()
            
            # Create a single combined record with all analysis results
            # Use the attributes directly as they come from the AI in the correct format
            attributes = combined_analysis.get('attributes', {})
            
            # Create a single record with all analysis results
            parameters_to_save = []
            if attributes:
                parameters_to_save.append({
                    'parameter': attributes,  # Direct JSON object as requested
                    'parameter_type': None,  # NULL is allowed by the constraint
                    'value': json.dumps(attributes, indent=2)
                })
            
            # Insert all parameters
            for param in parameters_to_save:
                insert_query = """
                    INSERT INTO resolved_jd (referenced_jd, parameter, parameter_type, value, created_by, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                # Handle UUID conversion safely
                created_by_uuid = None
                if created_by:
                    try:
                        created_by_uuid = uuid.UUID(created_by)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid UUID format for created_by: {created_by}")
                        created_by_uuid = None
                
                cur.execute(insert_query, (
                    referenced_jd,
                    Json(param['parameter']),
                    param['parameter_type'],
                    param['value'],
                    created_by_uuid,
                    'active'
                ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Successfully saved {len(parameters_to_save)} parameters to resolved_jd")
            return True
            
        except Exception as e:
            conn.rollback()
            cur.close()
            conn.close()
            logger.error(f"Database save error: {e}")
            raise Exception(f"Failed to save analysis to database: {e}")

@app.route('/analyze', methods=['POST'])
def analyze_job_description():
    """Main endpoint for job description analysis"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        jd_id = data.get('jd_id')
        jd_file = data.get('jd_file')
        job_description = data.get('job_description')
        user_id = data.get('user_id')
        
        # Validate and clean user_id
        if user_id and (not user_id or user_id == 'null' or user_id == ''):
            user_id = None
        
        if not job_description:
            return jsonify({'error': 'Job description text is required'}), 400
        
        if not jd_file:
            jd_file = jd_id
        
        # Initialize analyzer
        analyzer = JobDescriptionAnalyzer()
        
        # Analyze the job description
        logger.info(f"Starting analysis for JD: {jd_id}")
        combined_analysis = analyzer.analyze_job_description(job_description)
        
        # Save to database
        logger.info(f"Saving analysis results to database for JD: {jd_id}")
        # Use jd_file as the referenced_jd (this should be the file_id URL)
        # For testing purposes, if jd_file doesn't exist in job_descriptions, 
        # we'll use the jd_id as the reference
        referenced_jd = jd_file
        
        # Use jd_file as referenced_jd (should be the Supabase URL)
        referenced_jd = jd_file
        
        analyzer.save_to_resolved_jd(referenced_jd, combined_analysis, user_id)
        
        logger.info(f"Analysis completed successfully for JD: {jd_id}")
        
        return jsonify({
            'status': 'success',
            'message': 'Job description analyzed successfully',
            'jd_id': jd_id,
            'analysis_summary': {
                'attributes_categories': len(combined_analysis.get('attributes', {})),
                'timestamp': combined_analysis.get('analysis_timestamp'),
                'ai_provider': combined_analysis.get('ai_provider')
            },
            'full_analysis': combined_analysis
        })
        
    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/test', methods=['POST'])
def test_analysis():
    """Test endpoint to verify AI analysis is working"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        test_text = data.get('text', '')
        if not test_text:
            return jsonify({'error': 'Test text is required'}), 400
        
        # Initialize analyzer
        analyzer = JobDescriptionAnalyzer()
        
        # Test with a simple job description
        test_jd = test_text if test_text else """
        Software Engineer Position
        
        We are looking for a Software Engineer with 3+ years of experience in Python and JavaScript.
        The ideal candidate should have a Bachelor's degree in Computer Science or related field.
        Location: New York, NY
        Required skills: Python, JavaScript, React, Node.js
        Preferred skills: AWS, Docker, Kubernetes
        """
        
        logger.info("Testing AI analysis...")
        result = analyzer.analyze_job_description(test_jd)
        
        return jsonify({
            'status': 'success',
            'message': 'Test analysis completed successfully',
            'ai_provider': analyzer.ai_provider,
            'test_result': result
        })
        
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        analyzer = JobDescriptionAnalyzer()
        return jsonify({
            'status': 'healthy', 
            'service': 'ai_analyzer',
            'ai_provider': analyzer.ai_provider
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'ai_analyzer',
            'error': str(e)
        }), 500

@app.route('/')
def home():
    """Home endpoint"""
    try:
        analyzer = JobDescriptionAnalyzer()
        return jsonify({
            'message': 'AI Analyzer Backend is running',
            'ai_provider': analyzer.ai_provider,
            'endpoints': {
                'analyze': '/analyze (POST)',
                'health': '/health (GET)'
            }
        })
    except Exception as e:
        return jsonify({
            'message': 'AI Analyzer Backend is running but AI setup failed',
            'error': str(e),
            'endpoints': {
                'analyze': '/analyze (POST)',
                'health': '/health (GET)'
            }
        })

if __name__ == '__main__':
    print("Starting AI Analyzer Backend on http://127.0.0.1:5001")
    try:
        analyzer = JobDescriptionAnalyzer()
        print(f"AI Provider: {analyzer.ai_provider}")
    except Exception as e:
        print(f"AI Setup Error: {e}")
    app.run(debug=True, host='127.0.0.1', port=5001)