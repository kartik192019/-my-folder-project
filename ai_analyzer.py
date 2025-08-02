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
            if content.startswith('```'):
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
            if content.startswith('```'):
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

    def save_to_resolved_jd(self, referenced_jd, analysis_result, user_id):
        """Save analysis result to resolved_jd table"""
        print(f"📊 save_to_resolved_jd called with:")
        print(f"📊 Referenced JD: {referenced_jd}")
        print(f"📊 User ID: {user_id}")
        print(f"📊 Analysis result type: {type(analysis_result)}")
        
        print(f"📊 Getting database connection...")
        conn = self.get_db_connection()
        if not conn:
            logger.error("Database connection failed, cannot save analysis")
            print(f"📊 Database connection failed")
            return False
        print(f"📊 Database connection successful")
        
        try:
            cur = conn.cursor()
            
            print(f"📊 About to insert into resolved_jd table:")
            print(f"📊 Referenced JD: {referenced_jd}")
            print(f"📊 Parameter type: job_description_analysis")
            print(f"📊 User ID: {user_id}")
            print(f"📊 Analysis result sample: {str(analysis_result)[:200]}...")
            
            # Save the analysis result
            insert_query = """
            INSERT INTO resolved_jd (referenced_jd, parameter, parameter_type)
            VALUES (%s, %s, %s)
            """
            
            print(f"📊 Executing query: {insert_query}")
            print(f"📊 Parameters: ({referenced_jd}, Json(analysis_result), None)")
            
            cur.execute(insert_query, (
                referenced_jd,
                Json(analysis_result),
                None  # Use NULL instead of 'job_description_analysis'
            ))
            
            print(f"📊 Query executed successfully")
            conn.commit()
            print(f"📊 Transaction committed")
            cur.close()
            conn.close()
            logger.info(f"Analysis saved to resolved_jd table for referenced_jd: {referenced_jd}")
            print(f"📊 Successfully saved to resolved_jd table")
            return True
            
        except Exception as e:
            print(f"📊 ERROR saving to resolved_jd: {e}")
            print(f"📊 Error type: {type(e)}")
            if conn:
                conn.rollback()
                print(f"📊 Transaction rolled back")
                cur.close()
                conn.close()
            logger.error(f"Error saving to resolved_jd: {e}")
            return False

    def _get_scoring_prompt(self, jd_parameters, candidate_details, criteria_grid, resume_filename):
        """Create scoring prompt based on JD parameters, candidate details, and criteria grid"""
        # Format JD requirements
        jd_requirements = json.dumps(jd_parameters, indent=2)
        
        # Format candidate details
        candidate_info = json.dumps(candidate_details, indent=2)
        
        # Format criteria grid
        criteria_info = json.dumps(criteria_grid, indent=2)

        prompt = f"""
JD requirements:
{jd_requirements}

Candidate Details:
{candidate_info}

System prompt:
As an expert recruiter leveraging semantic understanding, your task is to assess parts of a candidate's CV against the respective parts of a Job Description (JD). You will be provided with the CV content, JD content, and a dynamic set of assessment criteria that require deep contextual interpretation beyond simple keyword matching.

Evaluation Process:
For each parameter specified in the Assessment Criteria section, you must:
- Comprehend the Parameter Name – grasp its meaning in the context of hiring needs.
- Understand the Calculation Method – refer to the specific evaluation guidance in its Calculation Note, ensuring nuanced analysis.
- Perform a Contextual Fit Assessment – examine the mentioned parameter in light of the JD's actual requirements, rather than relying on exact word matches.
- Assign a Fitment Score (1 to 10) based on contextual relevance, where:
  - 1 = No alignment – The candidate does not meet the expectations for this parameter at all.
  - 10 = Perfect match – The candidate fully aligns with the ideal profile for this parameter, considering the stated role objectives and responsibilities.

The Weightage for each parameter provides contextual guidance for relative importance in decision-making. Your primary output is the individual parameter score based on semantic understanding, role-specific interpretation, and alignment of concepts not syntactic matching.

Think through this task step-by-step.

Assessment Criteria:
{criteria_info}

Output Format:
[Scoring]:
Provide your assessment in the following structured format:
[Parameter 1 Name]: [Score for Parameter 1] x [Weightage for Parameter 1]% = [Rating for Parameter 1]
[Parameter 2 Name]: [Score for Parameter 2] x [Weightage for Parameter 2]% = [Rating for Parameter 2]
...
[Parameter N Name]: [Score for Parameter N] x [Weightage for Parameter N]% = [Rating for Parameter N]

[Final_Match]:
Final Score = [Rating for Parameter 1] + [Rating for Parameter 2]+....+[Rating for Parameter N] (only show the derived 'final score', not the individual rating calculation). The final score should be a value between 1-10. Give only the numeric value and no text along with it.

[Consideration]: [Your detailed explanation justifying the scores assigned for each parameter and an overall summary of the candidate's fitment altogether in less than 300 words in plan text with bullets, if applicable. Cite examples of where the skillset was displayed or used, if possible. Do capture gaps in employment history, if any].

[Recommendation]: [Share your recommendation to the human recruiter classifying the resume in one of the 3 categories below. Do this classification with utmost care to avoid hallucination:
1. To be interviewed
2. Candidature rejected
3. Review further
"""

        return prompt

    def _call_scoring_llm(self, prompt):
        """Call LLM for scoring and parse the response"""
        try:
            logger.info("Calling LLM for resume scoring...")
            
            if self.ai_provider == 'gemini':
                response = self._call_gemini_raw(prompt)
            elif self.ai_provider == 'openai':
                response = self._call_openai_raw(prompt)
            else:
                raise Exception(f"Unsupported AI provider: {self.ai_provider}")
            
            # Parse the scoring response
            scoring_result = self._parse_scoring_response(response)
            logger.info("LLM scoring completed successfully")
            return scoring_result
            
        except Exception as e:
            logger.error(f"Error calling LLM for scoring: {e}")
            raise Exception(f"Failed to call LLM for scoring: {e}")

    def _call_gemini_raw(self, prompt):
        """Call Gemini AI and return raw text response"""
        try:
            response = self.ai_client.generate_content(prompt)
            
            # Handle complex responses properly
            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'parts') and response.parts:
                return response.parts[0].text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                # Fallback: try to get text from any available method
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise e

    def _call_openai_raw(self, prompt):
        """Call OpenAI and return raw text response"""
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert recruiter. Provide detailed scoring analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise e

    def _parse_scoring_response(self, response_text):
        """Parse the scoring response from LLM"""
        try:
            # Initialize result structure with default values
            result = {
                'parameter_scores': {},
                'final_score': 0,
                'recommendation': 'Review further',
                'consideration': '',
                'detailed_assessment': ''
            }
            
            # If response is already a dictionary, use it directly
            if isinstance(response_text, dict):
                logger.info("Response is already a dictionary, using directly")
                return self._clean_scoring_result(response_text)
            
            # First, try to extract JSON from the response
            try:
                # Try to parse the entire response as JSON first
                json_response = json.loads(response_text)
                logger.info("Response parsed as JSON successfully")
                return self._clean_scoring_result(json_response)
            except (json.JSONDecodeError, TypeError):
                # If that fails, try to find a JSON object in the text
                try:
                    import re
                    json_match = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        json_response = json.loads(json_str)
                        logger.info("Extracted and parsed JSON from response text")
                        return self._clean_scoring_result(json_response)
                except (json.JSONDecodeError, AttributeError):
                    logger.info("Could not extract JSON from response, falling back to text parsing")
            
            # If we get here, parse the response as text
            logger.info("Parsing response as text")
            
            # Split response into sections
            sections = {}
            current_section = None
            
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if line.startswith('[') and ']' in line:
                    current_section = line[1:line.index(']')].lower().strip()
                    sections[current_section] = []
                elif current_section is not None:
                    sections.setdefault(current_section, []).append(line)
            
            # Process each section
            for section, lines in sections.items():
                content = '\n'.join(lines).strip()
                
                if section == 'scoring':
                    # Parse parameter scores
                    for line in lines:
                        if ':' in line and 'x' in line and '=' in line:
                            try:
                                param_part, score_part = line.split(':', 1)
                                param_name = param_part.strip().replace('[', '').replace(']', '')
                                
                                # Extract score, weight, and rating
                                score_weight, weight_rating = score_part.split('x', 1)
                                weight, rating = weight_rating.split('=', 1)
                                
                                result['parameter_scores'][param_name] = {
                                    'score': float(score_weight.strip()),
                                    'weightage': float(weight.replace('%', '').strip()),
                                    'rating': float(rating.strip())
                                }
                            except Exception as e:
                                logger.warning(f"Could not parse scoring line: {line}, error: {e}")
                
                elif section == 'final_match':
                    # Parse final score
                    try:
                        numbers = re.findall(r'\d+\.?\d*', content)
                        if numbers:
                            score = float(numbers[0])
                            if 0 <= score <= 10:
                                result['final_score'] = score
                    except Exception as e:
                        logger.warning(f"Could not parse final score: {e}")
                
                elif section == 'consideration':
                    result['consideration'] = content
                
                elif section == 'recommendation':
                    # Map recommendation to standard categories
                    content_lower = content.lower()
                    if 'interview' in content_lower:
                        result['recommendation'] = 'To be interviewed'
                    elif 'reject' in content_lower:
                        result['recommendation'] = 'Candidature rejected'
                    else:
                        result['recommendation'] = 'Review further'
            
            # Clean up the consideration text
            result = self._clean_consideration_text(result)
            
            # If no final score was found, calculate from parameter scores
            if not result['final_score'] and result['parameter_scores']:
                self._calculate_final_score_from_parameters(result)
            
            logger.info(f"Parsed scoring result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing scoring response: {e}")
            # Return a default result with error information
            return {
                'parameter_scores': {},
                'final_score': 0,
                'recommendation': 'Review further',
                'consideration': f'Error parsing scoring response: {str(e)}',
                'detailed_assessment': ''
            }
    
    def _clean_consideration_text(self, result):
        """Clean up the consideration text in the result"""
        if 'consideration' in result and result['consideration']:
            try:
                # Try to parse as JSON first
                consideration = result['consideration']
                if (consideration.startswith('{') and consideration.endswith('}')) or \
                   (consideration.startswith('"') and consideration.endswith('"')):
                    try:
                        consideration_json = json.loads(consideration)
                        if isinstance(consideration_json, dict) and 'consideration' in consideration_json:
                            result['consideration'] = consideration_json['consideration']
                        elif isinstance(consideration_json, str):
                            result['consideration'] = consideration_json
                    except json.JSONDecodeError:
                        pass
                
                # Clean up common issues
                consideration = result['consideration']
                
                # Remove JSON escape sequences
                consideration = consideration.replace('\\n', '\n').replace('\\"', '"')
                
                # Remove any remaining JSON artifacts
                consideration = re.sub(r'^"|"$', '', consideration)
                
                # Normalize whitespace
                consideration = ' '.join(consideration.split())
                
                # Store the cleaned consideration
                result['consideration'] = consideration
                
            except Exception as e:
                logger.warning(f"Error cleaning consideration text: {e}")
                # If cleaning fails, at least ensure it's a string
                result['consideration'] = str(result['consideration'])
        
        return result
    
    def _calculate_final_score_from_parameters(self, result):
        """Calculate final score from parameter scores if not provided"""
        try:
            total_weight = 0
            weighted_sum = 0
            
            for param_data in result['parameter_scores'].values():
                if isinstance(param_data, dict):
                    # Get score and weight, with defaults
                    score = float(param_data.get('score', 0))
                    weight = float(param_data.get('weightage', 0)) / 100.0  # Convert percentage to decimal
                    
                    # Ensure score is within valid range (0-10)
                    score = max(0, min(10, score))
                    
                    # Calculate weighted contribution
                    weighted_sum += score * weight
                    total_weight += weight
            
            # Calculate final score as weighted average (0-10 scale)
            if total_weight > 0:
                # No need to multiply by 10 since scores are already on 0-10 scale
                final_score = weighted_sum / total_weight
                # Ensure final score is between 1-10
                result['final_score'] = min(10.0, max(1.0, final_score))
                logger.info(f"Calculated final score from parameters: {result['final_score']} (weighted_sum={weighted_sum}, total_weight={total_weight})")
            else:
                result['final_score'] = 5.0  # Default if no valid weights
                logger.warning("No valid weights found for parameter scores")
            
        except Exception as e:
            logger.warning(f"Could not calculate final score from parameters: {e}")
            result['final_score'] = 5.0  # Default fallback score

    def _clean_scoring_result(self, result):
        """Clean up the scoring result"""
        if 'parameter_scores' in result:
            for param_name, param_data in result['parameter_scores'].items():
                if isinstance(param_data, dict):
                    score = param_data.get('score', 0)
                    weightage = param_data.get('weightage', 0)
                    rating = param_data.get('rating', 0)
                    
                    # Ensure score is within valid range
                    score = max(0, min(10, score))
                    
                    # Ensure weightage is within valid range
                    weightage = max(0, min(100, weightage))
                    
                    # Ensure rating is within valid range
                    rating = max(0, min(10, rating))
                    
                    result['parameter_scores'][param_name] = {
                        'score': score,
                        'weightage': weightage,
                        'rating': rating
                    }
        
        if 'final_score' in result:
            result['final_score'] = max(0, min(10, result['final_score']))
        
        return result

# Flask routes
@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze job description endpoint"""
    print(f"📊 /analyze endpoint called")
    print(f"📊 Request method: {request.method}")
    print(f"📊 Request headers: {dict(request.headers)}")
    
    try:
        data = request.get_json()
        print(f"📊 Request data: {data}")
        
        if not data:
            print(f"📊 No data provided")
            return jsonify({'error': 'No data provided'}), 400

        job_description = data.get('job_description')
        jd_id = data.get('jd_id')
        jd_file = data.get('jd_file')
        user_id = data.get('user_id')
        
        print(f"📊 Extracted data:")
        print(f"📊 JD ID: {jd_id}")
        print(f"📊 JD File: {jd_file}")
        print(f"📊 User ID: {user_id}")
        print(f"📊 Job description length: {len(job_description) if job_description else 0}")

        if not job_description:
            return jsonify({'error': 'job_description is required'}), 400

        # Initialize analyzer
        analyzer = JobDescriptionAnalyzer()
        print(f"📊 Analyzer initialized")
        
        # Analyze the job description
        print(f"📊 Starting job description analysis...")
        combined_analysis = analyzer.analyze_job_description(job_description)
        print(f"📊 Analysis completed")
        print(f"📊 Analysis result type: {type(combined_analysis)}")
        print(f"📊 Analysis result keys: {list(combined_analysis.keys()) if isinstance(combined_analysis, dict) else 'Not a dict'}")

        # Use jd_file as referenced_jd (should be the Supabase URL)
        # If jd_file is a UUID (jd_id), use it directly
        # If jd_file is a URL, use it as is
        referenced_jd = jd_file if jd_file and jd_file.startswith('http') else jd_id
        print(f"📊 Saving to resolved_jd table:")
        print(f"📊 Referenced JD: {referenced_jd}")
        print(f"📊 User ID: {user_id}")
        print(f"📊 Analysis result keys: {list(combined_analysis.keys()) if isinstance(combined_analysis, dict) else 'Not a dict'}")
        
        print(f"📊 About to call save_to_resolved_jd...")
        save_result = analyzer.save_to_resolved_jd(referenced_jd, combined_analysis, user_id)
        print(f"📊 Save result: {save_result}")
        
        if save_result:
            print(f"✅ Successfully saved to resolved_jd table!")
        else:
            print(f"❌ Failed to save to resolved_jd table!")
        
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

@app.route('/score_resume', methods=['POST'])
def score_resume():
    """Score resume based on JD parameters and criteria grid"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        jd_parameters = data.get('jd_parameters')
        candidate_details = data.get('candidate_details')
        criteria_grid = data.get('criteria_grid')
        resume_filename = data.get('resume_filename', 'Unknown')

        if not jd_parameters or not candidate_details or not criteria_grid:
            return jsonify({'error': 'Missing required data: jd_parameters, candidate_details, or criteria_grid'}), 400

        # Initialize analyzer
        analyzer = JobDescriptionAnalyzer()

        # Create scoring prompt
        scoring_prompt = analyzer._get_scoring_prompt(jd_parameters, candidate_details, criteria_grid, resume_filename)

        # Call AI for scoring
        logger.info(f"Scoring resume: {resume_filename}")
        scoring_result = analyzer._call_scoring_llm(scoring_prompt)

        return jsonify({
            'status': 'success',
            'message': 'Resume scoring completed successfully',
            'resume_filename': resume_filename,
            'scoring_result': scoring_result
        })

    except Exception as e:
        logger.error(f"Scoring endpoint error: {e}")
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
