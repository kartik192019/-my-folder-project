import os

class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Supabase Configuration
    SUPABASE_URL = 'https://wyuncrdbikdwqstdrvbg.supabase.co'
    SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind5dW5jcmRiaWtkd3FzdGRydmJnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3MDUxODMsImV4cCI6MjA2OTI4MTE4M30.ZqmlJK3QBp0l0aXSXnw_GBjP6Ypt6kVeZXxu_dURqUM'
    
    # Supabase Storage Configuration
    SUPABASE_STORAGE_BUCKET = 'job-descriptions'
    SUPABASE_RESUME_BUCKET = 'resumes'  # Using the new resumes bucket
    SUPABASE_STORAGE_PUBLIC = True
    
    # AI Configuration - Use only one API key
    # Choose either Gemini or OpenAI, but not both
    GEMINI_API_KEY = 'AIzaSyBnRGFBXDBmoyRZzmEnSln_HrDVerB-rzw'  # Your Gemini API key
    OPENAI_API_KEY = ''  # Leave empty if not using OpenAI
    
    # Set this to 'gemini' or 'openai' to specify which API to use
    PREFERRED_AI_PROVIDER = 'gemini'  # Using Gemini as the preferred provider
    
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'aws-0-ap-south-1.pooler.supabase.com')
    DB_PORT = os.getenv('DB_PORT', '6543')
    DB_NAME = os.getenv('DB_NAME', 'postgres')
    DB_USER = os.getenv('DB_USER', 'postgres.wyuncrdbikdwqstdrvbg')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'nG4stT2sWAmghHPM')
    
    @classmethod
    def get_db_config(cls):
        """Get database configuration as dictionary"""
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'database': cls.DB_NAME,
            'user': cls.DB_USER,
            'password': cls.DB_PASSWORD
        }
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_configs = [
            ('SUPABASE_URL', cls.SUPABASE_URL),
            ('SUPABASE_KEY', cls.SUPABASE_KEY),
            ('SUPABASE_STORAGE_BUCKET', cls.SUPABASE_STORAGE_BUCKET)
        ]
        
        missing_configs = []
        for name, value in required_configs:
            if not value or value == '':
                missing_configs.append(name)
        
        if missing_configs:
            raise ValueError(f"Missing required configuration: {', '.join(missing_configs)}")
        
        return True
