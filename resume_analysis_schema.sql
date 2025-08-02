-- Resume Analysis Table (Single table approach)
CREATE TABLE IF NOT EXISTS public.resume_analyses (
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

-- Indexes for resume_analyses
CREATE INDEX IF NOT EXISTS idx_resume_analyses_jd_id ON public.resume_analyses USING btree (jd_id);
CREATE INDEX IF NOT EXISTS idx_resume_analyses_filename ON public.resume_analyses USING btree (resume_filename);
CREATE INDEX IF NOT EXISTS idx_resume_analyses_status ON public.resume_analyses USING btree (status);
CREATE INDEX IF NOT EXISTS idx_resume_analyses_created_at ON public.resume_analyses USING btree (created_at);

-- Resume Scoring Table for Criteria-based Calculations
CREATE TABLE IF NOT EXISTS public.resume_scores (
  score_id uuid not null default gen_random_uuid (),
  analysis_id uuid not null,
  criteria_id uuid not null,
  jd_id uuid not null,
  resume_filename text not null,
  parameter_scores jsonb not null, -- Stores individual parameter scores
  final_score numeric(6,2) not null, -- Final calculated score (increased precision to handle scores up to 9999.99)
  recommendation text not null, -- 'To be interviewed', 'Candidature rejected', 'Review further'
  consideration text, -- Detailed explanation
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  status character varying(50) null default 'active'::character varying,
  constraint resume_scores_pkey primary key (score_id),
  constraint fk_resume_scores_analysis foreign key (analysis_id) references resume_analyses(analysis_id) on delete cascade,
  constraint fk_resume_scores_criteria foreign key (criteria_id) references criteria(criteria_id) on delete cascade,
  constraint fk_resume_scores_jd foreign key (jd_id) references job_descriptions(jd_id) on delete cascade
);

-- Indexes for resume_scores
CREATE INDEX IF NOT EXISTS idx_resume_scores_analysis_id ON public.resume_scores USING btree (analysis_id);
CREATE INDEX IF NOT EXISTS idx_resume_scores_criteria_id ON public.resume_scores USING btree (criteria_id);
CREATE INDEX IF NOT EXISTS idx_resume_scores_jd_id ON public.resume_scores USING btree (jd_id);
CREATE INDEX IF NOT EXISTS idx_resume_scores_final_score ON public.resume_scores USING btree (final_score);
CREATE INDEX IF NOT EXISTS idx_resume_scores_recommendation ON public.resume_scores USING btree (recommendation);
CREATE INDEX IF NOT EXISTS idx_resume_scores_created_at ON public.resume_scores USING btree (created_at); 