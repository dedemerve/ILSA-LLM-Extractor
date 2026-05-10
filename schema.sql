-- ILSA Literature Review Database Schema
-- PostgreSQL 14+
-- Supervisor rationale:
--   Variables 1-7 define the INPUT pipeline (who, what, how).
--   Variable 8 (outcome_summary) closes the OUTPUT loop (what was found).


-- metadata table (12 fields)
CREATE TABLE IF NOT EXISTS metadata (
    file_name               TEXT PRIMARY KEY,
    title                   TEXT,
    authors                 TEXT,
    year                    INTEGER,
    doi                     TEXT,
    venue                   TEXT,
    publication_type        TEXT,
    open_access             BOOLEAN,
    extraction_timestamp    TIMESTAMP WITH TIME ZONE,
    extraction_cost_usd     NUMERIC(10, 6),
    prompt_tokens           INTEGER,
    completion_tokens       INTEGER
);


-- core_data table: 8-variable model
CREATE TABLE IF NOT EXISTS core_data (
    file_name               TEXT PRIMARY KEY REFERENCES metadata(file_name) ON DELETE CASCADE,
    ilsa_type               TEXT,
    ilsa_year               INTEGER,
    sample_size             INTEGER,
    research_design_type    TEXT,
    ml_technique_primary    TEXT,
    survey_weights_used     BOOLEAN,
    confounders_identified  TEXT[],
    outcome_summary         TEXT
);

-- Remove old columns if migrating from a prior schema (idempotent with IF EXISTS)
ALTER TABLE core_data DROP COLUMN IF EXISTS validation_strategy;
ALTER TABLE core_data DROP COLUMN IF EXISTS baseline_model_type;
ALTER TABLE core_data DROP COLUMN IF EXISTS code_available;
ALTER TABLE core_data DROP COLUMN IF EXISTS overfitting_checks;
ALTER TABLE core_data DROP COLUMN IF EXISTS best_metric_name;
ALTER TABLE core_data DROP COLUMN IF EXISTS best_metric_value;

-- Add new columns if not already present (idempotent)
ALTER TABLE core_data ADD COLUMN IF NOT EXISTS confounders_identified TEXT[];
ALTER TABLE core_data ADD COLUMN IF NOT EXISTS outcome_summary TEXT;


-- Junction tables
CREATE TABLE IF NOT EXISTS metadata_authors (
    file_name   TEXT    NOT NULL REFERENCES metadata(file_name) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    author_name TEXT    NOT NULL,
    PRIMARY KEY (file_name, ordinal)
);

CREATE TABLE IF NOT EXISTS core_ilsa_types_all (
    file_name TEXT    NOT NULL REFERENCES metadata(file_name) ON DELETE CASCADE,
    ordinal   INTEGER NOT NULL,
    ilsa_type TEXT    NOT NULL,
    PRIMARY KEY (file_name, ordinal)
);

CREATE TABLE IF NOT EXISTS core_ml_techniques_all (
    file_name TEXT    NOT NULL REFERENCES metadata(file_name) ON DELETE CASCADE,
    ordinal   INTEGER NOT NULL,
    technique TEXT    NOT NULL,
    PRIMARY KEY (file_name, ordinal)
);


-- Indexes
CREATE INDEX IF NOT EXISTS idx_metadata_year        ON metadata  (year);
CREATE INDEX IF NOT EXISTS idx_core_ilsa_type       ON core_data (ilsa_type);
CREATE INDEX IF NOT EXISTS idx_core_ml_technique    ON core_data (ml_technique_primary);
CREATE INDEX IF NOT EXISTS idx_core_research_design ON core_data (research_design_type);
CREATE INDEX IF NOT EXISTS idx_core_weights         ON core_data (survey_weights_used);

DROP INDEX IF EXISTS idx_core_validation;
DROP INDEX IF EXISTS idx_core_baseline_model;


-- View: v_study_overview
CREATE OR REPLACE VIEW v_study_overview AS
SELECT
    m.file_name,
    m.title,
    m.year,
    m.venue,
    m.publication_type,
    c.ilsa_type,
    c.ilsa_year,
    c.sample_size,
    c.research_design_type,
    c.ml_technique_primary,
    c.survey_weights_used,
    c.confounders_identified,
    CARDINALITY(c.confounders_identified) AS confounder_count,
    c.outcome_summary
FROM metadata m
JOIN core_data c USING (file_name);


-- View: v_quality_flags
CREATE OR REPLACE VIEW v_quality_flags AS
SELECT
    m.file_name,
    m.title,
    m.year,
    c.survey_weights_used,
    CASE
        WHEN c.survey_weights_used IS NULL  THEN 'unspecified'
        WHEN c.survey_weights_used = TRUE   THEN 'pass'
        ELSE                                     'fail'
    END AS weights_flag,
    CASE
        WHEN c.confounders_identified IS NULL
          OR CARDINALITY(c.confounders_identified) = 0 THEN 'no_confounders_reported'
        ELSE                                                'confounders_reported'
    END AS confounder_flag,
    CASE
        WHEN c.outcome_summary IS NULL OR TRIM(c.outcome_summary) = '' THEN 'missing_outcome'
        ELSE                                                                 'has_outcome'
    END AS outcome_flag
FROM metadata m
JOIN core_data c USING (file_name);


-- Trigger: auto-populate extraction_timestamp on INSERT when caller omits it
CREATE OR REPLACE FUNCTION set_extraction_timestamp()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.extraction_timestamp IS NULL THEN
        NEW.extraction_timestamp := NOW();
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_metadata_timestamp ON metadata;
CREATE TRIGGER trg_metadata_timestamp
    BEFORE INSERT OR UPDATE ON metadata
    FOR EACH ROW
    WHEN (NEW.extraction_timestamp IS NULL)
    EXECUTE FUNCTION set_extraction_timestamp();
