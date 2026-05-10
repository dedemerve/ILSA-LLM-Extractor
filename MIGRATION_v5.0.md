# ILSA Schema Refactoring Guide: v4.1 → v5.0

## Overview
This guide documents the safe migration from schema v4.1 (10-variable core_data model) to v5.0 (8-variable model with outcome_summary).

## Changes Summary

| Category | Action | Variables |
|----------|--------|-----------|
| **Removed** | Drop from core_data | validation_strategy, baseline_model_type, code_available, best_metric_name, best_metric_value |
| **Removed** | Drop junction table | core_overfitting_checks (overfitting_checks) |
| **Kept** | Maintain in core_data | ilsa_type, ilsa_year, sample_size, research_design_type, ml_technique_primary, survey_weights_used |
| **Kept** | Maintain in junction tables | ilsa_types_all, ml_techniques_all, confounders_identified |
| **Added** | New field in core_data | outcome_summary (TEXT, nullable) |

## Migration Path (7 Steps)

### Step 1: Backup Current Database
```bash
cp ilsa_knowledge_base.db ilsa_knowledge_base.db.backup.v4.1
```

### Step 2: Create Migration Script
Generate this SQL script to handle both new and existing databases:

```sql
-- schema_v5.0.sql already includes:
-- 1. Updated core_data table definition (8 columns)
-- 2. Removal of validation_strategy, baseline_model_type, code_available, 
--    best_metric_name, best_metric_value columns
-- 3. Addition of outcome_summary column
-- 4. Removal of core_overfitting_checks junction table
-- 5. Updated indices

-- For existing databases with v4.1 schema:
-- ALTER TABLE core_data DROP COLUMN validation_strategy;
-- ALTER TABLE core_data DROP COLUMN baseline_model_type;
-- ALTER TABLE core_data DROP COLUMN code_available;
-- ALTER TABLE core_data DROP COLUMN best_metric_name;
-- ALTER TABLE core_data DROP COLUMN best_metric_value;
-- ALTER TABLE core_data ADD COLUMN outcome_summary TEXT;
-- DROP TABLE IF EXISTS core_overfitting_checks;

-- Note: SQLite has limited ALTER TABLE support. Use the approach below instead.
```

### Step 3: Migration for Existing SQLite Databases
If you have existing v4.1 data, use this Python script to safely migrate:

```python
import sqlite3
from pathlib import Path

def migrate_v4_1_to_v5_0(db_path: str) -> None:
    """Safely migrate core_data table from v4.1 to v5.0 schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Step 1: Verify current schema
        cursor.execute("PRAGMA table_info(core_data)")
        columns = {row[1] for row in cursor.fetchall()}
        
        # Step 2: Create backup table with v4.1 structure
        cursor.execute("""
            CREATE TABLE core_data_backup AS 
            SELECT * FROM core_data
        """)
        
        # Step 3: Drop old table
        cursor.execute("DROP TABLE core_data")
        
        # Step 4: Create new table with v5.0 structure
        cursor.execute("""
            CREATE TABLE core_data (
                file_name              TEXT PRIMARY KEY,
                ilsa_type              TEXT,
                ilsa_year              INTEGER,
                sample_size            INTEGER,
                research_design_type   TEXT,
                ml_technique_primary   TEXT,
                survey_weights_used    INTEGER,
                outcome_summary        TEXT,
                FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
            )
        """)
        
        # Step 5: Migrate data from backup (outcome_summary defaults to NULL)
        cursor.execute("""
            INSERT INTO core_data 
            (file_name, ilsa_type, ilsa_year, sample_size, research_design_type, 
             ml_technique_primary, survey_weights_used, outcome_summary)
            SELECT 
                file_name, ilsa_type, ilsa_year, sample_size, research_design_type,
                ml_technique_primary, survey_weights_used, NULL
            FROM core_data_backup
        """)
        
        # Step 6: Drop backup and overfitting_checks table
        cursor.execute("DROP TABLE core_data_backup")
        cursor.execute("DROP TABLE IF EXISTS core_overfitting_checks")
        
        # Step 7: Recreate indices
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_core_ilsa_type
            ON core_data (ilsa_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_core_ml_technique
            ON core_data (ml_technique_primary)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_core_research_design
            ON core_data (research_design_type)
        """)
        
        # Step 8: Record schema version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO schema_version (version, description)
            VALUES ('v5.0', 'Refactored to 8-variable model: removed validation/metrics, added outcome_summary')
        """)
        
        conn.commit()
        print(f"✓ Successfully migrated {db_path} to v5.0")
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_v4_1_to_v5_0("ilsa_knowledge_base.db")
```

### Step 4: Update Extraction Code
The updated `CoreData` class in `ilsa_pipeline/schemas/article_schema.py` now expects:
- All 7 core variables (kept from v4.1)
- New `outcome_summary` field (required for extraction, but defaults to None)

### Step 5: Update GPT-4o Extraction Prompt
Modify the extraction prompt to include outcome_summary instructions:

```
For outcome_summary: Provide a 2-4 sentence summary that captures:
1. Which factors emerged as most predictive (based on feature importance or model coefficients)
2. What the model revealed about the outcome variable
3. Any notable or unexpected findings

Example:
"XGBoost identified socioeconomic status (ESCS) and reading comprehension as 
the top two predictors of mathematics achievement, explaining 42% of variance. 
The model outperformed baseline OLS regression by 18% (R² 0.74 vs 0.56). 
Notably, school infrastructure had minimal predictive power after controlling for SES."
```

### Step 6: Validate Migration
Run this validation script to ensure data integrity:

```python
def validate_v5_0_schema(db_path: str) -> None:
    """Validate that migration to v5.0 completed successfully."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check core_data columns
    cursor.execute("PRAGMA table_info(core_data)")
    columns = {row[1] for row in cursor.fetchall()}
    
    expected = {
        'file_name', 'ilsa_type', 'ilsa_year', 'sample_size',
        'research_design_type', 'ml_technique_primary', 
        'survey_weights_used', 'outcome_summary'
    }
    
    removed = {
        'validation_strategy', 'baseline_model_type', 'code_available',
        'best_metric_name', 'best_metric_value'
    }
    
    assert columns == expected, f"Column mismatch. Got: {columns}"
    assert not (columns & removed), f"Removed columns still present: {columns & removed}"
    
    # Check overfitting_checks table is gone
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='core_overfitting_checks'
    """)
    assert not cursor.fetchone(), "core_overfitting_checks table still exists"
    
    # Check row counts
    cursor.execute("SELECT COUNT(*) FROM core_data")
    count = cursor.fetchone()[0]
    print(f"✓ Schema valid. {count} rows in core_data")
    
    conn.close()

if __name__ == "__main__":
    validate_v5_0_schema("ilsa_knowledge_base.db")
```

### Step 7: Update Application Code
- Update `ilsa_pipeline/utils/storage.py`: ✓ Already done
- Update `ilsa_pipeline/schemas/article_schema.py`: ✓ Already done
- Update any downstream queries that reference removed columns

## Handling NULL Values in outcome_summary

During transition period:
- Existing records: `outcome_summary` will be NULL (backward compatible)
- New extractions: populate `outcome_summary` from GPT-4o extraction
- Optional field: application code must handle None gracefully

```python
outcome_text = record.outcome_summary or "(Not provided)"
```

## Backward Compatibility

**Before Migration (v4.1 → After Migration (v5.0):**

| Old Field | Status | Replacement |
|-----------|--------|-------------|
| validation_strategy | Removed | Inference from best_metric_value and ml_technique_primary |
| baseline_model_type | Removed | Extract from outcome_summary context |
| code_available | Removed | Query metadata directly if needed |
| best_metric_name | Removed | Infer from outcome_summary or ML task type |
| best_metric_value | Removed | Recover from outcome_summary text if critical |
| overfitting_checks | Removed | Implicit in research_design_type and ml_technique_primary |

**Non-Breaking:**
- metadata table structure unchanged
- Junction tables (authors, ILSA types, ML techniques, confounders) structure unchanged
- study_id (file_name) primary key unchanged
- Foreign key constraints preserved

## Rollback Plan

If migration fails:
```bash
rm ilsa_knowledge_base.db
cp ilsa_knowledge_base.db.backup.v4.1 ilsa_knowledge_base.db
git checkout HEAD -- ilsa_pipeline/schemas/article_schema.py ilsa_pipeline/utils/storage.py
```

## Timeline & Testing

- **QA Phase**: Test migration script on copy of production database
- **Staging**: Deploy to staging environment, validate queries and reports
- **Production**: Run migration during maintenance window (low-traffic period)
- **Monitoring**: Track error logs for any unforeseen schema-related failures
