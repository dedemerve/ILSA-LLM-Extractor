"""
Three-layer storage for extraction results.

v5.4 — SQLite persistence aligned with nested ILSAArticleMetadata (metadata + data).
Changes from v5.2:
  - metadata: added source_category
  - core_data: removed obsolete ilsa_type/ilsa_year;
    added replicate_weights_used, weight_variable_name,
    plausible_values_handling, missing_data_handling,
    feature_selection, baseline_model, xai_method
  - renamed: sample_size -> total_students, survey_weights_used -> student_weights_used
  - new junction table: core_countries (country_code, n_students)
  - removed: core_ilsa_types_all (no longer in schema)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.schemas.models import ILSAArticleMetadata
from src.extractors.gpt_extractor import ExtractionResult
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON layer
# ---------------------------------------------------------------------------

def save_json(result: ExtractionResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.file_name).stem
    json_path = output_dir / f"{stem}.json"

    payload = {
        "file_name": result.file_name,
        "success": result.success,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "duration_seconds": result.duration_seconds,
        "error": result.error,
        "extraction": result.extraction.model_dump() if result.extraction else None,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return json_path


# ---------------------------------------------------------------------------
# Parquet layer
# ---------------------------------------------------------------------------

def flatten_extraction(extraction, file_name: str) -> dict:
    d = extraction if isinstance(extraction, dict) else extraction.model_dump()
    flat = {"file_name": file_name}
    for section_key, section_val in d.items():
        if isinstance(section_val, dict):
            for inner_key, inner_val in section_val.items():
                flat[f"{section_key}__{inner_key}"] = inner_val
        else:
            flat[section_key] = section_val
    return flat


def build_master_parquet(json_dir: Path, parquet_path: Path) -> pd.DataFrame:
    rows = []
    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed JSON: {json_file}")
            continue

        if not data.get("success") or data.get("extraction") is None:
            rows.append({
                "file_name": data.get("file_name"),
                "extraction_success": False,
                "error": data.get("error"),
                "input_tokens": data.get("input_tokens", 0),
                "output_tokens": data.get("output_tokens", 0),
                "cost_usd": data.get("cost_usd", 0.0),
            })
            continue

        extraction = ILSAArticleMetadata(**data["extraction"])
        flat = flatten_extraction(extraction, data["file_name"])
        flat["extraction_success"] = True
        flat["error"] = None
        flat["input_tokens"] = data.get("input_tokens", 0)
        flat["output_tokens"] = data.get("output_tokens", 0)
        flat["cost_usd"] = data.get("cost_usd", 0.0)
        rows.append(flat)

    df = pd.DataFrame(rows)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    logger.info(f"Wrote {len(df)} rows to {parquet_path}")
    return df


# ---------------------------------------------------------------------------
# SQLite layer
# ---------------------------------------------------------------------------

def build_sqlite_database(parquet_path: Path, db_path: Path) -> None:
    json_dir = parquet_path.parent / "json"
    storage = StorageManager(str(db_path))

    inserted = 0
    skipped = 0

    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed JSON: {json_file.name}")
            skipped += 1
            continue

        if not data.get("success") or data.get("extraction") is None:
            skipped += 1
            continue

        try:
            from src.extractors.gpt_extractor import ExtractionResult as _ER

            extraction = ILSAArticleMetadata(**data["extraction"])
            result = _ER(
                file_name=data["file_name"],
                success=True,
                extraction=extraction,
                input_tokens=data.get("input_tokens", 0),
                output_tokens=data.get("output_tokens", 0),
                cost_usd=data.get("cost_usd", 0.0),
                duration_seconds=data.get("duration_seconds", 0.0),
            )
            storage.insert_article(result)
            inserted += 1
        except Exception as e:
            logger.warning(f"Skipping {json_file.name}: {e}")
            skipped += 1

    storage.close()
    logger.info(f"Built SQLite at {db_path}: {inserted} inserted, {skipped} skipped")


class StorageManager:
    """SQLite storage aligned with ILSAArticleMetadata (nested metadata + data)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _create_tables(self):
        """Create tables for flattened extraction fields."""
        self.cursor.executescript("""
        PRAGMA foreign_keys = ON;

        -- Bibliographic and extraction provenance
        CREATE TABLE IF NOT EXISTS metadata (
            file_name           TEXT PRIMARY KEY,
            title               TEXT,
            authors             TEXT,
            year                INTEGER,
            doi                 TEXT,
            venue               TEXT,
            publication_type    TEXT,
            source_category     TEXT,
            open_access         INTEGER,
            extraction_timestamp TEXT,
            extraction_cost_usd REAL,
            prompt_tokens       INTEGER,
            completion_tokens   INTEGER
        );

        -- Core extraction fields (one row per article)
        CREATE TABLE IF NOT EXISTS core_data (
            file_name                TEXT PRIMARY KEY,
            total_students           INTEGER,
            research_design_type     TEXT,
            student_weights_used     INTEGER,
            replicate_weights_used   INTEGER,
            weight_variable_name     TEXT,
            plausible_values_handling TEXT,
            missing_data_handling    TEXT,
            ml_technique_primary     TEXT,
            feature_selection        TEXT,
            baseline_model           TEXT,
            xai_method               TEXT,
            outcome_summary          TEXT,
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Junction: authors
        CREATE TABLE IF NOT EXISTS metadata_authors (
            file_name   TEXT NOT NULL,
            ordinal     INTEGER NOT NULL,
            author_name TEXT NOT NULL,
            PRIMARY KEY (file_name, ordinal),
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Junction: all ML techniques
        CREATE TABLE IF NOT EXISTS core_ml_techniques_all (
            file_name TEXT NOT NULL,
            ordinal   INTEGER NOT NULL,
            technique TEXT NOT NULL,
            PRIMARY KEY (file_name, ordinal),
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Junction: confounders
        CREATE TABLE IF NOT EXISTS core_confounders (
            file_name       TEXT NOT NULL,
            ordinal         INTEGER NOT NULL,
            confounder_name TEXT NOT NULL,
            PRIMARY KEY (file_name, ordinal),
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Junction: country-level sample sizes
        CREATE TABLE IF NOT EXISTS core_countries (
            file_name    TEXT NOT NULL,
            ordinal      INTEGER NOT NULL,
            country_code TEXT NOT NULL,
            n_students   INTEGER,
            PRIMARY KEY (file_name, ordinal),
            FOREIGN KEY (file_name) REFERENCES metadata (file_name) ON DELETE CASCADE
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_metadata_year
            ON metadata (year);
        CREATE INDEX IF NOT EXISTS idx_metadata_source_category
            ON metadata (source_category);
        CREATE INDEX IF NOT EXISTS idx_core_research_design
            ON core_data (research_design_type);
        CREATE INDEX IF NOT EXISTS idx_core_ml_technique
            ON core_data (ml_technique_primary);
        CREATE INDEX IF NOT EXISTS idx_core_pv_handling
            ON core_data (plausible_values_handling);
        CREATE INDEX IF NOT EXISTS idx_core_xai
            ON core_data (xai_method);
        """)
        self.conn.commit()

    @staticmethod
    def _bool_to_int(value: bool | None) -> int | None:
        """Map True->1, False->0, None->NULL."""
        return None if value is None else int(value)

    def _insert_junction(
        self, file_name: str, table: str, column: str, values: list[str]
    ) -> None:
        """Insert rows into a junction table."""
        for ordinal, value in enumerate(values):
            self.cursor.execute(
                f"INSERT OR IGNORE INTO {table} (file_name, ordinal, {column}) "
                f"VALUES (?, ?, ?)",
                (file_name, ordinal, str(value)),
            )

    def insert_article(self, result: ExtractionResult) -> None:
        """Persist a successful ExtractionResult as a single atomic transaction."""
        if not result.success or result.extraction is None:
            raise ValueError(
                f"Cannot insert failed extraction for '{result.file_name}'"
            )

        m = result.extraction.metadata
        file_name = m.file_name

        d = result.extraction.data
        survey = d.survey_design
        sample = d.sample_details
        ml = d.ml_techniques

        try:
            # metadata table
            self.cursor.execute("""
            INSERT OR REPLACE INTO metadata (
                file_name, title, authors, year, doi, venue,
                publication_type, source_category, open_access,
                extraction_timestamp, extraction_cost_usd,
                prompt_tokens, completion_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_name,
                m.title,
                json.dumps(m.authors or [], ensure_ascii=False),
                m.year,
                m.doi,
                m.venue,
                m.publication_type,
                m.source_category,
                self._bool_to_int(m.open_access),
                datetime.now(timezone.utc).isoformat(),
                result.cost_usd,
                result.input_tokens,
                result.output_tokens,
            ))

            # core_data table
            self.cursor.execute("""
            INSERT OR REPLACE INTO core_data (
                file_name, total_students, research_design_type,
                student_weights_used, replicate_weights_used,
                weight_variable_name, plausible_values_handling,
                missing_data_handling, ml_technique_primary,
                feature_selection, baseline_model, xai_method,
                outcome_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                file_name,
                sample.total_students if sample else None,
                d.research_design_type,
                self._bool_to_int(
                    survey.student_weights_used if survey else None
                ),
                self._bool_to_int(
                    survey.replicate_weights_used if survey else None
                ),
                survey.weight_variable_name if survey else None,
                d.plausible_values_handling,
                d.missing_data_handling,
                ml.primary if ml else None,
                None,
                None,
                None,
                d.outcome_summary,
            ))

            # junction tables
            self._insert_junction(
                file_name, "metadata_authors", "author_name",
                m.authors or [],
            )
            self._insert_junction(
                file_name, "core_ml_techniques_all", "technique",
                ml.all_techniques if ml else [],
            )
            self._insert_junction(
                file_name, "core_confounders", "confounder_name",
                d.confounders_identified or [],
            )

            # country-level samples
            countries = sample.countries if sample else []
            for ordinal, cs in enumerate(countries):
                self.cursor.execute(
                    "INSERT OR IGNORE INTO core_countries "
                    "(file_name, ordinal, country_code, n_students) "
                    "VALUES (?, ?, ?, ?)",
                    (file_name, ordinal, cs.country_code, cs.n_students),
                )

            self.conn.commit()
            logger.info(f"Inserted '{file_name}' (${result.cost_usd:.4f})")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Rolled back '{file_name}': {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Backward compatibility alias
ILSAStorage = StorageManager