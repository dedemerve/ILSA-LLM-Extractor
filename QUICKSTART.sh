#!/usr/bin/env bash

# ILSA Pipeline Refactored (v2.0) - Quick Start Guide
# ===================================================
# This script sets up and runs the production-ready refactored extraction pipeline.

set -e  # Exit on error

echo "=========================================="
echo "ILSA Pipeline v2.0 - Quick Start"
echo "=========================================="

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing Python dependencies..."
pip install -r requirements_refactored.txt

# Step 2: Verify API credentials
echo ""
echo "Step 2: Checking API credentials..."

if [ -z "$GOOGLE_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Please set API credentials:"
    echo ""
    echo "For Gemini (recommended for nano LLM):"
    echo "  export GOOGLE_API_KEY='your-gemini-api-key-here'"
    echo ""
    echo "Or for OpenAI (fallback):"
    echo "  export OPENAI_API_KEY='your-openai-api-key-here'"
    echo ""
    echo "Get API keys from:"
    echo "  - Gemini: https://makersuite.google.com/app/apikey"
    echo "  - OpenAI: https://platform.openai.com/account/api-keys"
    exit 1
fi

if [ -n "$GOOGLE_API_KEY" ]; then
    echo "✓ GOOGLE_API_KEY is set"
else
    echo "✓ OPENAI_API_KEY is set"
fi

# Step 3: Verify project structure
echo ""
echo "Step 3: Verifying project structure..."

if [ ! -d "data/raw_pdfs" ]; then
    echo "WARNING: data/raw_pdfs directory not found"
    echo "  Creating it now..."
    mkdir -p data/raw_pdfs
fi

if [ ! -f "src/schemas/article_schema_refactored.py" ]; then
    echo "ERROR: Schema file not found: src/schemas/article_schema_refactored.py"
    exit 1
fi

if [ ! -f "src/extractors/gpt_extractor_refactored.py" ]; then
    echo "ERROR: Extractor file not found: src/extractors/gpt_extractor_refactored.py"
    exit 1
fi

if [ ! -f "scripts/run_pipeline_refactored.py" ]; then
    echo "ERROR: Pipeline script not found: scripts/run_pipeline_refactored.py"
    exit 1
fi

echo "✓ All required files present"

# Step 4: Configure pipeline
echo ""
echo "Step 4: Configuring pipeline..."

# Set sensible defaults if not already set
export ILSA_MODEL="${ILSA_MODEL:-gemini-2.5-flash}"
export MAX_CONTEXT_PAGES="${MAX_CONTEXT_PAGES:-15}"
export ILSA_OUTPUT_DIR="${ILSA_OUTPUT_DIR:-./outputs}"
export ILSA_SKIP_PROCESSED="${ILSA_SKIP_PROCESSED:-true}"
export ILSA_DRY_RUN="${ILSA_DRY_RUN:-false}"

echo "Model:           $ILSA_MODEL"
echo "Max pages:       $MAX_CONTEXT_PAGES"
echo "Output dir:      $ILSA_OUTPUT_DIR"
echo "Skip processed:  $ILSA_SKIP_PROCESSED"
echo "Dry run:         $ILSA_DRY_RUN"

# Step 5: Run dry-run first (optional)
read -p "Run dry-run first to verify setup? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running dry-run..."
    export ILSA_DRY_RUN=true
    python scripts/run_pipeline_refactored.py
    
    read -p "Proceed with actual extraction? (y/n) " -n 1 -r
    echo
    if ! [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    export ILSA_DRY_RUN=false
fi

# Step 6: Run pipeline
echo ""
echo "Starting extraction pipeline..."
echo "This will process all PDFs in: data/raw_pdfs"
echo "Results will be saved to:      $ILSA_OUTPUT_DIR"
echo ""

python scripts/run_pipeline_refactored.py

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  JSON extractions: $ILSA_OUTPUT_DIR/json/"
echo "  Logs:             $ILSA_OUTPUT_DIR/logs/"
echo ""
echo "Next steps:"
echo "  1. Review quality flags: check for needs_human_review=true"
echo "  2. Validate quotes: ensure _quote fields match extracted values"
echo "  3. Compare against manual sample: pick 5-10 papers to manually verify"
echo "  4. Monitor accuracy: track research_design_type, missing_data_handling, plausible_values_handling"
echo ""
