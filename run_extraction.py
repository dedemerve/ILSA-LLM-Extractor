import os, sys, json, time, logging, fitz
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "ilsa_pipeline"))

from dotenv import load_dotenv
load_dotenv(Path("ilsa_pipeline/.env"))
from src.schemas.models import ILSAArticleMetadata
from openai import OpenAI

PDF_DIR = Path.home() / "Desktop" / "deneme"
OUTPUT_DIR = Path("outputs/batch_deneme")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL = "gpt-4o-2024-08-06"
MAX_CHARS = 400_000

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# SYSTEM_PROMPT'u dosyadan oku
code = Path("src/extractors/gpt_extractor.py").read_text()
t1 = code.find('SYSTEM_PROMPT')
t2 = code.find('"""', t1)
t3 = code.find('"""', t2 + 3)
SYSTEM_PROMPT = code[t2+3:t3].strip()

pdfs = sorted(PDF_DIR.glob("*.pdf"))
log.info(f"Found {len(pdfs)} PDFs")

total_cost = 0.0
for pdf_path in pdfs:
    fname = pdf_path.name
    out_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
    if out_path.exists():
        d = json.load(open(out_path))
        if d.get("extraction_success", True) != False:
            log.info(f"Skip: {fname}")
            continue

    log.info(f"Processing: {fname}")
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))
        doc.close()
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]

        t0 = time.time()
        resp = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract metadata from this article:\n\n{text}"},
            ],
            response_format=ILSAArticleMetadata,
        )
        elapsed = time.time() - t0
        u = resp.usage
        cost = (u.prompt_tokens/1e6)*2.50 + (u.completion_tokens/1e6)*10.00
        total_cost += cost

        result = resp.choices[0].message.parsed.model_dump(mode="json")
        result["_meta"] = {
            "file_name": fname, "model": MODEL,
            "input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
            "cost_usd": round(cost, 6), "elapsed_s": round(elapsed, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        log.info(f"  OK ${cost:.4f} | {u.prompt_tokens} in | {u.completion_tokens} out | {elapsed:.1f}s")
    except Exception as e:
        log.error(f"  FAIL: {e}")
        json.dump({"_meta":{"file_name":fname,"error":str(e)},"extraction_success":False},
                  open(out_path,"w",encoding="utf-8"), indent=2)

log.info(f"Total cost: ${total_cost:.4f}")

# --- Excel + Summary ---
import pandas as pd
results = []
for jf in sorted(OUTPUT_DIR.glob("*.json")):
    d = json.load(open(jf, encoding="utf-8"))
    if d.get("extraction_success", True) == False:
        continue
    meta = d.get("metadata", {})
    sd = d.get("survey_design", {})
    ml = d.get("ml_techniques", {})
    sample = d.get("sample_details", {})
    m = d.get("_meta", {})
    countries = sample.get("countries", [])
    cc = [c.get("country_code","") for c in countries] if isinstance(countries, list) else []
    conf = d.get("confounders_identified", [])
    results.append({
        "file_name": m.get("file_name",""),
        "title": meta.get("title",""),
        "year": meta.get("year"),
        "venue": meta.get("venue",""),
        "source_category": meta.get("source_category",""),
        "research_design_type": d.get("research_design_type",""),
        "pv_handling": d.get("plausible_values_handling",""),
        "missing_data": d.get("missing_data_handling",""),
        "student_weights": sd.get("student_weights_used"),
        "replicate_weights": sd.get("replicate_weights_used"),
        "weight_variable": sd.get("weight_variable_name",""),
        "total_students": sample.get("total_students"),
        "n_countries": len(cc),
        "country_codes": ", ".join(cc),
        "ml_primary": ml.get("primary",""),
        "ml_all": "; ".join(ml.get("all_techniques",[])),
        "feature_selection": ml.get("feature_selection",""),
        "baseline_model": ml.get("baseline_model",""),
        "xai_method": ml.get("xai_method",""),
        "n_confounders": len(conf) if isinstance(conf, list) else 0,
        "confounders": "; ".join(conf) if isinstance(conf, list) else "",
        "outcome_summary": d.get("outcome_summary",""),
        "cost": m.get("cost_usd",0),
        "tokens_in": m.get("input_tokens",0),
        "tokens_out": m.get("output_tokens",0),
    })

df = pd.DataFrame(results)
df.to_excel(str(OUTPUT_DIR / "ilsa_deneme_results.xlsx"), index=False, engine="openpyxl")
log.info(f"Excel: {OUTPUT_DIR}/ilsa_deneme_results.xlsx")

print("\n" + "="*80)
print("EXTRACTION SUMMARY — ALL FIELDS")
print("="*80)
for _, r in df.iterrows():
    print(f"\n{'─'*60}")
    print(f"FILE: {r['file_name'][:70]}")
    print(f"  title:                {r['title'][:80]}")
    print(f"  year:                 {r['year']}")
    print(f"  source_category:      {r['source_category']}")
    print(f"  research_design:      {r['research_design_type']}")
    print(f"  pv_handling:          {r['pv_handling']}")
    print(f"  missing_data:         {r['missing_data']}")
    print(f"  student_weights:      {r['student_weights']}")
    print(f"  replicate_weights:    {r['replicate_weights']}")
    print(f"  weight_variable:      {r['weight_variable']}")
    print(f"  total_students:       {r['total_students']}")
    print(f"  countries ({r['n_countries']}):        {r['country_codes']}")
    print(f"  ml_primary:           {r['ml_primary']}")
    print(f"  ml_all:               {r['ml_all'][:100]}")
    print(f"  feature_selection:    {r['feature_selection']}")
    print(f"  baseline_model:       {r['baseline_model']}")
    print(f"  xai_method:           {r['xai_method']}")
    print(f"  confounders ({r['n_confounders']}):     {r['confounders'][:120]}")
    print(f"  outcome_summary:      {r['outcome_summary'][:150]}...")
    print(f"  cost:                 ${r['cost']:.4f} ({r['tokens_in']} in / {r['tokens_out']} out)")

print(f"\n{'='*80}")
print(f"TOTAL: {len(df)} articles | ${df['cost'].sum():.4f} | {df['tokens_in'].sum()} tokens")
print(f"Output: {OUTPUT_DIR}/ilsa_deneme_results.xlsx")
print(f"{'='*80}")