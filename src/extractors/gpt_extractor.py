import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from pydantic import ValidationError

from src.schemas import ILSAArticleMetadata

if TYPE_CHECKING:
    from src.extractors.pdf_processor import ProcessedPDF

logger = logging.getLogger(__name__)

COUNTRY_NAME_TO_ISO = {
    "turkey": "TUR", "usa": "USA", "united states": "USA",
    "germany": "DEU", "deutschland": "DEU", "france": "FRA",
    "japan": "JPN", "korea": "KOR", "south korea": "KOR",
    "china": "CHN", "brazil": "BRA", "finland": "FIN",
    "singapore": "SGP", "australia": "AUS", "canada": "CAN",
    "uk": "GBR", "united kingdom": "GBR", "england": "GBR",
    "spain": "ESP", "italy": "ITA", "netherlands": "NLD",
    "sweden": "SWE", "norway": "NOR", "denmark": "DNK",
    "israel": "ISR", "new zealand": "NZL", "ireland": "IRL",
    "austria": "AUT", "belgium": "BEL", "switzerland": "CHE",
    "portugal": "PRT", "poland": "POL", "czech republic": "CZE",
    "hungary": "HUN", "greece": "GRC", "romania": "ROU",
    "russia": "RUS", "thailand": "THA", "indonesia": "IDN",
    "malaysia": "MYS", "chile": "CHL", "mexico": "MEX",
    "colombia": "COL", "argentina": "ARG", "india": "IND",
    "south africa": "ZAF", "taiwan": "TWN", "hong kong": "HKG",
    "macao": "MAC", "macau": "MAC", "estonia": "EST",
    "latvia": "LVA", "lithuania": "LTU", "slovakia": "SVK",
    "slovenia": "SVN", "croatia": "HRV", "serbia": "SRB",
    "bulgaria": "BGR", "cyprus": "CYP", "malta": "MLT",
    "luxembourg": "LUX", "iceland": "ISL", "qatar": "QAT",
    "uae": "ARE", "saudi arabia": "SAU", "jordan": "JOR",
    "iran": "IRN", "egypt": "EGY", "morocco": "MAR",
    "tunisia": "TUN", "ghana": "GHA", "kenya": "KEN",
    "nigeria": "NGA", "pakistan": "PAK", "vietnam": "VNM",
    "philippines": "PHL", "peru": "PER", "uruguay": "URY",
    "costa rica": "CRI", "panama": "PAN",
    "lebanon": "LBN", "dominican republic": "DOM",
    "chinese taipei": "TWN", "beijing-shanghai-jiangsu-zhejiang": "CHN",
    "b-s-j-z": "CHN", "bsjz": "CHN", "b-s-j-g": "CHN",
    "northern ireland": "GBR", "türkiye": "TUR",
    "republic of korea": "KOR", "czechia": "CZE",
    "macao sar": "MAC", "hong kong sar": "HKG",
    "chinese mainland": "CHN", "united arab emirates": "ARE",
    "scotland": "GBR", "wales": "GBR", "great britain": "GBR",
    "flemish": "BEL", "flemish community": "BEL",
    "philippine": "PHL", "filipino": "PHL",
    "korean": "KOR", "moroccan": "MAR",
    "tunisian": "TUN", "b-s-j-z (china)": "CHN",
    "beijing, shanghai, jiangsu, and zhejiang": "CHN",
    "beijing, shanghai, jiangsu, and guangdong": "CHN",
    "b-s-j-g (china)": "CHN",
    "lebanese": "LBN", "lebanese republic": "LBN",
    "brazilian": "BRA", "spanish": "ESP",
    "german": "DEU", "french": "FRA",
    "japanese": "JPN", "finnish": "FIN",
    "australian": "AUS", "canadian": "CAN",
    "irish": "IRL", "swedish": "SWE",
    "norwegian": "NOR", "danish": "DNK",
    "estonian": "EST", "latvian": "LVA",
    "hungarian": "HUN", "peruvian": "PER",
    "mexican": "MEX", "chilean": "CHL",
    "colombian": "COL", "uruguayan": "URY",
    "singaporean": "SGP", "dutch": "NLD",
    "swiss": "CHE", "belgian": "BEL",
    "polish": "POL", "austrian": "AUT",
    "greek": "GRC", "slovenian": "SVN",
    "italian": "ITA", "portuguese": "PRT",
    "luxembourgish": "LUX", "icelandic": "ISL",
    "qatari": "QAT", "emirati": "ARE",
    "saudi": "SAU", "jordanian": "JOR",
    "iranian": "IRN", "egyptian": "EGY",
    "ghanaian": "GHA", "kenyan": "KEN",
    "nigerian": "NGA", "pakistani": "PAK",
    "vietnamese": "VNM", "thai": "THA",
    "indonesian": "IDN", "indian": "IND",
    "taiwanese": "TWN", "macanese": "MAC",
    "new zealander": "NZL", "british": "GBR",
    "american": "USA", "chinese": "CHN",
    "turkish": "TUR", "czech": "CZE",
    "slovak": "SVK", "croatian": "HRV",
    "serbian": "SRB", "bulgarian": "BGR",
    "cypriot": "CYP", "maltese": "MLT",
    "romanian": "ROU", "russian": "RUS",
    "south african": "ZAF", "panamanian": "PAN",
    "dominican": "DOM", "israeli": "ISR",
    "scandinavian": "NOR",
    "malaysian": "MYS", "lithuanian": "LTU",
    "argentinian": "ARG", "argentine": "ARG",
    "costa rican": "CRI",
    "the netherlands": "NLD",
    "republic of china": "TWN",
    "korea, republic of": "KOR",
}

MODEL_NAME = "gpt-5.4-nano"
PRICE_INPUT_PER_1M = 2.50
PRICE_OUTPUT_PER_1M = 10.00

SYSTEM_PROMPT = """You are an expert research analyst specializing in International \
Large-Scale Assessments (ILSA: PISA, TIMSS, PIRLS, TALIS, ICILS, ICCS, PIAAC) \
and related national/regional large-scale assessments (NAEP, CEDRE, INVALSI) \
and the intersection of Machine Learning in educational data mining.

Your task is to extract a highly detailed, structured metadata and methodological \
sheet from an academic article. You must rigorously map academic jargon to the \
strict schema provided, and MINIMIZE null values through deep semantic search \
and expert domain inference.

═══════════════════════════════════════════════════════════════
CRITICAL EXTRACTION & INFERENCE RULES
═══════════════════════════════════════════════════════════════

1) STRICT ENUMERATIONS & CATEGORIES:
   - publication_type MUST be exactly one of:
     ['journal', 'conference', 'book_chapter', 'preprint', 'report', 'thesis'].
   - source_category MUST be exactly one of:
     ['technical_report', 'review_article', 'methodology_paper', 'peer_reviewed_research'].
   - research_design_type MUST be exactly one of:
     ['predictive', 'causal_observational', 'causal_experimental', 'exploratory'].
     Mapping: prediction/classification/regression → "predictive";
     causal forests, propensity scores, diff-in-diff, IV → "causal_observational";
     randomized experiment → "causal_experimental";
     clustering, topic modeling, EDA, data description → "exploratory".
   - plausible_values_handling MUST be exactly one of:
     ['rubin_rules', 'single_pv', 'average_pv', 'mitml', 'not_applicable', 'not_reported'].
     Synonym table:
       "Rubin's rules" / "Rubin combining rules" / "combined PV estimates" /
       "pooled across PVs" / "PV estimates combined"               → rubin_rules
       "first plausible value" / "PV1 only" / "PV1MATH" / "PV1READ" /
       "PV1SCIE" / "PV2SCIE" / "single PV draw" /
       "one PV per student" / "separate analyses per PV" /
       "PV1 as outcome" / "used PV2SCIE" / "used PV1READ" /
       "target indicator from one PV" /
       "binary variable from PV benchmarks (single draw)"         → single_pv
       "averaged plausible values" / "mean of PVs" / "PV average" /
       "all five PVs averaged" / "all ten PVs averaged" /
       "PV1MATH–PV10MATH averaged" / "BSSSCI01–BSSSCI05 averaged" /
       "mean of 10 plausible values" / "average of plausible values" /
       "average of PV1MATH through PV10MATH" /
       "averaged across all plausible values"                      → average_pv
       "mitml" / "Mplus complex survey" / "multilevel MI"          → mitml
       TALIS/PIAAC without PVs, or DV is Likert/direct measure    → not_applicable
       "WLE" / "Warm's WLE" / "weighted likelihood estimator" /
       "IRT ability estimates" / "theta estimates" / "EAP estimates" /
       "latent trait scores" / "CFA-based scores" /
       "scale scores (not PVs)"                                    → not_applicable
       DV is binary classification (correct/incorrect, high/low) /
       DV is process data (actions, response times) /
       DV is affective/attitudinal (life satisfaction, self-efficacy) /
       DV is curriculum-based (not ILSA achievement)               → not_applicable
    SOFTWARE-BASED PV INFERENCE — when the paper does not state PV handling
    explicitly, infer from the software / R packages mentioned:
      "bifiesurvey" / "repest" / "intsvy" / "EdSurvey" /
      "IEA IDB Analyzer" / "RALSA" / "lavaan.survey" /
      "WeMix" / "mitml"                                              → rubin_rules
      These packages implement Rubin's combining rules internally;
      their use is strong evidence that PVs were handled properly.
      "five plausible values" / "5 PVs" / "all five PVs" /
      "five_pv" / "5_pv" / "ten plausible values" / "10 PVs" /
      "all ten PVs" / "PV1–PV5" / "PV1–PV10" /
      "analyses repeated across PVs and pooled"                      → rubin_rules
    ILSA domain default: PISA/TIMSS/PIRLS always ship PVs for achievement scores.
    If the paper models achievement and never mentions PV handling → average_pv.
    If the paper dichotomizes achievement into binary (e.g. "proficient
    vs not") using PV benchmarks, it still used PVs → infer from context.
   - missing_data_handling MUST be exactly one of:
     ['listwise_deletion', 'pairwise_deletion', 'mean_imputation',
      'multiple_imputation', 'not_reported'].
     Synonym table:
       "listwise deletion" / "complete case" / "excluded incomplete" /
       "removed cases with missing" / "cases with missing data were
        removed" / "after exclusion of missing"                    → listwise_deletion
       "pairwise deletion" / "available case analysis"             → pairwise_deletion
       "mean substitution" / "mean replacement" / "imputed with mean" /
       "series mean" / "mode imputation" / "median imputation" /
       "SimpleImputer (mode)" / "SimpleImputer (median)" /
       "substituted mode values" / "imputed with median"           → mean_imputation
       "MICE" / "MI" / "missForest" / "missRanger" / "FIML" /
       "EM algorithm" / "expectation maximization" /
       "chained equations" / "hot-deck" /
       "kNN imputation" / "k-nearest neighbor imputation" /
       "predictive mean matching" / "PMM" /
       "MCMC imputation" / "Markov Chain Monte Carlo" /
       "two-level FCS" / "fully conditional specification" /
       "multivariate imputation" / "RF-based imputation" /
       "rblimp" / "blimp" / "Bayesian imputation" /
       "stochastic regression imputation" /
       any ML-based imputation method                              → multiple_imputation
       "zero imputation" / "zero fill" / "replaced with zero" /
       "filled with zero" / "imputed with zero"                    → mean_imputation
     CAUTION: SMOTE / oversampling / undersampling / data augmentation /
     SMOTETomek / ADASYN / CTGAN / VAE-augmentation are class-balancing
     or synthetic-data techniques, NOT missing data handling.
     Do NOT map them to any missing_data_handling value.
     CAUTION: "winsorized" / "trimmed at percentile" are outlier-treatment
     techniques, NOT missing data handling. Do NOT map them either.

2) COUNTRY CODES (ISO 3166-1 alpha-3):
   - country_code MUST always be a 3-letter UPPERCASE ISO code (e.g. 'TUR', 'USA', \
     'DEU', 'GBR', 'FRA', 'JPN', 'KOR', 'CHN', 'BRA', 'FIN', 'SGP', 'AUS').
   - NEVER write full country names, 2-letter codes, or non-standard abbreviations.
   - SPECIAL ECONOMIES & REGIONS — use these mappings:
     "Beijing-Shanghai-Jiangsu-Zhejiang" / "B-S-J-Z" / "B-S-J-G"
       / "BSJZ" / "Chinese mainland"                              → CHN
     "Chinese Taipei" / "Taiwan"                                   → TWN
     "Hong Kong" / "Hong Kong SAR"                                 → HKG
     "Macao" / "Macau" / "Macao SAR"                               → MAC
     "England" / "Northern Ireland"                                → GBR
     "Republic of Korea" / "South Korea" / "Korea"                 → KOR
     "Türkiye" / "Turkey"                                          → TUR
     "United Arab Emirates" / "UAE"                                → ARE
     "Dominican Republic"                                          → DOM
     "Czech Republic" / "Czechia"                                  → CZE
     "The Netherlands" / "Netherlands"                             → NLD
   - If a study covers 37+ OECD countries or 44+ TIMSS countries or 79+ PISA \
     countries, list ALL countries found in the text or tables. If only a \
     count is given (e.g. "80 countries"), extract every country explicitly \
     named in the manuscript and set n_students to null for unnamed ones. \
     Do NOT leave the countries list empty when the paper clearly analyzed \
     specific nations.

3) ML vs. TRADITIONAL STATISTICS (critical for ml_techniques):
   - For 'all_techniques' and 'primary', extract ONLY Machine Learning / \
     predictive-modeling algorithms. See the COMPREHENSIVE mapping table below.
   - DO NOT include traditional / descriptive / psychometric methods:
     PCA, factor analysis, t-tests, ANOVA, ANCOVA, MANOVA, chi-square,
     basic correlations, descriptive statistics, EFA/CFA, SEM, HLM/mixed-effects
     (unless explicitly used as an ML baseline), ESCS index computations,
     Latent Profile Analysis (LPA), Latent Class Analysis (LCA),
     IRT models, Rasch models, Partial Credit Model, measurement invariance,
     Interpretive Structural Modeling (ISM), bibliometric analysis,
     Shapley value decomposition (standalone; report SHAP only under XAI),
     DBSCAN / k-means / k-medoids / hierarchical clustering /
     Gaussian Mixture Model (GMM) (ONLY if used purely for unsupervised
     exploration without any predictive goal — if combined with a
     prediction pipeline, include the supervised learner, not the
     clustering step).
     Process Mining (Disco, ProM, fuzzy miner) — visualization/discovery
     tools, NOT ML algorithms.
     Finite Mixture Models / Latent Transition Analysis — psychometric
     mixture models, NOT ML.
   - Latent Profile Analysis (LPA) and Latent Class Analysis (LCA) are
     ALWAYS psychometric / mixture modeling methods and NEVER ml_techniques,
     even when used in process-data papers to identify behavioral profiles.
     The same applies to Confirmatory Factor Analysis (CFA), measurement
     invariance testing, and Hierarchical Linear Modeling (HLM) — these
     are statistical methods, not ML.
   - DO NOT include psychometric Diagnostic Classification Models (DCMs):
     HO-DINA, HO-GDINA, DINO, ACDM, LCDM, G-DINA — these are
     psychometric measurement models, NOT machine learning. Only include
     them if the paper explicitly frames them as ML classifiers.
   - DO NOT include Structural Topic Modeling (STM) UNLESS the paper
     uses STM output as features for a supervised prediction task.
     When STM is used solely for exploratory text analysis on abstracts
     or corpora (e.g., in review papers), it is NOT an ML technique.
   - DATA AUGMENTATION methods (SMOTE, CTGAN, VAE-based augmentation)
     are preprocessing steps, NOT ml_techniques. Mention them in
     missing_data_handling or confounders_identified if relevant,
     but never list them as primary or all_techniques entries.
   - COMPREHENSIVE algorithm name mapping (use canonical short names):
     ── TREE & ENSEMBLE ──
     "gradient boosted trees" / "GBT" / "GBM" / "GBDT"    → Gradient Boosting
     "XGBoost" / "extreme gradient boosting" / "XGB"       → XGBoost
     "LightGBM" / "Light GBM" / "LGBM" / "light gradient
      boosting"                                            → LightGBM
     "CatBoost" / "category boosting"                      → CatBoost
     "Histogram GBR" / "HGB" / "HistGradientBoosting"      → Histogram GBR
     "random forests" / "RF"                               → Random Forest
     "Extra Trees" / "ExtraTrees" / "extremely randomized
      trees" / "ET"                                        → Extra Trees
     "AdaBoost" / "adaptive boosting"                      → AdaBoost
     "Decision Tree" / "CART" / "C5.0" / "J48" /
      "classification tree" / "regression tree"            → Decision Tree
     "stacking" / "stacked ensemble" / "meta-model" /
      "stacked generalization"                             → Stacking
     "blending" / "blend"                                  → Blending
     "bagging" / "bootstrap aggregation"                   → Bagging
     "Conditional Inference Trees" / "CIT" / "ctree"       → Conditional Inference Trees
     "Conditional Inference Forests" / "CIF" / "cforest"   → Conditional Inference Forests
     "Boruta" (wraps RF for feature selection)             → Random Forest
     ── LINEAR / REGULARIZED ──
     "LASSO" / "L1 regression" / "glmnet L1"               → LASSO
     "Ridge Regression" / "L2 regression"                  → Ridge Regression
     "Elastic Net" / "L1+L2" / "glmnet" / "Enet"          → Elastic Net
     "Group Mnet" / "group MCP" / "group penalized"        → Group Mnet
     "Logistic Regression" (classification only)           → Logistic Regression
     "Linear Regression" / "MLR" (prediction/baseline)     → Linear Regression
     ── SVM / INSTANCE-BASED ──
     "SVM" / "SVC" / "SVR" / "support vector"              → SVM
     "k-NN" / "KNN" / "k-nearest neighbor"                 → k-NN
     ── PROBABILISTIC ──
     "Naive Bayes" / "GNB" / "NB" / "Gaussian Naive Bayes" → Naive Bayes
     "Bayesian Ridge" / "ARD"                              → Bayesian Ridge
     ── NEURAL NETWORKS & DEEP LEARNING ──
     "ANN" / "MLP" / "deep learning" / "feed-forward NN" /
      "multilayer perceptron"                              → Neural Network
     "LSTM" / "Long Short-Term Memory"                     → LSTM
     "GRU" / "Gated Recurrent Units"                       → GRU
     "CNN" / "Convolutional Neural Network"                → CNN
     "Autoencoder" / "variational autoencoder" / "VAE"     → Autoencoder
     "RNN" / "recurrent neural network"                    → RNN
     "Elman neural network" / "Jordan neural network"      → Neural Network
     ── CAUSAL ML ──
     "BART" / "Bayesian Additive Regression Trees"         → BART
     "BCF" / "Bayesian Causal Forests"                     → BCF
    ── FUZZY / HYBRID ──
    "ANFIS" / "neuro-fuzzy" / "adaptive neuro-fuzzy"      → ANFIS
    ── BAYESIAN ML ──
    "Bayesian Network" / "BN" / "Bayesian classifier" /
     "Bayesian belief network" / "directed acyclic graph
      classifier"                                         → Bayesian Network
    ── PENALIZED MULTILEVEL ──
    "glmmLasso" / "GLMM + LASSO" / "penalized GLMM" /
     "penalized mixed model"                              → glmmLasso
    "blackboost" / "conditional gradient boosting" /
     "mboost" / "model-based boosting"                    → Gradient Boosting
    ── NLP-BASED (when combined with supervised prediction) ──
    "Word2Vec + classifier" / "TF-IDF + classifier" /
     "Doc2Vec + classifier"                               → report the CLASSIFIER
    "RoBERTa" / "BERT" (for scoring/classification)       → report the architecture
    "Bag-of-Words + ANN" / "BoW + Neural Network"         → Neural Network
    ── OTHER CLASSIFIERS ──
    "Discriminant Analysis" / "LDA" / "QDA" /
     "linear discriminant analysis"                       → Discriminant Analysis
    "Gaussian Process" / "GP regression" / "GP classifier" → Gaussian Process
    ── KNOWLEDGE TRACING ──
    "DKT" / "Deep Knowledge Tracing"                       → Deep Knowledge Tracing
    ── SEMI-SUPERVISED / ACTIVE ──
    "active learning" / "semi-supervised learning" /
     "self-training" / "co-training" (when combined with
     a base classifier for label propagation)             → report the base classifier

4) WEIGHTING & REPLICATE DESIGN LOGIC:
   - student_weights_used: set true if ANY of these appear:
     "student weights", "sampling weights", "survey weights",
     "W_FSTUWT", "TOTWGT", "SCHWGT", "HOUWGT", "SENWGT", "MATWGT",
     "SCIWGT", "REAWGT", any variable starting with "W_" or ending in "WGT",
     "final weight", "senate weight", "house weight", "overall weight",
     "analysis weight", "probability weight", "design weight",
     "weighted estimation", "weighted analysis", "weighted mean",
     "adjusting for complex survey design", "adjusting for stratification",
     "adjusting for clustering", "multilevel weighting",
     "population-representative" + mention of weight application.
   - SOFTWARE-BASED WEIGHT INFERENCE: If the paper mentions using any of \
     these weight-aware tools, infer student_weights_used = true unless \
     explicitly contradicted:
     "IEA IDB Analyzer", "IDB Analyzer", "bifiesurvey", "BIFIEsurvey",
     "WeMix", "lavaan.survey", "survey package" (in R), "svy:" (Stata),
     "RALSA", "intsvy", "EdSurvey", "repest".
   - replicate_weights_used: set true if "BRR", "balanced repeated replication", \
     "Fay's method", "jackknife", "JK2", "JRR", "JK1", "replicate weights", \
     "jackknife repeated replication", "Taylor series linearization" appear.
   - weight_variable_name: exact variable name string if mentioned (e.g. 'W_FSTUWT').
   - ILSA domain default: if a study uses ILSA micro-data and never discusses \
     weights → student_weights_used = false (omission = likely unweighted).
   - EXPLICIT NON-USE PATTERN: Many ML-focused studies deliberately ignore \
     survey weights because ML algorithms (RF, XGBoost, SVM, Neural Networks) \
     do not natively support survey weights. If the paper uses ML models \
     on ILSA data and never mentions weights, set student_weights_used = false \
     and FILL weight_fields_interpretation explaining: "The study applied ML \
     algorithms that do not natively incorporate survey weights. The manuscript \
     does not discuss weighting, suggesting an unweighted analysis."
   - weight_fields_interpretation: FILL ONLY IF student_weights_used, \
     replicate_weights_used, AND weight_variable_name are ALL null or false. \
     Write 3-4 analytical sentences: what the manuscript says about sample design, \
     why weights might be missing (e.g. ML study without weight support, small \
     convenience sample, secondary analysis without original weights, process-data \
     study focused on behavioral sequences rather than population estimation), \
     and what exact wording would be needed to extract them. \
     IF ANY weight field is non-null (true or has a variable name), \
     this field MUST BE null.

5) NULL FIELDS INTERPRETATION (THE FALLBACK):
   - null_fields_interpretation: trigger ONLY if the overall extraction is \
     extremely sparse — e.g. missing sample sizes, missing ML algorithms, missing \
     PV handling, multiple metadata fields null. Write a structured diagnostic note \
     (plain text) explaining WHY the paper lacks data (e.g. "This is a theoretical \
     review paper, hence no sample size or ML models are evaluated." or "The \
     manuscript is a meta-analysis without original ILSA micro-data analysis.").
   - If the record is reasonably dense (most fields filled), this MUST BE null.

6) EXHAUSTIVE DATA & METHODOLOGY SEARCH (NO LAZY EXTRACTIONS):
   - DataBlock and SurveyDesign are the MOST CRITICAL sections. You must \
     aggressively scan "Methodology", "Data", "Measures", "Analytical Strategy", \
     "Sample", "Participants", "Data Processing", "Data Preprocessing", \
     "Data Cleaning", AND footnotes, table notes, and appendices.
   - EXTENDED WEIGHT SYNONYMS — also look for: "senate weights", "house weights", \
     "overall weights", "SENWGT", "MATWGT", "SCIWGT", "REAWGT", variables starting \
     with "W_" or ending in "WGT". For replicate weights also: "JRR", "jackknife \
     repeated replication", "Taylor series linearization".
   - SOFTWARE-BASED INFERENCE — if the paper mentions using IEA IDB Analyzer, \
     bifiesurvey, WeMix, lavaan.survey, EdSurvey, RALSA, intsvy, repest, or \
     any ILSA-specific analysis tool, these tools inherently apply survey weights \
     → infer student_weights_used = true.
   - INFERRING COMPLEX DESIGN — if the authors mention adjusting for "complex \
     survey design", "stratification", "clustering", or "multilevel weighting", \
     you MUST infer student_weights_used = true.
   - ML-SPECIFIC PATTERN — Many ML studies (RF, XGBoost, SVM, NN) on ILSA data \
     deliberately omit survey weights because these algorithms lack native weight \
     support. If the paper uses ML without mentioning weights, set \
     student_weights_used = false and weight_fields_interpretation must explain \
     this ML-specific omission pattern.
   - REVIEW / NON-EMPIRICAL PAPERS — If the paper is a systematic review, \
     bibliometric analysis, or theoretical framework without original ILSA \
     micro-data analysis, set student_weights_used = null, replicate_weights_used \
     = null, weight_variable_name = null, and explain in weight_fields_interpretation.
   - STRICT FAIL-SAFE ENFORCEMENT — if BOTH student_weights_used and \
     replicate_weights_used end up as false or null, AND weight_variable_name \
     is null, you ABSOLUTELY MUST fill weight_fields_interpretation with 3-4 \
     sentences explaining: (a) what the methodology section says about the data, \
     (b) why weight information is missing (e.g. "The authors focused solely on \
     the ML architecture without detailing data preparation or ILSA weighting"), \
     (c) what explicit wording or variable names would be needed to confirm \
     weights were used.
   - *** FATAL ERROR ***: returning false/null for all weight fields AND leaving \
     weight_fields_interpretation as null is a schema violation. You must always \
     provide either evidence of weighting OR an explanation of its absence.

7) AGGRESSIVE SAMPLE & COUNTRY EXTRACTION (sample_details):
   - total_students: NEVER default to null without an exhaustive search. Scan \
     "Method", "Participants", "Data", "Data Cleaning", and "Results" sections \
     for keywords: "N =", "n =", "final sample", "consisted of", "analytic \
     sample", "valid responses", "after removing", "after exclusion", \
     "remaining students", "total of". Check tables and figure captions too.
   - countries: identify ALL countries or economies analyzed. If the abstract \
     says "using PISA data from the USA", extract country_code = "USA". If a \
     table lists multiple countries, extract ALL of them with ISO 3166-1 alpha-3 \
     codes. Do not leave the list empty if the data source inherently implies \
     a country (e.g. "TIMSS 2019 data from Morocco" → [{"country_code":"MAR"}]).

8) LOGICAL DEDUCTION FOR ML TECHNIQUES (ml_techniques):
   - primary: DO NOT leave primary null if all_techniques is populated!
     a) If ONLY ONE algorithm is in all_techniques (e.g. ["LASSO"]), that \
        algorithm IS inherently the primary model — copy it to primary.
     b) If MULTIPLE algorithms are listed, scan "Results", "Abstract", or \
        "Conclusion" for: "performed best", "achieved the highest accuracy", \
        "outperformed", "best-performing model", "highest R²/AUC/F1". Assign \
        that winning model to primary.
     c) If the paper genuinely compares models without declaring a winner, \
        pick the one highlighted in the abstract or conclusion.
   - *** FATAL ERROR ***: primary left null while all_techniques has values \
     is a schema violation.

9) ENFORCING THE NULL INTERPRETATION FALLBACK:
   - If after exhaustive search total_students is still null, OR primary is \
     null while all_techniques is empty, you MUST trigger null_fields_interpretation. \
     Write 2-3 sentences diagnosing the omission (e.g. "The study is a scoping \
     review without an empirical sample" or "The authors listed LASSO and Random \
     Forest but did not report which model achieved the best metric.").
   - This rule complements Rule 5 — both may apply simultaneously.

10) RESEARCH DESIGN CLASSIFICATION (research_design_type):
   - Papers that predict/classify student outcomes using ML → "predictive"
   - Papers using causal ML (BART, BCF, propensity score matching, \
     diff-in-diff, instrumental variables, causal forests) → "causal_observational"
   - Papers with randomized experiments / RCTs → "causal_experimental"
   - Papers using unsupervised methods ONLY (clustering, LPA, topic modeling, \
     bibliometric analysis, process mining) without a prediction target → "exploratory"
   - Systematic reviews / meta-analyses / theoretical frameworks / \
     methodological papers → "exploratory"
   - Papers that combine prediction AND clustering (e.g. cluster then predict) \
     → "predictive" (the supervised component dominates)
   - Process-data papers that classify engagement or strategy profiles → "predictive"

11) ANTI-HALLUCINATION:
   - Never INVENT: DOIs, exact N, country codes, author names, weight variable \
     names, or algorithm names not present in the text.
   - Inference is allowed ONLY for categorical/boolean/enum fields where ILSA \
     domain knowledge provides a clear default (rules 1 and 4 above).
   - Numeric fields (total_students, n_students, year) MUST come from the text.

12) NATIONAL / REGIONAL LARGE-SCALE ASSESSMENTS AND OTHER ILSAs:
   - Papers using NAEP (USA), CEDRE (France), INVALSI (Italy), or other
     national LSAs should be treated with the SAME extraction rigor as
     ILSA papers. They are valid data sources for this pipeline.
   - NAEP uses plausible values → apply PV handling rules as for PISA/TIMSS.
   - CEDRE and INVALSI may use IRT-based scores rather than PVs → check
     methodology; if WLE or theta scores are used → plausible_values_handling
     = "not_applicable"; if PVs are generated → apply standard PV rules.
   - PIAAC (Programme for the International Assessment of Adult Competencies):
     Uses plausible values for literacy, numeracy, and PS-TRE domains.
     Apply the SAME PV handling inference rules as PISA/TIMSS.
     Process data from PIAAC PS-TRE items (log files, action sequences)
     → plausible_values_handling = "not_applicable" for the process
     component; "rubin_rules" or "average_pv" for the achievement component.
   - ICILS (International Computer and Information Literacy Study):
     Uses plausible values for CIL and CT scores → apply standard PV rules.
   - ICCS (International Civic and Citizenship Education Study):
     Uses plausible values for civic knowledge → apply standard PV rules.
     Engagement/attitude scales are IRT-scaled but NOT PVs →
     plausible_values_handling = "not_applicable" when the study models
     only attitudes/engagement without civic knowledge scores.
   - PISA-VET (OECD's vocational assessment, in development):
     Treat as framework/assessment-design paper unless it reports
     empirical student data; typically non-empirical at this stage.
   - For PSLC DataShop, LUCA simulations, licensure examinations,
     professional certification tests, or other digital learning
     platforms → treat as non-ILSA empirical data; PV handling is
     typically "not_applicable".

13) PROCESS DATA PAPERS (log files, clickstreams, response times, action sequences):
   - These papers analyze HOW students interact with computer-based assessments \
     rather than traditional achievement scores.
   - Typical data: action sequences, response times, mouse clicks, keystrokes, \
     navigation paths, time-on-task variables, VOTAT strategies, N-gram features, \
     directed graph features, network statistics (centralization, density, \
     flow hierarchy), time-to-first-action, number of visits / short visits, \
     Differential Response Time (DRT), Response Time Effort (RTE), \
     behavioral effort indicators, action sequence embeddings (Word2Vec, \
     Doc2Vec on action logs), LCS-based sequence similarity measures.
   - Process analysis tools (NOT ml_techniques): Process Mining (Disco, ProM), \
     action sequence autoencoders (when used purely for representation learning \
     without a downstream prediction task), and profiling via LPA/LCA.
   - plausible_values_handling: usually "not_applicable" because process data \
     studies typically use binary correctness (correct/incorrect), IRT-based \
     ability estimates (EAP, WLE, theta), or behavioral indicators rather \
     than PVs.
     EXCEPTION: when a process data paper ALSO models achievement scores \
     from PVs (e.g., predicting science PV-based performance from behavioral \
     effort), extract PV handling for the achievement component normally \
     (rubin_rules / average_pv / single_pv) and note "not_applicable" only \
     if the DV is purely process-based.
   - student_weights_used: usually false or null — process data studies focus \
     on behavioral patterns, not population-representative estimation.
   - ml_techniques: include ALL ML algorithms used for classification/prediction \
     of process outcomes. Common ones: Random Forest, LSTM, GRU, CNN, SVM, \
     Autoencoder, k-means (if combined with prediction), Neural Network, \
     XGBoost, Gradient Boosting, Logistic Regression, HMM (when used for \
     prediction, NOT when used purely as a psychometric measurement model).
   - DO NOT exclude algorithms just because the DV is process-based rather \
     than achievement-based. Any supervised/semi-supervised learner counts.
   - DO NOT include as ml_techniques: Diagnostic Classification Models \
     (HO-DINA, GDINA, DINO, ACDM), Partial Credit Models, IRT models, \
     Latent Profile Analysis, Process Mining software, or cluster editing \
     algorithms used purely for grouping.
   - research_design_type: "predictive" if classifying engagement/performance \
     from process features; "exploratory" if only clustering/profiling without \
     a prediction target.
   - Capture process-specific features (response time, action counts, \
     time-to-first-action, number of visits, VOTAT score, preparation \
     time, execution time) in independent variables or confounders_identified.

14) REVIEW / META-ANALYSIS / BIBLIOMETRIC PAPERS:
   - These papers synthesize existing literature rather than analyzing ILSA \
     micro-data directly.
   - source_category: "review_article" (systematic review, scoping review, \
     meta-analysis, bibliometric analysis, literature survey).
   - research_design_type: "exploratory".
   - total_students: null (no original empirical sample) UNLESS the review \
     reports a pooled sample size from included studies.
   - ml_techniques.primary: null; ml_techniques.all_techniques: [] UNLESS \
     the review itself applies ML (e.g., topic modeling on abstracts, \
     automated text classification of papers).
   - plausible_values_handling: "not_applicable".
   - missing_data_handling: "not_reported" unless the review describes a \
     specific protocol for handling missing studies/data.
   - MUST trigger null_fields_interpretation explaining: "This is a \
     systematic review / meta-analysis / bibliometric study without original \
     ILSA micro-data analysis."
   - student_weights_used: null; replicate_weights_used: null.

15) NON-EMPIRICAL / FRAMEWORK / APP-DEVELOPMENT PAPERS:
   - Papers that develop theoretical frameworks, assessment designs, mobile \
     apps, or methodological proposals without analyzing ILSA student data.
   - Examples: ISM-based cognitive model construction, CAT algorithm design, \
     mobile learning app development, scaling methodology papers, simulation \
     studies.
   - total_students: null (or the expert panel / pilot sample if reported).
   - ml_techniques: extract ONLY if the paper actually trains/evaluates ML \
     models. Framework proposals citing ML concepts do NOT count.
   - plausible_values_handling: "not_applicable".
   - research_design_type: "exploratory" for theoretical/framework papers; \
     "predictive" if simulations test predictive models.
   - MUST trigger null_fields_interpretation explaining the non-empirical nature.

16) ANTI-LAZINESS ENFORCEMENT — MANDATORY EXTRACTION RULES:
   - *** ZERO-TOLERANCE FOR UNNECESSARY NULLS ***
   - The following fields MUST NEVER be null without exhaustive justification:
     a) total_students — scan EVERY section for sample size indicators. \
        If the paper says "N=4,552 students" anywhere, extract 4552.
     b) countries — if the paper names ANY country, economy, or region \
        in connection with data analysis, extract its ISO code. NEVER \
        return an empty countries list for an empirical paper.
     c) ml_techniques.primary — if all_techniques has ≥1 entry, primary \
        MUST be filled. This is a FATAL ERROR if violated.
     d) outcome_summary — MUST always be 4-5 substantive sentences with \
        specific metrics. Never write vague placeholders.
     e) research_design_type — MUST be classified for every paper.
     f) publication_type and source_category — MUST be classified.
   - For EVERY null field in your output, ask yourself: "Did I truly search \
     the abstract, methodology, results, tables, footnotes, and appendices?" \
     If not, search again.
   - PENALTY PATTERN: If you return more than 3 null fields in metadata or \
     more than 2 null fields in data (excluding the interpretation fields), \
     you are being LAZY. Re-read and extract harder.

17) XAI & CAUSALITY SCRUTINY (Explainability ≠ Causality):
   - Many ML papers use SHAP, LIME, Accumulated Local Effects (ALE), or \
     feature importance rankings and then implicitly or explicitly suggest \
     causal relationships. This is methodologically unsound on cross-sectional \
     ILSA data.
   - When extracting, rigorously differentiate between 'predictive feature \
     importance' (what variables improve the model's prediction accuracy) and \
     'causal inference' (what variables actually cause the outcome).
   - If the paper uses only SHAP / LIME / Gini importance / permutation \
     importance → research_design_type stays "predictive", NOT causal.
   - ONLY classify as "causal_observational" if the paper employs actual \
     causal ML methods (BART, BCF, Propensity Score Matching, diff-in-diff, \
     instrumental variables, regression discontinuity) AND explicitly states \
     causal identification assumptions (SUTVA, parallel trends, unconfoundedness).
   - If the authors overstate causality based on predictive ML feature \
     importance alone, capture this in outcome_summary as a limitation.
   - XAI technique names (SHAP, LIME, counterfactual XAI, ALE plots) should \
     be mentioned in outcome_summary when used but NEVER in ml_techniques \
     (they are interpretation tools, not learning algorithms).

18) HIERARCHICAL DATA AWARENESS (Nested Structure & i.i.d. Violations):
   - ILSA data is strictly nested: students → classrooms → schools → countries.
   - Standard ML algorithms (regular XGBoost, Random Forest, SVM, NN) assume \
     independent and identically distributed (i.i.d.) observations, which is \
     violated by the clustered ILSA sampling design.
   - When extracting methodology, identify EXACTLY how the ML model accounted \
     for hierarchical structure:
     a) Multi-level ML models: Mixed-Effects Random Forests, glmmLasso, \
        multilevel XGBoost, hierarchical neural networks.
     b) Feature aggregation: school-level or country-level averages used as \
        additional predictors alongside student-level features.
     c) Post-hoc corrections: HLM or cluster-robust standard errors applied \
        after ML prediction stage.
     d) Country-stratified modeling: separate models per country or per school.
     e) No adjustment at all: flat ML on raw student-level data.
   - If the study applies standard flat ML on nested student-level data \
     WITHOUT survey weights, WITHOUT hierarchical adjustments, and WITHOUT \
     cluster-stratified modeling, note this in outcome_summary as a \
     methodological limitation. Do NOT silently ignore it.
   - student_weights_used is especially important here: ILSA weights partially \
     correct for clustering. If weights are omitted AND hierarchy is ignored, \
     both findings should be flagged.

19) PROCESS DATA DYNAMICS — TIME & SEQUENCE GRANULARITY:
   - Beyond Rule 13's general process data guidance, rigorously extract HOW \
     time and action sequences were operationalized:
   - TIME METRICS — differentiate the following:
     a) Raw total time (crude, loses item-level dynamics).
     b) Time-to-first-action (engagement onset latency).
     c) Item-standardised log response times (accounts for item difficulty).
     d) Effort regulation slope (change in response time across items).
     e) Differential Response Time (DRT = observed − expected time).
     f) Response Time Effort (RTE = binary rapid-guess thresholding).
   - SEQUENCE MINING — differentiate:
     a) Exact chronological sequence modeling (n-grams, HMM, LSTM on action \
        streams, sequence autoencoders, Markov chains).
     b) Lazy frequency aggregation (total clicks, total resets, action counts \
        without order preservation).
     c) Graph-based representations (directed graph features, network \
        statistics from action transitions).
   - STRATEGY INFERENCE — extract how cognitive strategy was operationalized:
     a) VOTAT detection (systematic vs. non-systematic exploration).
     b) Clustering of sequential paths (k-means on action embeddings).
     c) Manual expert coding of strategy types.
   - Capture these distinctions in confounders_identified (list the specific \
     process features) and outcome_summary (describe the approach).

20) ML ROBUSTNESS, CLASS IMBALANCE & DATA LEAKAGE:
   - Do NOT blindly extract overall model "Accuracy" as the sole metric.
   - CLASS IMBALANCE HANDLING:
     a) If the target variable is skewed (e.g., top 5% resilient students, \
        dropout prediction, cheating detection), extract whether the study \
        used SMOTE, ADASYN, under-sampling, class-weighted loss functions, \
        cost-sensitive learning, or threshold calibration.
     b) Remember: SMOTE/ADASYN/CTGAN/VAE-augmentation are CLASS BALANCING \
        methods, NOT missing_data_handling (see Rule 3 and checklist A).
   - EVALUATION METRICS:
     a) For imbalanced classification, extract F1-Score, Cohen's Kappa, \
        Precision-Recall AUC, Matthews Correlation Coefficient (MCC), or \
        balanced accuracy — these are robust to class skew.
     b) If ONLY "Accuracy" is reported for a known-imbalanced target, note \
        this as a limitation in outcome_summary.
   - DATA LEAKAGE:
     a) Check whether imputation, standardization, SMOTE, or feature \
        selection were performed INSIDE cross-validation folds (correct) or \
        on the ENTIRE dataset before splitting (leakage).
     b) If the paper reports suspiciously high performance (e.g., >95% on \
        complex ILSA tasks) without rigorous nested CV, flag potential \
        data leakage in outcome_summary.
   - VALIDATION STRATEGY: extract the exact method — k-fold CV, stratified \
     k-fold, leave-one-group-out (LOGO), nested CV, hold-out, repeated \
     random splits — and note it in outcome_summary.

═══════════════════════════════════════════════════════════════
OUTPUT SCHEMA
═══════════════════════════════════════════════════════════════

Return a single JSON with exactly two top-level keys: metadata, data.

metadata fields: file_name, title, authors, year, doi, venue, publication_type,
  open_access, source_category.

data fields: survey_design, plausible_values_handling, missing_data_handling,
  sample_details, ml_techniques, confounders_identified, outcome_summary,
  research_design_type, null_fields_interpretation.

data.survey_design: student_weights_used, replicate_weights_used,
  weight_variable_name, weight_fields_interpretation.

data.sample_details: total_students, countries (each: country_code, n_students).

data.ml_techniques: primary, all_techniques.

Do not emit any other top-level or nested keys.
No markdown fences, no preamble — valid JSON only.

"""


@dataclass
class ExtractionResult:
    file_name: str
    success: bool
    extraction: ILSAArticleMetadata | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_seconds: float
    error: str | None = None


class GPTExtractor:
    _use_structured: bool | None = None

    def __init__(
        self,
        api_key: str = None,
        model: str = MODEL_NAME,
        max_retries: int = 4,
        base_delay: float = 2.0,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay

    @staticmethod
    def _post_process_model(extraction: ILSAArticleMetadata) -> None:
        """Post-process a structured-output extraction in place."""
        for c in extraction.data.sample_details.countries:
            code = c.country_code.strip()
            if len(code) != 3 or not code.isalpha():
                mapped = COUNTRY_NAME_TO_ISO.get(code.lower())
                if mapped:
                    c.country_code = mapped
            c.country_code = c.country_code.upper()

        ml = extraction.data.ml_techniques
        if ml.primary is None and len(ml.all_techniques) == 1:
            ml.primary = ml.all_techniques[0]

        sd = extraction.data.survey_design
        has_positive = (
            sd.student_weights_used is True
            or sd.replicate_weights_used is True
            or (sd.weight_variable_name and sd.weight_variable_name.strip())
        )
        if has_positive:
            sd.weight_fields_interpretation = None

    def _build_user_message(self, processed: "ProcessedPDF") -> list[dict]:
        sections_label = ", ".join(processed.sections.keys()) or "none"
        title_hint = ""
        if processed.metadata.get("extracted_title"):
            title_hint = (
                f"\nEXTRACTED_TITLE_HINT (use only if the body never states "
                f"a title; still do not contradict the PDF): "
                f"{processed.metadata['extracted_title']}\n"
            )

        document_text = (
            f"FILE: {processed.file_name}\n"
            f"SOURCE: {processed.source_database}\n"
            f"SECTIONS_DETECTED: {sections_label}\n"
            f"{title_hint}"
            f"--- BEGIN ARTICLE TEXT ---\n\n"
            f"{processed.extraction_text}\n\n"
            f"--- END ARTICLE TEXT ---\n\n"
            "Extract the structured JSON from the article above using the "
            "system prompt rules. Return valid JSON only."
        )

        interpretation_text = (
            "EXPERT INFERENCE CHECKLIST (same article; apply BEFORE finalising JSON):\n\n"

            "A) STRICT ENUMS — publication_type, source_category, research_design_type, "
            "plausible_values_handling, missing_data_handling must each be EXACTLY one "
            "of the allowed values listed in the system prompt. Use the synonym tables "
            "to map academic jargon. Never write free-text descriptions or new slugs.\n"
            "  PV SYNONYM TABLE: 'FIML' → multiple_imputation; 'complete cases' → "
            "listwise_deletion; 'averaged across five PVs' → average_pv; 'PV1' → "
            "single_pv; 'WLE scores' → wle; 'IRT theta' → irt_theta; 'EAP' → "
            "irt_theta; '10 PVs averaged' → all_pv.\n"
            "  MISSING DATA TABLE (map ALL to schema literals):\n"
            "    'kNN imputation' / 'missForest' / 'missRanger' / 'RF-based' / "
            "'MCMC' / 'PMM' / 'MICE' / 'chained equations' / 'FIML' / "
            "'EM algorithm' / 'expectation maximization' / 'hot-deck' / "
            "'rblimp' / 'blimp' / 'Bayesian imputation' / "
            "'stochastic regression imputation' / 'two-level FCS' / "
            "'fully conditional specification' → multiple_imputation.\n"
            "    'mean substitution' / 'series mean' / 'mode imputation' / "
            "'median imputation' / 'SimpleImputer' / 'zero imputation' / "
            "'replaced with zero' → mean_imputation.\n"
            "    'listwise deletion' / 'complete case' / 'removed missing' / "
            "'excluded incomplete' → listwise_deletion.\n"
            "    'pairwise deletion' / 'available case' → pairwise_deletion.\n"
            "  CAUTION: SMOTE / SMOTETomek / ADASYN / CTGAN / VAE-augmentation "
            "are CLASS BALANCING / synthetic-data techniques, NOT missing data handling.\n"
            "  CAUTION: 'winsorized' / 'trimmed at percentile' are OUTLIER TREATMENT, "
            "NOT missing data handling.\n\n"

            "B) COUNTRY CODES — every country_code must be ISO 3166-1 alpha-3 "
            "(3 uppercase letters). Never write full names or 2-letter codes.\n"
            "  SPECIAL MAPPINGS: B-S-J-Z / Beijing-Shanghai-Jiangsu-Zhejiang → CHN; "
            "B-S-J-G / Beijing-Shanghai-Jiangsu-Guangdong → CHN; "
            "Chinese Taipei / Taiwan → TWN; Hong Kong → HKG; Macao / Macau → MAC; "
            "England / Northern Ireland / Scotland / Wales → GBR; "
            "Türkiye / Turkey → TUR; "
            "Republic of Korea / South Korea → KOR; UAE → ARE; "
            "Czech Republic / Czechia → CZE; Dominican Republic → DOM; "
            "Flemish Community → BEL; Costa Rica → CRI.\n"
            "  NOTE: PIAAC, ICILS, and ICCS use the same ISO codes as PISA/TIMSS.\n\n"

            "C) ML TECHNIQUES ONLY — all_techniques and primary must contain ONLY "
            "Machine Learning / predictive-modeling algorithms. DO NOT include: "
            "PCA, factor analysis, t-tests, ANOVA, chi-square, correlations, "
            "descriptive statistics, EFA/CFA, SEM, HLM (unless ML baseline), "
            "Mantel-Haenszel, IRT model fitting, or ESCS computations.\n"
            "  INCLUDE: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, "
            "Histogram GBR, SVM/SVR, LASSO, Elastic Net, Ridge Regression, "
            "Group Mnet, glmmLasso, KNN, Naive Bayes, Bayesian Ridge, "
            "Decision Tree / CART / C5.0, Conditional Inference Trees/Forests, "
            "Logistic Regression (when used as ML classifier), "
            "Neural Networks (ANN/MLP/DNN), LSTM, GRU, CNN, RNN, "
            "Autoencoder, BART/BCF (Bayesian causal ML), Bayesian Network, "
            "AdaBoost, Extra Trees, Stacking / Blending / ensemble meta-models, "
            "ANFIS, Discriminant Analysis, Gaussian Process, "
            "Deep Knowledge Tracing, Word2Vec, Doc2Vec, TF-IDF + classifier.\n"
            "  DO NOT INCLUDE: LPA, LCA, k-means/DBSCAN/k-medoids/hierarchical "
            "clustering/GMM (when purely exploratory without a supervised prediction "
            "goal), HLM, CFA/SEM, IRT, DCMs (HO-DINA/GDINA/DINO/ACDM), ISM, "
            "Process Mining (Disco/ProM), finite mixture models, "
            "bibliometric analysis.\n"
            "  Set primary to the best-performing model; if ambiguous pick the one "
            "highlighted in the abstract.\n\n"

            "D) SURVEY WEIGHTS (CRITICAL — system rules 4 + 6):\n"
            "  Aggressively scan methodology, data, footnotes, and table notes for "
            "weight terms (W_FSTUWT, TOTWGT, senate/house weights, BRR, jackknife, "
            "complex survey design, stratification, clustering).\n"
            "  SOFTWARE-BASED INFERENCE: If paper uses IEA IDB Analyzer, bifiesurvey, "
            "WeMix, lavaan.survey, EdSurvey, RALSA, intsvy, or repest → infer "
            "student_weights_used = true (these tools inherently apply weights).\n"
            "  ML-SPECIFIC PATTERN: Many ML studies (RF, XGBoost, SVM, NN) on ILSA "
            "data deliberately omit survey weights. If ML is used without weight "
            "mention → set student_weights_used = false and FILL "
            "weight_fields_interpretation explaining the ML omission pattern.\n"
            "  *** FAIL-SAFE ***: when both student_weights_used and "
            "replicate_weights_used are false or null AND weight_variable_name is null, "
            "you MUST fill weight_fields_interpretation with 3-4 sentences. "
            "Leaving all weight fields as false/null AND weight_fields_interpretation "
            "as null is a FATAL ERROR.\n\n"

            "E) SAMPLE DETAILS (system rule 7) — exhaustively search Method, "
            "Participants, Data, Data Cleaning, Data Preprocessing, and Results for "
            "total N. Look for 'N =', 'final sample', 'analytic sample', 'valid "
            "responses', 'after removing/exclusion', 'remained for analysis'. "
            "Check tables and figure captions. For countries, extract ALL ISO "
            "alpha-3 codes; never leave countries empty if the data source implies "
            "a country. If the paper says '80 countries' but lists specific ones, "
            "extract EVERY named country.\n\n"

            "F) ML PRIMARY (system rule 8) — *** FATAL ERROR *** to leave primary "
            "null while all_techniques has values. If only ONE algorithm is listed, "
            "it IS the primary. If multiple, scan Results/Abstract/Conclusion for "
            "'performed best', 'highest accuracy/R²/AUC', 'outperformed', 'lowest "
            "RMSE/MAE/MAPE'. If truly ambiguous pick the one highlighted in the "
            "abstract or conclusion.\n\n"

            "G) CONFOUNDERS — list variable names or short phrases controlled for "
            "as predictors (SES, gender, parental education, ESCS, immigration "
            "status, school type, etc.). [] only if the paper truly names none.\n\n"

            "H) outcome_summary — 4-5 sentences of findings and performance metrics "
            "ONLY from the text. Include specific numbers (accuracy, R², RMSE, AUC, "
            "F1) when available. Do NOT put null-field commentary here.\n\n"

            "I) null_fields_interpretation — trigger if total_students is still "
            "null, or primary is null while all_techniques is empty, or extraction "
            "is extremely sparse. Write a diagnostic note explaining WHY. "
            "If the record is reasonably dense, this MUST be null.\n\n"

            "J) ANTI-HALLUCINATION — never invent DOIs, exact N, country codes, "
            "weight variable names, or algorithm names absent from the text. "
            "Inference applies ONLY to categorical/boolean/enum fields.\n\n"

            "K) RESEARCH DESIGN — classify using system rule 10: predictive (ML "
            "prediction/classification), causal_observational (BART, BCF, PSM, "
            "diff-in-diff), causal_experimental (RCT), exploratory (clustering-only, "
            "LPA-only, reviews, theoretical). If paper combines prediction AND "
            "clustering → 'predictive'.\n\n"

            "L) PROCESS DATA PAPERS (system rule 13) — if paper analyzes log files, "
            "clickstreams, response times, action sequences, VOTAT strategies, "
            "N-grams, or mouse/keyboard traces:\n"
            "  - plausible_values_handling → 'not_applicable' (process data uses "
            "binary correctness or IRT ability, not PVs).\n"
            "  - student_weights_used → usually false (process data studies focus "
            "on behavioral patterns, not population estimation).\n"
            "  - ml_techniques: include ALL ML algorithms used for classification "
            "or prediction of process outcomes (RF, LSTM, GRU, CNN, SVM, "
            "Autoencoder, k-means if part of a prediction pipeline, etc.).\n"
            "  - research_design_type → 'predictive' if classifying engagement/"
            "performance; 'exploratory' if only profiling/clustering.\n"
            "  - Capture process-specific features (response time, action counts, "
            "time-to-first-action, number of visits) in independent variables.\n\n"

            "M) REVIEW / META-ANALYSIS / BIBLIOMETRIC PAPERS (system rule 14):\n"
            "  - source_category → 'review_article'.\n"
            "  - research_design_type → 'exploratory'.\n"
            "  - total_students → null (no original sample).\n"
            "  - ml_techniques.primary → null; all_techniques → [] UNLESS the "
            "review itself applies ML (e.g., topic modeling on abstracts).\n"
            "  - plausible_values_handling → 'not_applicable'.\n"
            "  - missing_data_handling → 'not_reported'.\n"
            "  - MUST trigger null_fields_interpretation explaining it is a review.\n\n"

            "N) NON-EMPIRICAL / FRAMEWORK / APP-DEVELOPMENT / DATA PAPERS (system rule 15):\n"
            "  - Papers designing theoretical frameworks, mobile apps, CAT "
            "algorithms, scaling methodologies, or DATA PAPERS that construct/"
            "harmonize datasets without ILSA micro-data analysis.\n"
            "  - publication_type: 'journal' for data papers published in journals "
            "(e.g., 'Data' journal); 'report' for technical data documentation.\n"
            "  - source_category: 'methodology_paper' for data papers and frameworks.\n"
            "  - total_students → null (or expert panel size if applicable).\n"
            "  - plausible_values_handling → 'not_applicable'.\n"
            "  - research_design_type → 'exploratory' for data description / "
            "dataset construction papers.\n"
            "  - MUST trigger null_fields_interpretation explaining the study type.\n\n"

            "O) XAI & CAUSALITY (system rule 17):\n"
            "  - If the paper uses SHAP / LIME / ALE / Gini importance / permutation "
            "importance, report these in outcome_summary but NEVER in ml_techniques.\n"
            "  - Do NOT classify as 'causal_observational' unless actual causal methods "
            "(BART, BCF, PSM, diff-in-diff, IV, RDD) with stated assumptions are used.\n"
            "  - If authors claim 'X causes Y' based solely on feature importance, "
            "flag this as overstated causality in outcome_summary.\n\n"

            "P) HIERARCHICAL DATA (system rule 18):\n"
            "  - Note in outcome_summary whether the ML model accounted for the nested "
            "ILSA data structure (multilevel ML, feature aggregation, cluster-stratified "
            "models, or survey weights).\n"
            "  - If standard flat ML was applied to student-level data with NO "
            "hierarchical adjustments and NO weights, flag it as a methodological "
            "limitation in outcome_summary.\n\n"

            "Q) PROCESS DATA GRANULARITY (system rule 19):\n"
            "  - For process data papers, differentiate in outcome_summary:\n"
            "    (a) time metric type: raw total time vs. item-standardised log "
            "response times vs. effort regulation slope vs. DRT/RTE;\n"
            "    (b) sequence modeling: exact chronological (n-grams, HMM, LSTM on "
            "actions, Markov) vs. lazy frequency counts (total clicks);\n"
            "    (c) strategy inference: VOTAT detection, path clustering, expert "
            "coding.\n"
            "  - List specific process features in confounders_identified.\n\n"

            "R) ML ROBUSTNESS & LEAKAGE (system rule 20):\n"
            "  - Extract ALL performance metrics reported (Accuracy, F1, AUC, Kappa, "
            "MCC, RMSE, MAE, R²). If only 'Accuracy' is reported for an imbalanced "
            "classification, note it as a limitation.\n"
            "  - Extract class imbalance handling (SMOTE, under-sampling, class "
            "weights) — remember these are NOT missing_data_handling.\n"
            "  - Note the validation strategy (k-fold CV, nested CV, hold-out, LOGO).\n"
            "  - If preprocessing (imputation, scaling, SMOTE, feature selection) was "
            "done BEFORE train/test split, flag potential data leakage.\n\n"

            "S) FINAL ANTI-LAZINESS CHECK (system rule 16):\n"
            "  Before submitting your JSON, count your null fields:\n"
            "  - total_students null for an empirical paper? → Re-read Method section.\n"
            "  - countries list empty for a paper that names countries? → FATAL ERROR.\n"
            "  - primary null but all_techniques has entries? → FATAL ERROR.\n"
            "  - outcome_summary vague or <3 sentences? → Add specific metrics.\n"
            "  - More than 3 null metadata fields? → You are being LAZY. Extract more.\n"
            "  - More than 2 null data fields (excluding interpretations)? → Re-scan.\n"
        )

        return [
            {"type": "text", "text": document_text},
            {"type": "text", "text": interpretation_text},
        ]

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * PRICE_INPUT_PER_1M / 1_000_000
            + output_tokens * PRICE_OUTPUT_PER_1M / 1_000_000
        )

    @staticmethod
    def _coerce_pv_literal(value) -> str:
        """Map free-text / invalid PV labels to schema literals."""
        allowed = frozenset({
            "rubin_rules", "single_pv", "average_pv", "mitml",
            "not_applicable", "not_reported",
            "wle", "irt_theta", "all_pv",
        })
        if value in allowed:
            return value
        if not isinstance(value, str):
            return "not_reported"
        t = value.lower().replace("-", "_").replace(" ", "_")
        if "rubin" in t or "combined_estimates" in t:
            return "rubin_rules"
        if "mitml" in t or "mplus" in t:
            return "mitml"
        sw_rubin = (
            "bifiesurvey", "repest", "intsvy", "edsurvey",
            "idb_analyzer", "idb analyzer", "ralsa",
            "lavaan.survey", "wemix",
        )
        if any(pkg in t for pkg in sw_rubin):
            return "rubin_rules"
        if (
            "five_pv" in t or "5_pv" in t
            or "five_plausible" in t
            or "ten_plausible" in t or "10_pv" in t and "pool" in t
            or "pv1_pv5" in t or "pv1_pv10" in t
            or "analyses_repeated_across" in t and "pv" in t
            or "repeated_across_pvs" in t
        ):
            return "rubin_rules"
        if (
            "not_applicable" in t
            or "no_pv" in t
            or "no_pvs" in t
            or "no_plausible" in t
            or "does_not_use" in t
            or "process_data" in t
            or "log_file" in t
            or "review_paper" in t
            or "non_empirical" in t
            or "no_assessment_score" in t
        ):
            return "not_applicable"
        if "wle" in t or "weighted_likelihood" in t or "warm" in t and "estimat" in t:
            return "wle"
        if (
            "irt_theta" in t
            or "eap" in t and ("estimat" in t or "score" in t or "abilit" in t)
            or "theta" in t and ("irt" in t or "estimat" in t or "latent" in t)
            or "latent_trait" in t
        ):
            return "irt_theta"
        if (
            "first_plausible" in t
            or "single_pv" in t
            or "pv1_only" in t
            or t == "pv1"
            or ("separate" in t and "plausible" in t)
            or "per_pv" in t
            or "per_plausible" in t
            or "one_pv" in t
            or ("target" in t and "indicator" in t)
            or ("binary" in t and ("pv" in t or "plausible" in t))
            or "pv1math" in t or "pv1read" in t or "pv1scie" in t
            or "pv2scie" in t or "pv2math" in t
        ):
            return "single_pv"
        if (
            "all_pv" in t
            or "all_10" in t and "pv" in t
            or "ten_pv" in t
            or "10_pv" in t
            or "each_pv" in t and "separate" in t
            or "pv1_through" in t
            or "pv1_to_pv10" in t
            or "all_plausible_values_separately" in t
        ):
            return "all_pv"
        if (
            "average" in t and "pv" in t
            or "all_plausible" in t
            or "across_pv" in t
            or "across_pvs" in t
            or "mean_pv" in t
            or "mean_of" in t and "plausible" in t
            or "averaged" in t and "plausible" in t
        ):
            return "average_pv"
        if "plausible" in t or "_pv" in t or "pv_" in t:
            return "not_reported"
        return "not_reported"

    @staticmethod
    def _coerce_md_literal(value) -> str:
        """Map free-text / invalid missing-data labels to schema literals."""
        allowed = frozenset({
            "listwise_deletion", "pairwise_deletion", "mean_imputation",
            "multiple_imputation", "not_reported",
        })
        if value in allowed:
            return value
        if not isinstance(value, str):
            return "not_reported"
        t = value.lower().replace("-", "_").replace(" ", "_")
        if len(t) > 120 or "the_manuscript" in t or "the_paper" in t or "the_dataset" in t:
            return "not_reported"
        if "no_missing" in t or "without_missing" in t or "no_missing_data" in t:
            return "not_reported"
        if (
            "winsoriz" in t
            or "winsoris" in t
            or "trimmed" in t and "percentile" in t
        ):
            return "not_reported"
        if (
            "smote" in t
            or "smotetomek" in t
            or "adasyn" in t
            or "oversampl" in t
            or "undersamp" in t
            or "class_balanc" in t
            or "resampl" in t and ("minority" in t or "imbalanc" in t)
            or "ctgan" in t
            or "vae_augment" in t
            or "synthetic_data" in t and ("generat" in t or "augment" in t or "balanc" in t)
        ):
            return "not_reported"
        if "pairwise" in t:
            return "pairwise_deletion"
        if (
            "listwise" in t
            or "complete_case" in t
            or ("exclusion" in t and "missing" in t)
            or ("removed" in t and "missing" in t)
            or ("deleted" in t and "missing" in t)
            or ("cases_with_missing" in t and ("removed" in t or "excluded" in t or "dropped" in t))
        ):
            return "listwise_deletion"
        if (
            ("mean" in t and "imput" in t)
            or ("mean" in t and "substitut" in t)
            or ("mean" in t and "replac" in t)
            or ("series_mean" in t)
            or ("mode" in t and ("imput" in t or "substitut" in t or "replac" in t))
            or ("median" in t and ("imput" in t or "substitut" in t or "replac" in t))
            or ("simple_imputer" in t and ("mode" in t or "median" in t or "mean" in t))
            or ("simpleimputer" in t)
            or ("substituted_mode" in t)
            or ("substituted_median" in t)
            or ("zero_fill" in t)
            or ("zero_imputation" in t)
            or ("replaced_with_zero" in t)
            or ("filled_with_zero" in t)
        ):
            return "mean_imputation"
        if (
            "imput" in t
            or "mice" in t
            or "missforest" in t
            or "miss_forest" in t
            or "missranger" in t
            or "miss_ranger" in t
            or "rf_based" in t
            or "fiml" in t
            or "full_information" in t
            or "maximum_likelihood" in t
            or "em_algorithm" in t
            or "hot_deck" in t
            or "hot deck" in t
            or "chained_equations" in t
            or "fully_conditional" in t
            or ("machine_learning" in t and "missing" in t)
            or t == "imputation"
            or "knn" in t
            or "k_nearest" in t
            or "k_nn" in t
            or "nearest_neighbor" in t
            or "mcmc" in t
            or "markov_chain" in t
            or "pmm" in t
            or "predictive_mean" in t
            or "two_level_fcs" in t
            or "multiple_imput" in t
            or "multivariate_imput" in t
            or "rblimp" in t
            or "blimp" in t
            or "expectation_maximiz" in t
            or "bayesian_imput" in t
            or "stochastic_regress" in t
        ):
            return "multiple_imputation"
        if (
            "dropped_missing" in t
            or "excluded_missing" in t
            or "removed_incomplete" in t
            or "omitted_missing" in t
            or "filtered_out" in t and "missing" in t
            or "discarded" in t and "missing" in t
            or "dropped" in t and "incomplete" in t
            or "cases_removed" in t
        ):
            return "listwise_deletion"
        return "not_reported"

    @staticmethod
    def _sanitize(parsed_data: dict) -> dict:
        """
        Post-process model output to fix known failure modes before Pydantic validation.
        Modifies parsed_data in place and returns it.
        """
        INVALID_STR = {"not_reported", "not_applicable", "N/A", "n/a", "unknown", ""}

        def _normalize_literal(value, field_name, allowed, default):
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                text = " ".join(
                    str(v) for v in value.values() if isinstance(v, str)
                ).lower()
                if field_name == "plausible_values_handling":
                    if "plausible" in text or "pv" in text:
                        return "not_reported"
                    if "no" in text or "not" in text and "pv" not in text:
                        return "not_applicable"
                if field_name == "missing_data_handling":
                    if "imput" in text or "mice" in text or "miss" in text:
                        return "multiple_imputation"
                    if "listwise" in text or "complete case" in text or "complete-case" in text:
                        return "listwise_deletion"
                    return "not_reported"
            return default

        DATA_KEYS = (
            "survey_design",
            "plausible_values_handling",
            "missing_data_handling",
            "sample_details",
            "ml_techniques",
            "confounders_identified",
            "outcome_summary",
            "research_design_type",
            "null_fields_interpretation",
        )

        if not isinstance(parsed_data.get("data"), dict):
            parsed_data["data"] = {}
        data = parsed_data["data"]

        # Legacy flat JSON → nest under data
        for k in DATA_KEYS:
            if k in parsed_data:
                if k not in data:
                    data[k] = parsed_data.pop(k)
                else:
                    parsed_data.pop(k, None)

        for key in list(parsed_data.keys()):
            if key not in ("metadata", "data"):
                parsed_data.pop(key, None)

        for key in list(data.keys()):
            if key not in DATA_KEYS:
                data.pop(key, None)

        ml = data.get("ml_techniques")
        if isinstance(ml, dict):
            for legacy in ("feature_selection", "baseline_model", "xai_method"):
                ml.pop(legacy, None)
            primary = ml.get("primary")
            if primary is None or (isinstance(primary, str) and primary in INVALID_STR):
                ml["primary"] = None
            elif isinstance(primary, list):
                ml["primary"] = None
            if isinstance(ml.get("all_techniques"), list):
                ml["all_techniques"] = [
                    t for t in ml["all_techniques"]
                    if isinstance(t, str) and t not in INVALID_STR
                ]
            elif isinstance(ml.get("all_techniques"), str):
                ml["all_techniques"] = [ml["all_techniques"]]
            elif ml.get("all_techniques") is None:
                ml["all_techniques"] = []

            if ml["primary"] is None and len(ml["all_techniques"]) == 1:
                ml["primary"] = ml["all_techniques"][0]

        sd = data.get("sample_details")
        if isinstance(sd, dict):
            countries = sd.get("countries")
            if isinstance(countries, list):
                cleaned = []
                for c in countries:
                    if not isinstance(c, dict):
                        continue
                    code = c.get("country_code")
                    if not code or not isinstance(code, str):
                        continue
                    code = code.strip()
                    if len(code) != 3 or not code.isalpha():
                        mapped = COUNTRY_NAME_TO_ISO.get(code.lower())
                        if mapped:
                            code = mapped
                        else:
                            continue
                    c["country_code"] = code.upper()
                    n = c.get("n_students")
                    if not isinstance(n, int):
                        c["n_students"] = None
                    cleaned.append(c)
                sd["countries"] = cleaned

        sdw = data.get("survey_design")
        if isinstance(sdw, dict):
            wfi = sdw.get("weight_fields_interpretation")
            if isinstance(wfi, str) and wfi.strip() in INVALID_STR:
                sdw["weight_fields_interpretation"] = None
            wn = sdw.get("weight_variable_name")
            if isinstance(wn, str) and wn in INVALID_STR:
                sdw["weight_variable_name"] = None
                wn = None

            has_positive_weight_evidence = (
                sdw.get("student_weights_used") is True
                or sdw.get("replicate_weights_used") is True
                or (isinstance(wn, str) and wn.strip())
            )
            if has_positive_weight_evidence:
                sdw["weight_fields_interpretation"] = None

        nfi = data.get("null_fields_interpretation")
        if isinstance(nfi, str) and nfi.strip() in INVALID_STR:
            data["null_fields_interpretation"] = None

        VALID_PUB_TYPES = frozenset({
            "journal", "conference", "book_chapter", "preprint", "report", "thesis",
        })
        VALID_SOURCE_CATS = frozenset({
            "technical_report", "review_article", "methodology_paper",
            "peer_reviewed_research",
        })
        VALID_DESIGN_TYPES = frozenset({
            "predictive", "causal_observational", "causal_experimental", "exploratory",
        })

        meta = parsed_data.get("metadata")
        if isinstance(meta, dict):
            for legacy in (
                "extraction_timestamp",
                "extraction_cost_usd",
                "prompt_tokens",
                "completion_tokens",
            ):
                meta.pop(legacy, None)
            for field in ("doi", "venue", "title"):
                if meta.get(field) in INVALID_STR:
                    meta[field] = None
            if not isinstance(meta.get("authors"), list):
                meta["authors"] = []
            pt = meta.get("publication_type")
            if isinstance(pt, str) and pt not in VALID_PUB_TYPES:
                normed = pt.lower().replace("-", "_").replace(" ", "_")
                if normed in VALID_PUB_TYPES:
                    meta["publication_type"] = normed
                else:
                    matched = None
                    if "conference" in normed or "proceeding" in normed or "symposium" in normed:
                        matched = "conference"
                    elif "review" in normed or "survey" in normed or "systematic" in normed:
                        matched = "journal"
                    elif "process_data" in normed or "paper" in normed:
                        matched = "journal"
                    elif "data_paper" in normed or "data_article" in normed:
                        matched = "journal"
                    elif "thesis" in normed or "dissertation" in normed:
                        matched = "thesis"
                    elif "preprint" in normed or "arxiv" in normed or "working_paper" in normed:
                        matched = "preprint"
                    elif "report" in normed or "technical" in normed:
                        matched = "report"
                    elif "book" in normed or "chapter" in normed:
                        matched = "book_chapter"
                    else:
                        for v in VALID_PUB_TYPES:
                            if v in normed or normed.startswith(v):
                                matched = v
                                break
                    meta["publication_type"] = matched

            sc = meta.get("source_category")
            if isinstance(sc, str) and sc not in VALID_SOURCE_CATS:
                normed = sc.lower().replace("-", "_").replace(" ", "_")
                if normed in VALID_SOURCE_CATS:
                    meta["source_category"] = normed
                else:
                    matched = None
                    if (
                        "review" in normed or "systematic" in normed
                        or "scoping" in normed or "bibliometric" in normed
                        or "meta_analysis" in normed or "meta_analytic" in normed
                        or "literature_survey" in normed
                    ):
                        matched = "review_article"
                    elif (
                        "method" in normed or "framework" in normed
                        or "simulation" in normed or "scaling" in normed
                        or "psychometric" in normed or "measurement" in normed
                        or "data_paper" in normed or "dataset" in normed
                    ):
                        matched = "methodology_paper"
                    elif "technical" in normed or "report" in normed:
                        matched = "technical_report"
                    elif (
                        "peer" in normed or "research" in normed
                        or "empirical" in normed or "original" in normed
                    ):
                        matched = "peer_reviewed_research"
                    else:
                        for v in VALID_SOURCE_CATS:
                            if v in normed or normed.startswith(v):
                                matched = v
                                break
                    if matched is None:
                        matched = "peer_reviewed_research"
                    meta["source_category"] = matched

        rdt = data.get("research_design_type")
        if isinstance(rdt, str) and rdt not in VALID_DESIGN_TYPES:
            normed = rdt.lower().replace("-", "_").replace(" ", "_")
            if normed in VALID_DESIGN_TYPES:
                data["research_design_type"] = normed
            else:
                matched = None
                if (
                    "predict" in normed or "classif" in normed
                    or "regress" in normed or "supervis" in normed
                ):
                    matched = "predictive"
                elif (
                    "causal" in normed and ("experiment" in normed or "rct" in normed)
                ):
                    matched = "causal_experimental"
                elif (
                    "causal" in normed or "propensity" in normed
                    or "diff_in_diff" in normed or "instrumental" in normed
                    or "counterfactual" in normed
                ):
                    matched = "causal_observational"
                elif (
                    "explor" in normed or "descript" in normed
                    or "cluster" in normed or "review" in normed
                    or "bibliometric" in normed or "profil" in normed
                    or "unsupervis" in normed or "framework" in normed
                ):
                    matched = "exploratory"
                else:
                    for v in VALID_DESIGN_TYPES:
                        if v in normed or normed.startswith(v):
                            matched = v
                            break
                data["research_design_type"] = matched

        data["plausible_values_handling"] = _normalize_literal(
            data.get("plausible_values_handling"),
            "plausible_values_handling",
            {
                "rubin_rules", "single_pv", "average_pv", "all_pv",
                "mitml", "wle", "irt_theta",
                "not_applicable", "not_reported",
            },
            "not_reported",
        )
        data["missing_data_handling"] = _normalize_literal(
            data.get("missing_data_handling"),
            "missing_data_handling",
            {
                "listwise_deletion", "pairwise_deletion", "mean_imputation",
                "multiple_imputation", "not_reported",
            },
            "not_reported",
        )

        pv_allowed = frozenset({
            "rubin_rules", "single_pv", "average_pv", "all_pv",
            "mitml", "wle", "irt_theta",
            "not_applicable", "not_reported",
        })
        md_allowed = frozenset({
            "listwise_deletion", "pairwise_deletion", "mean_imputation",
            "multiple_imputation", "not_reported",
        })

        pv_raw = data.get("plausible_values_handling")
        if pv_raw not in pv_allowed:
            data["plausible_values_handling"] = GPTExtractor._coerce_pv_literal(pv_raw)

        md_raw = data.get("missing_data_handling")
        if md_raw not in md_allowed:
            data["missing_data_handling"] = GPTExtractor._coerce_md_literal(md_raw)

        outcome = data.get("outcome_summary")
        if isinstance(outcome, dict):
            if isinstance(outcome.get("summary"), str):
                data["outcome_summary"] = outcome["summary"]
            else:
                data["outcome_summary"] = " ".join(
                    str(v) for v in outcome.values() if isinstance(v, str)
                )

        conf = data.get("confounders_identified")
        if not isinstance(conf, list):
            data["confounders_identified"] = []
        else:
            data["confounders_identified"] = [
                c for c in conf if isinstance(c, str) and c not in INVALID_STR
            ]

        return parsed_data

    def extract(self, processed: "ProcessedPDF") -> ExtractionResult:
        first_error = processed.parse_errors[0] if processed.parse_errors else None
        if not processed.extraction_text:
            return ExtractionResult(
                file_name=processed.file_name,
                success=False,
                extraction=None,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                duration_seconds=0.0,
                error=first_error or "Empty extracted text",
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(processed)},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                # ── Path A: Structured Outputs (token-level schema enforcement) ──
                if GPTExtractor._use_structured is not False:
                    try:
                        response = self.client.beta.chat.completions.parse(
                            model=self.model,
                            messages=messages,
                            temperature=0.0,
                            response_format=ILSAArticleMetadata,
                        )
                        duration = time.perf_counter() - start
                        extraction = response.choices[0].message.parsed
                        if extraction is not None:
                            GPTExtractor._use_structured = True
                            extraction.metadata.file_name = processed.file_name
                            self._post_process_model(extraction)
                            usage = response.usage
                            cost = self._calculate_cost(
                                usage.prompt_tokens, usage.completion_tokens
                            )
                            return ExtractionResult(
                                file_name=processed.file_name,
                                success=True,
                                extraction=extraction,
                                input_tokens=usage.prompt_tokens,
                                output_tokens=usage.completion_tokens,
                                cost_usd=cost,
                                duration_seconds=duration,
                            )
                        last_error = "Model refused structured output"
                        continue
                    except (AttributeError, TypeError):
                        GPTExtractor._use_structured = False
                        logger.info(
                            "Structured outputs unavailable in SDK, "
                            "falling back to JSON mode"
                        )
                    except APIError as struct_err:
                        if GPTExtractor._use_structured is None:
                            GPTExtractor._use_structured = False
                            logger.info(
                                "Model does not support structured outputs "
                                f"({struct_err}), falling back to JSON mode"
                            )
                        else:
                            raise

                # ── Path B: JSON mode with _sanitize + manual validation ──
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                duration = time.perf_counter() - start
                content = response.choices[0].message.content
                parsed_data = json.loads(content) if content else None

                if parsed_data is None:
                    last_error = "Model returned empty response"
                    break

                if isinstance(parsed_data.get("metadata"), dict):
                    parsed_data["metadata"]["file_name"] = processed.file_name

                parsed_data = self._sanitize(parsed_data)

                try:
                    extraction = ILSAArticleMetadata.model_validate(parsed_data)
                except ValidationError as e:
                    last_error = f"Schema validation failed: {e}"
                    logger.warning(
                        f"Validation error on {processed.file_name} "
                        f"attempt {attempt + 1}: {e}"
                    )
                    continue

                usage = response.usage
                cost = self._calculate_cost(
                    usage.prompt_tokens, usage.completion_tokens
                )
                return ExtractionResult(
                    file_name=processed.file_name,
                    success=True,
                    extraction=extraction,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cost_usd=cost,
                    duration_seconds=duration,
                )

            except RateLimitError as e:
                wait = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit on {processed.file_name}, "
                    f"retry {attempt + 1}/{self.max_retries} in {wait:.1f}s"
                )
                time.sleep(wait)
                last_error = f"Rate limit: {e}"

            except APITimeoutError as e:
                wait = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Timeout on {processed.file_name}, "
                    f"retry {attempt + 1} in {wait:.1f}s"
                )
                time.sleep(wait)
                last_error = f"Timeout: {e}"

            except APIError as e:
                last_error = f"API error: {e}"
                logger.error(f"API error on {processed.file_name}: {e}")
                break

            except Exception as e:
                last_error = f"Unexpected: {e}"
                logger.exception(f"Unexpected error on {processed.file_name}")
                break

        return ExtractionResult(
            file_name=processed.file_name,
            success=False,
            extraction=None,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            duration_seconds=0.0,
            error=last_error,
        )
