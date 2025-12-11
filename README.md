# Protein Thermostability Prediction with Machine Learning
## Regressing ESM-2 Embeddings with Experimental ∆∆G Values from a Publicly Available Database

<p align="center">
<img width="600" height="336" alt="image" src="https://github.com/user-attachments/assets/3e84c6bd-a093-43b4-9f70-0519e07ecd9f" />
</p>

---

### 1. Problem Statement
  Enzymes that have increased thermal stability are of significant importance to bio-industrial processes where biochemical reactions performed at higher temperatures can increase reactor productivity 
  (the amount of commercial product produced per unit time) and overcome limited solubility of starting materials.  Traditional approaches to improving this property involves protein engineering, where hundreds (if not thousands)
  of proteins must be individually expressed from custom DNA sequences, then screened in assays measuring the temperature (Tm) where the protein unfolds.  Machine learning, especially deep learning, holds the potential for
  expediting this otherwise time-consuming and laborious wet-lab process and predict thermal stability much like AlphaFold assists in structure prediction.

### 2. Project Overview
  This project takes advantage of recently developed protein language models (PLMs), namely ESM-2, and integrates it with FireProt DB, a publicly available dataset of thermodynamic data.  More specifically, we build a machine learning model
  to predict thermal stability changes (∆∆G) from single-point mutations.  The pipeline entails curated measurements from the FireProt database, ESM-2 protein embeddings, and leave-out-one-protein (LOPO) cross-validation.  
  
  ESM protein language models inherently learn what amino acid sequences are evolutionarily preferred and are therefore well-suited for identifying destabilizing mutations.  However, these models struggle with identifying engineered,
  stabilizing mutations that endow proteins with properties (i.e. thermal stability at 50 °C) not usually found in nature.  As a quick note, we show below that known engineered mutations that confer thermostability to a sample protein 
  (TdT polymerase) cannot be predicted solely on the basis of ESM-2 log-likelihood differences.  We therefore hypothesize that we can take the information contained in the ESM-2 feature vectors, then perform 
  regression with experimental sequence - thermodynamic value pairs.   While the FireProt database has been used with a number of traditional ML models and structure-based approaches, this is (to my knowledge) the first effort exploring 
  the potential of sequence-based protein language models for thermostability prediction.
  
<p align="center">
<img width="500" height="300" alt="masked_likelihood" src="https://github.com/user-attachments/assets/99225c15-f61b-485a-87f2-5c58fd56ed9d" />
</p>

### 3. Key Features
  - Pre-processed / cleaned the FireProt dataset by using the multi-modal GPT 5 API and publication reference PDFs to filter for single mutations and proteins with two-state folding mechanisms
  - Also used the GPT 5 API to extract mutations as noted in references then align with FireProt nomenclature
  - Generated ESM-2-ready sequences manually by integrating UniProt protein entries with information in references 
  - Generated ESM-2 (3B parameter model) embeddings for wild-type, mutant, and pooled representations using layer 33 (other layers currently being investigated)
  - Feature engineering with delta per-residue embeddings, delta pooled embeddings, and raw wild-type/mutant embeddings
  - LOPO cross-validation to ensure regression model can generalize to new proteins and minimize data leakage
  - Using an MLP regressor (which was superior to XGBoost) on embedding features and ∆∆G values
  - Full modular codebase: preprocessing -> embedding -> feature building -> model training -> inference (model inference currently being investigated)

### Repository Structure
```
├── requirements.txt
├── README.md
└── src
    ├── __init__.py
    ├── analysis
    │   ├── compare_rmse.py
    │   └── visualize_rmse.py
    ├── build_records.py
    ├── dataset_builder.py
    ├── embedding_loaders.py
    ├── embeddings
    │   ├── build_delta_per_residue_embeddings.py
    │   ├── build_delta_pooled_embeddings.py
    │   ├── build_sample_delta_embeddings.py
    │   ├── compute_embeddings.py
    │   ├── delta_builder.py
    │   ├── embed_sequence.py
    │   ├── esm_model_loader.py
    │   └── save_embeddings.py
    ├── features.py
    ├── load_dataset.py
    ├── models
    │   ├── __init__.py
    │   └── mlp.py
    ├── mutation_record.py
    ├── preprocessing
    │   ├── folding_state
    │   │   ├── check_if_two_state_gpt.py
    │   │   └── two_state_filter.py
    │   ├── generate_esm_sequences.py
    │   ├── ingest_fireprot_db.py
    │   ├── initial_filter_db.py
    │   ├── mutations
    │   │   ├── align_mutations.py
    │   │   ├── combine_shards.py
    │   │   ├── fix_mayo_paper.py
    │   │   ├── main_mutation_gpt.py
    │   │   └── worker_mutation_gpt.py
    │   ├── pdf_reference
    │   │   ├── confirm_ref_pdf_gpt.py
    │   │   └── convert_pdf_score.py
    │   └── protein_sequence_mapping.py
    ├── protein_grouping.py
    └── training
        ├── __init__.py
        ├── loops.py
        ├── train_mlp.py
        ├── train_mlp_lopo.py
        ├── train_xgboost_lopo.py
        └── train_xgboost_lopo_multifeature.py
```

  
  

  

