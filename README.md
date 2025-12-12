# Protein Thermostability Prediction with Machine Learning

## Regressing ESM-2 Embeddings with Experimental ∆∆G Values from a Public Database

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
  This project takes advantage of recently developed protein language models (PLMs), namely ESM-2, and integrates it with FireProt DB (https://loschmidt.chemi.muni.cz/fireprotdb/), a publicly available dataset of thermodynamic data.  More specifically, we build a machine learning model
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

### 4. Repository Structure

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
    ├── inference
    │   ├── sample_inference.py
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
The preprocessing sub-folder has scripts for the ingestion and filtering of of the publicly available FireProt DB csv file.  We first note that these preprocessing scripts were not 100% foolproof, and did require manual correction on a small subset 
of the data.  We also note that for mutation extraction, we used CPU multiprocessing with a main and worker script (21 workers) to reduce inference time from a few hours to ~20 minutes.

The embeddings sub-folder contains all logic for creating the raw ESM-2 embeddings along with associated features (i.e. delta embeddings).

The individual files in the src folder (build_records,py, embeddings_loader.py, features.py, dataset_buider.py, mutation_record.py, and protein_grouping.py) contain the logic for three training-relevant tasks: 1) building a data class record for every FireProt         entry containing mutation, sequence, ddG, and embedding path information, 2) loading the tensors for the regression models, and 3) grouping the proteins for LOPO cross-validation.

The training sub-folder contains logic for training different models in combination with different cross-validation and feature inputs. 

Finally, the analysis subfolder contains logic for visualizing the RMSE results.

### 5. Installation

```
    git clone https://github.com/pradeepbugga/Protein_Thermostability_ML.git
    cd Protein_Thermostability_ML
    pip install -r requirements.txt
```
  This project was tested on Python 3.12.3 and run on an A40 GPU (RunPod) with 48 GB VRAM and 48GB RAM.

### 5. Data Preparation

  This repo does not include the curated FireProt CSV containing protein sequence, mutation, and ∆∆G information.  I am happy to share this CSV file upon request. 

### 6. Results Summary

  - MLP Model (three layers with ReLU and dropout)  (note: XGBoost was far inferior) 
  - Used delta per-residue embedding feature (i.e. the embedding vector for the mutant sequence at the specific mutant residue minus embedding vector for the wild type sequence at the specific residue)
  - Used LOPO Cross-Validation with a minimum value of 10 (i.e. all proteins with >10 mutation entries are in individual folds, while proteins with <10 mutations are grouped together)

  <p align="center">
  <img width="2011" height="611" alt="image" src="https://github.com/user-attachments/assets/db781277-77ae-41b4-be35-f148c2e98c69" />
  </p>

  We see from the above plot that our MLP can predict ∆∆G from ESM-2 embeddings with RMSE <0.7, but significant RMSE variation among folds.  In some part, this is to be expected as there is not only
  experimental variability in the buffer, pH, temperature, and method used to obtain ∆∆G, but also biological variability in the selection of proteins and associated mutations.  We pay special attention 
  to folds with the highest RMSEs : 2 and 5.  Fold 2 corresponds to thermonuclease, which has >600 mutations.  This protein was frequently used in thermal stability research in the '90s and bears both variable experimental approaches to extracting 
  thermodynamic data and even variable data for the same mutation measured multiple times by different groups.  Carefully looking at the experimental conditions used for this entry would help us
  explain the observed RMSE.  Fold 5 on the other hand is the "other" category containing all proteins that did not meet the minimum threshold of "10" mutation entries.  It is certainly expected that 
  predictions on a heterogeneous set of proteins would be large.  

  Different ESM-2 layers correspond to extraction of different features of the protein, with the earliest layers representing more of the raw amino acid sequence while the later layers representing more of the 
  three-dimensional architecture.  Trying layer 25 instead of layer 33 shows marginal change, suggesting layers 25-33 do not contribute significantly to prediction.
  
  <p align="center">
  <img width="2011" height="611" alt="image" src="https://github.com/user-attachments/assets/09dd7134-d610-4982-84d5-d34c3ec10d83" />
  </p>  

  Nevertheless, this result is an improvement over previous approaches considering we achieve low RMSEs on a biologically realistic LOPO cross-validation method.  This is perhaps a testament to the power of 
  large protein language models (ESM-2) and the use of actual experimental data rather than poorer proxies.

### 7. Ongoing Efforts

  Additional areas of improvement include more careful curation of experimental data (removing entries where pH is << or >> physiological pH), hyperparameter tuning, feature engineering (do local environment embeddings help?), and finally inference on the above thermostable mutations from TdT to see if our model can truly offer a benefit to protein engineers.

On that last point, below is our initial inference on these previously identified mutations(left).
  <p align="center">
  <img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/11c7f5ef-22e9-4721-a5da-4833301bd092" />

<img width="500" height="300" alt="masked_likelihood" src="https://github.com/user-attachments/assets/99225c15-f61b-485a-87f2-5c58fd56ed9d" />
</p>
  Comparing this result to the previous "ESM-2 only" delta log-likelihood values (right), we notice that mutation M238K, which ESM-2 found to be significantly destabilizing, now shows up with a positive ∆∆G value ("stabilizing") from inference.  This observation is perhaps the best validation of our approach, because it highlights raw ESM-2's limitation in identifying non-natural mutations that can engineer higher thermostability and our combined ESM-2 embedding / ∆∆G MLP regression model's ability to overcome that limitation.   
  

### 8. Limitations

  Our input dataset contains experimental measurements going back forty years and spanning old and new techniques for calculating ∆∆G.  In addition, these assays themselves have a standard deviation that can affect our model features.
  That being said, the entire field of AI for biology suffers from a lack of sufficient experimental data, so this is likely the best we can do.
  Finally, our method specifically filtered for single mutants to create the simplest conditions for developing a well-performing model.  In practice, multiple mutations are combined in the same protein to achieve additive improvements 
  in  thermal stability.  Expanding our approach to multi- mutants will be an important next step.

### 9. License

  This project is released under the MIT License.
  






  
  

  

