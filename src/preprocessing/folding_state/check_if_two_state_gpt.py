#check_if_two_state_gpt.py
#this script will use ChatGPT API to check if a given protein is two-state folding based on the PDF of a publication

from openai import AsyncOpenAI
from pathlib import Path
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import asyncio

fireprot_csv = "./data/fireprotdb_ddg_single_pub_with_pdf_scores_num_manual_cleaned4.csv"
model = "gpt-5-nano"

client = AsyncOpenAI(
    api_key="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"
)

SEM = asyncio.Semaphore(5)

SYSTEM_PROMPT = '''You are an expert in protein thermodynamics. 
                    You must classify folding models strictly by explicit statements in the paper. 
                    If the paper does not explicitly mention two-state or three-state folding, return 'unknown'. Never infer.  
                    Please respond with only one of the following options: 'two-state', 'three-state', or 'unknown'.'''

async def upload_pdf(path: Path):
    async with SEM:
        file = await client.files.create(
            file=path.open("rb"),
            purpose="assistants"
        )
        return file.id

def get_pdf_path_for_doi(doi: str):
    if not isinstance(doi, str):
        return None
    doi = doi.strip()
    if doi == "" or doi.lower() == "nan":
        return None
    pdf_path = Path("./data/pdfs") / (doi.replace('/', '_') + '.pdf')
    return pdf_path


#loop through each unique protein in dataframe, go to first DOI, upload PDF, ask GPT, and if result is unknown, go to next DOI for that protein until all DOIs are exhausted or a non-unknown result is found

async def folding_model_query(file_id: str, doi: str, model: str):
    async with SEM:
        resp = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    
                    {"type": "input_file", "file_id": file_id}
                ]}
            ]
        )
        result = resp.output_text.strip().lower()
        
        if result in ["two-state", "three-state", "unknown"]:
            return result

        return "unknown"

async def classify_model(protein, dois, df):
    for doi in dois:
        pdf_path = get_pdf_path_for_doi(doi)
        if pdf_path is None:
            print(f"DOI for protein {protein} is invalid, skipping...")
            continue
        
        if not pdf_path.exists():
            print(f"PDF for DOI {doi} not found, skipping...")
            continue
    
        try: 
            file_id = await upload_pdf(pdf_path)
        except:
            print(f"Failed to upload PDF for DOI: {doi}")
            continue

        try:
            label = await folding_model_query(file_id, doi, model)
        except:
            print(f"Failed to get folding model for DOI: {doi}")
            continue

        if label != "unknown":
            df.loc[df["PROTEIN"] == protein, 'FOLDING_MODEL'] = label
            df.loc[df["PROTEIN"] == protein, 'FOLDING_MODEL_SOURCE_DOI'] = doi
            return 

    df.loc[df["PROTEIN"] == protein, 'FOLDING_MODEL'] = "unknown"
    df.loc[df["PROTEIN"] == protein, 'FOLDING_MODEL_SOURCE_DOI'] = None
    return 

async def main():
    df = pd.read_csv(fireprot_csv) 
    protein_to_dois = (df.groupby("PROTEIN")["PUBLICATION_DOI"].apply(lambda s: list(s.unique())).to_dict())


    for protein, dois in tqdm_asyncio(protein_to_dois.items(), desc="Classifying proteins"):
        await classify_model(protein, dois, df)

    df.to_csv("./data/fireprotdb_ddg_with_folding_label.csv", index=False)
    print("Done.")

asyncio.run(main())