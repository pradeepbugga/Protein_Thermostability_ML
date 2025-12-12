#worker_mutation_gpt.py
#this script will use ChatGPT API to extract mutations from PDFs of publications listed in FireProtDB
#this script runs a worker for a specific shard of DOIs
#and is executed via main_mutation_gpt.py

from openai import AsyncOpenAI, RateLimitError
from pathlib import Path
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import asyncio
import argparse
import json

fireprot_csv = "./data/fireprotdb_ddg_two_state_only.csv"
model = "gpt-5-nano"

client = AsyncOpenAI()

SEM = asyncio.Semaphore(10)

SYSTEM_PROMPT = '''You are an expert in protein mutation nomenclature.

Your task:
Given a publication (PDF) on a protein's thermal stability, extract ALL mutations mentioned anywhere in text, figures, or tables.  Mutations can be described in various formats, including but not limited to:
- Single-letter amino acid codes (e.g., A123C)
- Three-letter amino acid codes (e.g., Ala123Cys)
- Descriptive phrases (e.g., "alanine at position 123 mutated to cysteine

Return ONLY a JSON list of canonical one-letter mutation strings.
Examples: N9A, Y42F, S64A, A17G, M100K.

Rules:
- Convert 3-letter codes to 1-letter.
- Ignore non-amino-acid tokens.
- Include mutations from tables.
- Do NOT infer; only report explicit statements.
- If no mutations are found, return an empty list [].'''

async def with_retry(coro, retries=5):
    for i in range(retries):
        try:
            return await coro              # <-- KEY FIX
        except (RateLimitError, APIError) as e:
            wait = 2 ** i + random.random()
            print(f"Rate limit/API error: {e}. Retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)
    raise Exception("Too many retries, giving up.")


async def upload_pdf(path: Path):
    async with SEM:
        print(f"Uploading PDF: {path.name}")
        file = await client.files.create(
            file=path.open("rb"),
            purpose="assistants"
        )
        print(f"Uploaded PDF: {path.name}, file ID: {file.id}")
        return file.id


def get_pdf_path_for_doi(doi: str):
    if not isinstance(doi, str):
        return None
    doi = doi.strip()
    if doi == "" or doi.lower() == "nan":
        return None
    pdf_path = Path("./data/pdfs") / (doi.replace('/', '_') + '.pdf')
    return pdf_path

async def mutation_query(file_id: str, model: str):
    async with SEM:
        
        resp = await with_retry(client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "input_file", "file_id": file_id}
                ]}
            ]
        ))
        result = resp.output_text.strip()    

        try: 
            #try to parse as list
            muts_list = json.loads(result)
            if isinstance(muts_list, list):
                return muts_list
        except:
            pass
        
        return result

async def extract_mutations(doi, df):  
    
    pdf_path = get_pdf_path_for_doi(doi)
    if pdf_path is None:
        print(f"Invalid DOI {doi}, skipping...")
        return doi, []
    
    if not pdf_path.exists():
        print(f"PDF for DOI {doi} not found, skipping...")
        return doi, []

    try: 
        file_id = await upload_pdf(pdf_path)
    except Exception as e:
        print(f"Failed to upload PDF for DOI: {doi}")
        print(e)
        return doi, []

    try:
        print(f"Querying mutations for DOI: {doi}")
        paper_muts = await mutation_query(file_id, model)
        print(f"Received mutations for DOI {doi}: {paper_muts}")
    except Exception as e:
        print(f"GPT failed for DOI: {doi}")
        print(e)
        return doi, []

    df.loc[df["PUBLICATION_DOI"] == doi, "EXTRACTED_MUTATIONS"] = str(paper_muts)

    return doi, paper_muts
        
async def main(shard_id):
    df = pd.read_csv(fireprot_csv)
    df['PUBLICATION_DOI'] = df['PUBLICATION_DOI'].fillna('').astype(str)

    with open(f"./data/shards/shard_{shard_id}.txt") as f:
        unique_dois = [line.strip() for line in f.readlines() if line.strip()]

    results = {}
    for doi in tqdm_asyncio(unique_dois, desc="Processing DOIs"):
        doi,muts = await extract_mutations(doi, df)
        results[doi] = muts

    outpath = f"./data/fireprotdb_ddg_mutations_extracted_shard_{shard_id}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Worker shard {shard_id} done. Results saved to {outpath}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mutations from PDFs using GPT.")
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.shard))