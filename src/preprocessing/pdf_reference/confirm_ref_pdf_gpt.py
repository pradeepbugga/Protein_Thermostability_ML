#confirm_ref_pdf_gpt.py
#this script will use ChatGPT API to check if the downloaded PDFs correspond to the correct publications based on their DOIs

from openai import AsyncOpenAI
from pathlib import Path
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import asyncio

fireprot_csv = "./data/fireprotdb_ddg_single_pub.csv"
model = "gpt-5-nano"
pdf_path = Path("./data/pdfs")

df = pd.read_csv(fireprot_csv)
client = AsyncOpenAI()

SEM = asyncio.Semaphore(5)

SYSTEM_PROMPT = '''You are a biochemistry expert and an expert at verifying the contents of a scientific paper gives its PDF.  
                Based on the given paper input (in the PDF attachment) and the given protein name input, please return a value 
                between 0 and 100 corresponding to the confidence you have that the paper indeed discusses the corresponding protein.  
                100 = maximum confidence, 0 = minimal confidence that paper discusses said protein.  If confidence is low, please double-check alternative names or synonyms for the protein.'''


async def upload_pdf(path: Path):
    async with SEM:
        file = await client.files.create(
            file=path.open("rb"),
            purpose="assistants"
        )
        return file.id


async def score_protein(file_id: str, doi: str, protein: str, model: str):
    async with SEM:
        resp = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "input_text", "text": f"Protein: {protein}"},
                    {"type": "input_file", "file_id": file_id}
                ]}
            ]
        )
        score = resp.output_text.strip()
        return protein, score



async def process_pdf(path: Path, df: pd.DataFrame):
    doi = path.stem.replace('_', '/')
    entries = df[df["PUBLICATION_DOI"] == doi]

    if entries.empty:
        print(f"No entries found for DOI {doi}, skipping...")
        return df

    unique_proteins = entries["PROTEIN"].unique()

    # Upload PDF once
    file_id = await upload_pdf(path)

    # Dispatch async scoring tasks for all proteins
    tasks = [
        score_protein(file_id, doi, protein, model)
        for protein in unique_proteins
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Write scores back into df
    for protein, score in results:
        df.loc[
            (df["PUBLICATION_DOI"] == doi) &
            (df["PROTEIN"] == protein),
            "PDF_PROTEIN_MATCH_SCORE"
        ] = score
        print(f"DOI: {doi}, Protein: {protein}, Score: {score}")

    return df


async def main():
    df = pd.read_csv(fireprot_csv) 
    pdfs = list(pdf_path.glob("*.pdf"))

    for path in tqdm_asyncio(pdfs, desc="Processing PDFs"):
        df = await process_pdf(path, df)

    df.to_csv("./data/fireprotdb_ddg_single_pub_with_pdf_scores.csv", index=False)
    print("Done.")

asyncio.run(main())