#convert_pdf_score.py
#this script takes the GPT extracted PDF_PROTEIN_MATCH_SCORE column and converts it to a numeric value in a new column PDF_PROTEIN_MATCH_SCORE_NUM
#note: with better prompting, we could have extracted numeric values directly, but this is a quick fix
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

fireprot_csv = "./data/fireprotdb_ddg_single_pub_with_pdf_scores.csv"

df = pd.read_csv(fireprot_csv)

for index, row in df.iterrows():
    
    score = str(df.at[index,'PDF_PROTEIN_MATCH_SCORE'])
    #extractt numeric value from beginning of score string

    match = re.match(r"(\d+)", score)
    if match:
        numeric_score = int(match.group(0))
        df.at[index, 'PDF_PROTEIN_MATCH_SCORE_NUM'] = numeric_score

df.to_csv("./data/fireprotdb_ddg_single_pub_with_pdf_scores_num.csv", index=False)

