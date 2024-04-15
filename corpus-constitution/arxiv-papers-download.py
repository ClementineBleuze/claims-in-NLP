import pandas as pd
import os
import json
import tqdm
from subprocess import run

# directory where the papers will be downloaded
output_dir = "../data/arxiv/pdf/"
# store the encountered errors (e.g a pdf file is not available)
error_path = "../data/arxiv/pdf_errors.txt"

error_ids = []
with open(error_path, "r") as f:
    for line in f:
        error_ids.append(line.split(",")[0])

# chargement des metadata arxiv pour l'accès aux identifiants des articles
df = pd.read_csv("../data/arxiv/arxiv-metadata-nlp-unpublished.csv", encoding = "utf-8")
print(f"{len(df[df['pdf_stored'] == False])} pdf files to download")

for i in tqdm.tqdm(range(df.shape[0])):

    # skip if the pdf has already been stored
    if df.loc[i, "pdf_stored"] == True:
        continue

    # get the full id of the paper and deduce its associated pdf name
    full_id = str(df.loc[i, "id"])

    # skip if the pdf has already caused an error
    if full_id in error_ids or "0"+full_id in error_ids:
        continue

     # depending on the id format, the pdf file will be named differently
    if "." in full_id:
        date_code = full_id.split(".")[0]
        if len(date_code) == 3:
            date_code = "0" + date_code
            full_id = "0" + full_id
        pdf_path= f"gs://arxiv-dataset/arxiv/arxiv/pdf/{str(date_code)}/{str(full_id)}"

    elif "/" in full_id:
        category, id = full_id.split("/")[:2]
        date = id[:4]
        pdf_path = f"gs://arxiv-dataset/arxiv/{str(category)}/pdf/{str(date)}/{str(id)}"
        
    else:
        with open(error_path, "a") as f:
            f.write(f"{full_id}, failed: id format not recognized\n")

    # get the different versions of the paper
    versions = json.loads(df.loc[i, "versions"].replace("\'", "\""))

    res_code = 1 # 1: not found, 0: found

    for version in versions[::-1]:
        v = version["version"]
        pdf_full_path = f"{pdf_path}{str(v)}.pdf"

        # we try to download the pdf file
        res_code = run(["gsutil", "cp", pdf_full_path, output_dir], shell = True, check=False).returncode

        if res_code == 0:
            df.at[i, "pdf_stored"] = True
            break
    
    # if the pdf file has not been found, we store the error
    if res_code == 1:
        with open(error_path, "a") as f:
            f.write(f"{full_id}, failed: no PDF available \n")
    
    # save the updated metadata regularly
    if i % 100 == 0:
        df.to_csv("../data/arxiv/arxiv-metadata-nlp-unpublished.csv", index=False, encoding = "utf-8")

