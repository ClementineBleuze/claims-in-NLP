# ArXiv corpus description

The metadata comes from the [ArXiv corpus](https://www.kaggle.com/datasets/Cornell-University/arxiv/data), hosted on Kaggle by Cornell University, containing 2.4M+ article up to 2024 with monthly updates. 

Then, PDF files have been downloaded from the dedicated Google Cloud Storage Bucket using `gsutil`. Finally, the PDF files have been parsed in XML files using [GROBID](https://github.com/kermitt2/grobid_client_python)
