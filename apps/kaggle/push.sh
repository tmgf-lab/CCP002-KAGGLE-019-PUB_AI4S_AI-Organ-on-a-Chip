#!/usr/bin/env bash
# Publish notebooks/walkthrough.ipynb as the public Kaggle notebook (needs the Kaggle CLI and an API token).
# The notebook clones this repository, so the GitHub repository must be public first.
set -e
cd "$(dirname "$0")"
cp ../../notebooks/walkthrough.ipynb .
kaggle kernels push -p .
rm walkthrough.ipynb
