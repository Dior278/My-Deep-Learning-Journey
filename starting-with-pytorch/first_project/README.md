# Final project: fake vs real news

We classify news text as fake (`0`) or real (`1`). First a plain MLP on raw TF-IDF, then we strip source names / punctuation / URLs and train again with batch norm and dropout.

## Data

Download [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) from Kaggle:

```text
final_project/data/Fake.csv
final_project/data/True.csv
```

## Run

From this folder, venv activated:

```bash
jupyter notebook code.ipynb
```

