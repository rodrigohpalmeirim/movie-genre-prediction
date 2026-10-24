# Description

Predicting a movie's genre from its plot summary. The notebook compares two classic text classifiers with a fine-tuned transformer.

The dataset (`data/movie_plots.csv`) has 17,096 movies, each with a release year, title, genre and plot summary. There are 10 genres: drama, comedy, horror, action, thriller, romance, western, sci-fi, crime and adventure. 20% of the data is held out for testing.

# Models
- **Naive Bayes**: multinomial Naive Bayes on TF-IDF features (one-vs-rest)
- **Linear SVM**: calibrated `LinearSVC` on TF-IDF features (one-vs-rest)
- **DistilBERT**: `distilbert-base-uncased` fine-tuned for 3 epochs with Hugging Face Transformers and TensorFlow

# Results
Test accuracy as recorded in the notebook:

| Model | Accuracy |
| --- | --- |
| Naive Bayes | 0.54 |
| Linear SVM | 0.52 |
| DistilBERT | 0.87 |

The notebook also includes confusion matrices and per-genre classification reports. Note that the train/test split is not seeded. When the saved DistilBERT weights are loaded, the test set may therefore overlap with the data the model was fine-tuned on.

# Usage
The fine-tuned weights (`models/distilbert_finetuned.h5`) are stored with Git LFS:
```sh
git lfs pull
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow transformers
jupyter notebook movie_genre_prediction.ipynb
```
To fine-tune DistilBERT from scratch instead of loading the saved weights, set `retrain = True`.
