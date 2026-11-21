# BERT SMM4H Sentiment Classification

## Project overview

This project applies transfer learning to sentiment analysis of short social-media posts from the **Social Media Mining for Health (SMM4H) 2024 Task 5** dataset. The goal is to automatically classify each post into one of two sentiment categories:

- **Negative** (`0`)
- **Positive** (`1`)

The project uses the pretrained `bert-base-uncased` language model. BERT is well suited to this task because it learns contextual relationships between words rather than treating each word independently. The pretrained model is adapted to the sentiment-classification task by adding a two-class classification head and fine-tuning it on the labeled SMM4H training data.

The notebook provides a complete machine-learning workflow:

1. Load labeled training and validation data from TSV files.
2. Clean noisy social-media text, including URLs, mentions, hashtags, HTML entities, and extra whitespace.
3. Convert the cleaned text into BERT-compatible tokens.
4. Fine-tune BERT using the training dataset.
5. Evaluate the model on the validation dataset using the F1 score.
6. Generate predicted sentiment labels and positive-class probabilities.

This project is intended as an educational and experimental baseline for applying transformer-based NLP models to health-related social-media text. It can be extended with additional preprocessing, hyperparameter tuning, larger pretrained models, test-set evaluation, or deployment as a sentiment-prediction service. The complete implementation is available in [`BERT_SMM4H.ipynb`](BERT_SMM4H.ipynb).

## What the notebook does

1. Loads the SMM4H training and validation TSV files.
2. Removes the `tweet_id` column.
3. Cleans tweet text by:
   - converting text to lowercase,
   - removing URLs,
   - removing mentions and hashtag symbols,
   - unescaping HTML entities, and
   - normalizing whitespace.
4. Tokenizes the text with the BERT tokenizer.
5. Fine-tunes BERT for two classes:
   - `0`: Negative
   - `1`: Positive
6. Evaluates the model using F1 score.
7. Prints validation predictions, class-1 logits, and normalized class probabilities.

## Project structure

```text
.
├── BERT_SMM4H.ipynb
├── SMM4H-2024-Task5-Training.tsv      # required input
└── SMM4H-2024-Task5-Validation.tsv    # required input
```

The TSV files are not included in this project. Place them in the notebook's working directory before running the data-loading cells.

## Requirements

- Python 3
- A CUDA-enabled GPU is recommended
- Jupyter Notebook or Google Colab
- PyTorch
- pandas
- NumPy
- Hugging Face `datasets`
- Hugging Face `transformers`
- scikit-learn
- `evaluate`
- TensorBoard (optional, for training logs)

Install the Python dependencies with:

```bash
pip install torch pandas numpy datasets transformers scikit-learn evaluate tensorboard
```

## Running in Google Colab

1. Open `BERT_SMM4H.ipynb` in Google Colab.
2. Enable a GPU runtime:
   **Runtime > Change runtime type > T4 GPU**.
3. Upload or copy the following files into the working directory:

   ```text
   SMM4H-2024-Task5-Training.tsv
   SMM4H-2024-Task5-Validation.tsv
   ```

4. Update the Google Drive path in the second cell if necessary:

   ```python
   %cd /content/drive/MyDrive/BERT_SMM4H
   ```

5. Run the cells from top to bottom. The notebook downloads the BERT tokenizer and model from the Hugging Face Hub on first use.

## Training configuration

The current notebook uses:

| Setting | Value |
|---|---:|
| Base model | `bert-base-uncased` |
| Batch size | 8 |
| Learning rate | `7.21422e-06` |
| Epochs | 3 |
| Weight decay | `0.00694763` |
| Maximum sequence length | 512 |
| Best-model metric | F1 score |
| Mixed precision | Enabled when CUDA is available |

Training checkpoints are written to `smm4h_bert_base_clean_v3`. At most two checkpoints are retained, and the best checkpoint is restored at the end of training.

## Output

The final notebook cell uses the best validation model to produce:

- predicted class labels,
- class-1 prediction values, and
- normalized probabilities for the positive class.

The Hugging Face `Trainer` also reports the validation F1 score at the end of each epoch. TensorBoard logs are enabled through the training arguments.

## Notes

- The notebook currently assumes the input columns include `tweet_id`, `text`, and the target label column expected by the dataset.
- `NUM_PROCS` is set to `32` for tokenization. Reduce this value if the runtime has fewer available CPU cores.
- The model and tokenizer are downloaded automatically, so internet access is required the first time they are loaded.
