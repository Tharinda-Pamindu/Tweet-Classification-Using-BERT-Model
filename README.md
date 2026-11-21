# BERT SMM4H Sentiment Classification

This project fine-tunes `bert-base-uncased` for binary sentiment classification on the SMM4H 2024 Task 5 dataset. The complete workflow is provided in [`BERT_SMM4H.ipynb`](BERT_SMM4H.ipynb).

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
