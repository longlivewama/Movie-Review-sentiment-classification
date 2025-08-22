# IMDB Movie Review Sentiment Analysis

## Overview
This project is an end-to-end NLP application for classifying IMDB movie reviews as positive or negative using a fine-tuned BERT model. It includes:
- A Jupyter Notebook (`IMDB_Movie_Review_cls(NLP).ipynb`) for data loading, model training, evaluation, and inference testing.
- A Streamlit web app (`bert_sentiment_app.py`) for interactive sentiment prediction with a user-friendly interface.
- Pre-trained model files (tokenizer configs, vocab, etc.) for the fine-tuned BERT model.

The model was fine-tuned on a subset of the IMDB dataset (5,000 training samples, 1,000 test samples) using Hugging Face's Transformers library, achieving approximately 90% accuracy. It handles text truncation, padding, and softmax for confidence scores.

### Features
- **Training Notebook**:
  - Loads IMDB dataset from Hugging Face Datasets.
  - Tokenizes reviews using BERT tokenizer.
  - Fine-tunes BERT-base-uncased for sequence classification.
  - Evaluates with accuracy, classification report, and confusion matrix.
  - Includes an inference function for custom text predictions.
- **Streamlit App**:
  - Custom background image and dark overlay for a cinematic feel.
  - Text area for user input.
  - Analyze button to predict sentiment with confidence bar.
  - Reset button to clear input.
  - Sidebar with positive/negative example reviews.
- **Model**: Saved in `./fine-tuned-bert-imdb` (loaded from local path in the app).

## Prerequisites
- Python 3.8+
- GPU recommended for training (e.g., Google Colab with T4 GPU).
- Install dependencies from `requirements.txt`.

## Installation
1. Clone the repository:
   ```
   git clone https://your-repo-url.git
   cd your-repo-folder
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the fine-tuned model (if not included) or train it via the notebook.
4. Ensure the background image (`IMDB.avif`) and model path (`D:\NLP\Movie Review sentiment classification\fine-tuned-bert-imdb`) are accessible or update paths in `bert_sentiment_app.py`.

## Usage

### Training the Model
1. Open `IMDB_Movie_Review_cls(NLP).ipynb` in Jupyter Notebook or Google Colab.
2. Run cells sequentially to load data, train, evaluate, and save the model to `./fine-tuned-bert-imdb`.
3. Test inference with sample reviews at the end.

### Running the Streamlit App
1. Update paths in `bert_sentiment_app.py` if needed (e.g., BG_PATH and MODEL_PATH).
2. Run the app:
   ```
   streamlit run bert_sentiment_app.py
   ```
3. Open the app in your browser (usually http://localhost:8501).
4. Enter a movie review, click "Analyze" to see the sentiment and confidence.
5. Use sidebar examples or "Reset" as needed.

## Project Structure
- `bert_sentiment_app.py`: Streamlit app script.
- `IMDB_Movie_Review_cls(NLP).ipynb`: Training and evaluation notebook.
- `special_tokens_map.json`, `tokenizer_config.json`, `vocab.txt`, `tokenizer.json`: Model tokenizer files.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## Results
- **Accuracy**: ~90% on test set.
- **Example Predictions**:
  - "This movie was absolutely fantastic!" → Positive (Confidence: 0.998)
  - "Terrible movie, waste of time." → Negative (Confidence: 0.997)

## Limitations
- Trained on a small subset for efficiency; full dataset could improve performance.
- App assumes local model paths; containerize for production (e.g., Docker).
- No real-time API; extend with FastAPI if needed.

## Future Improvements
- Integrate larger datasets or multilingual support.
- Add explainability (e.g., SHAP for token importance).
- Deploy to cloud (Heroku, Streamlit Sharing).

