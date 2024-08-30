# ML CRM Project

This project implements machine learning models for predicting customer churn, health scores, expansion opportunities, and upsell potential based on CRM data. It uses PyTorch for model implementation, ensemble learning techniques, SHAP for explainability, and LangChain for generating human-readable explanations.

## Project Structure

- `data/`: Contains raw and processed data
- `models/`: Stores trained model checkpoints
- `notebooks/`: Jupyter notebooks for exploratory data analysis
- `src/`: Source code for the project
  - `data/`: Data loading and processing
  - `features/`: Feature engineering, text analysis, and feature selection
  - `models/`: PyTorch model implementations, including ensemble models
  - `utils/`: Helper functions, explanation agent, and model evaluation tools
  - `api/`: FastAPI implementation for model serving
- `tests/`: Unit tests for the models
- `config.yaml`: Configuration file for the project
- `requirements.txt`: Python dependencies

## Key Features

- Ensemble learning using stacking for improved model performance
- Advanced feature selection techniques
- Hyperparameter tuning for optimal model configuration
- Anomaly detection for identifying unusual account behavior
- Comprehensive model evaluation and metrics calculation
- SHAP-based model explainability
- LangChain-powered natural language explanations of model predictions

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ml-crm-project.git
   cd ml-crm-project
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix or MacOS: `source venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Set up your OpenAI API key:
   - Create a copy of `config.yaml.example` and name it `config.yaml`
   - Replace `your_openai_api_key_here` with your actual OpenAI API key

## Usage

1. Prepare your data:
   - Place your raw CRM, market, and usage data CSV files in the `data/raw/` directory
   - Update the file paths in `config.yaml` if necessary

2. Run data processing and feature engineering:
   ```
   python src/data/data_processor.py
   ```

3. Train the models:
   ```
   python src/models/train_models.py
   ```

4. Start the API server:
   ```
   uvicorn src.api.main:app --reload
   ```

5. Make predictions:
   - Send POST requests to `http://localhost:8000/predict` with account data
   - Refer to the API documentation for request and response formats

## Running Tests

Run the unit tests using:
```
python -m unittest discover tests
```
## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Running Jupyter Notebooks

To easily start a Jupyter Lab server and run the notebooks, follow these steps:

1. Make sure you have installed all dependencies using Poetry:
   ```
   poetry install
   ```

2. Start the Jupyter Lab server:
   ```
   poetry run jupyter
   ```

3. Your default web browser should open automatically with the Jupyter Lab interface. If it doesn't, copy and paste the URL displayed in the terminal into your browser.

4. Navigate to the `notebooks` directory in the Jupyter Lab interface to access and run the project notebooks.