### AI-Powered Transaction Fraud Detection Engine

A backend system that processes transactions, scores them for fraud risk using an unsupervised machine learning model, and prioritizes them using a hand-built max-heap priority queue. 

### How it Works

1. **AI Scoring:** The FastAPI gateway receives raw transaction data and uses an IsolationForest model to assign an anomaly risk score from 0.0 to 1.0.
2. **Custom Queueing:** The transaction is pushed into an in-memory custom max-heap priority queue, sorting items strictly by risk.
3. **Processing Core:** Calling /process-next pops the single highest-risk item from the top of the queue for operational review.

### Technical Highlights

* **Unsupervised ML:** Uses an IsolationForest to catch anomalies without needing massive labeled historical fraud datasets.
* **Hand-rolled Data Structure:** Implements a custom binary max-heap with an internal counter to resolve tie-break risk scores deterministically (FIFO).
* **Decoupled Architecture:** Completely separates data simulation logic from production application runtime code.

### Project Structure

* **app/**: Core backend application logic (main API endpoints, model wrappers, and heap queue)
* **scripts/**: Standalone files for data generation and model training pipelines
* **tests/**: Automated unit and integration test suite run via pytest
* **models/**: Local directory storing the trained fraud_model.joblib binary (Git ignored)

### Installation & Running

### 1. Set Up the Backend Environment

bash

python3 -m venv venv
source venv/bin/activate        
pip install -r requirements.txt

Use code with caution.

### 2. Train the Fraud Model

bash

python -m scripts.train_model

Use code with caution.

### 3. Launch the API Gateway

bash

uvicorn app.main:app --reload

Use code with caution.

### Running Tests

Run the entire automated test matrix locally using: 

bash

pytest

Use code with caution.