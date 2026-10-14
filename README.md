# AI-Powered Smart City Intelligence System

A locally runnable Streamlit mini-project for a B.Tech AI & Data Science practical portfolio.

## Features
- PEAS intelligent-agent formulation
- BFS, DFS and Uniform Cost Search route planner
- A* warehouse robot pathfinding with editable obstacles
- Hotel booking rule engine with SQLite persistence
- Bigram and trigram customer-support text generation
- Synthetic-data ANN + fuzzy-style traffic recommendation
- Minimax pricing game and alpha-beta search demonstration
- Fuzzy-style smart irrigation recommendation
- Educational Mamdani-style and Sugeno-style credit-risk outputs
- Perceptron customer churn classification with held-out metrics
- Custom winner-neuron competitive learning for customer segmentation
- Algorithm comparison and viva questions

## Run
Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## Data and safety notes
- Traffic, churn and segmentation data are synthetic and reproducible.
- The hotel module uses a local SQLite database (`smart_city.db`) created at runtime.
- The hotel module uses explicit Python rules as a fallback. It does not execute SWI-Prolog.
- Fuzzy traffic, irrigation and credit-risk outputs are educational demonstrations, not validated real-world controls or decisions.
- No paid API, API key, cloud AI service, live traffic feed, or physical hardware is required.
