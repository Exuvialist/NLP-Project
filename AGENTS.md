# AGENTS.md

## Project Context

This repository implements **Tugas 2: Proposal Pipeline & Eksperimen Baseline** for an NLP project that predicts the movement of the USD exchange rate using:

1. Historical USD exchange-rate data.
2. Geopolitical news text.
3. Classical NLP feature extraction.
4. Time-series / machine-learning baseline models.

The assignment requires a comprehensive NLP + ML pipeline and initial baseline experiments. The final repository must contain separated train/validation/test data, modular source code for feature extraction and modeling, notebooks for EDA/initial experiments, and a PDF report describing the task formulation, NLP approach, and detailed pipeline architecture.

## Primary Objective

Build and evaluate a reproducible baseline pipeline answering the following research question:

> Does geopolitical news-derived NLP information provide additional predictive signal for next-day USD direction beyond historical USD data alone?

The implementation should prioritize methodological correctness, temporal integrity, reproducibility, and clear comparison between a market-only baseline and models augmented with NLP features.

## Assignment Constraints

### Required NLP approaches

The project may use classical NLP methods such as:

- Lexicon-based sentiment analysis, e.g. VADER or Loughran-McDonald.
- Classical vectorization, e.g. TF-IDF or Bag-of-Words.

### Explicitly prohibited

Do **not** use:

- Pre-trained embeddings.
- Transformer-based models.
- FinBERT.
- RoBERTa.
- LLM embeddings.
- Sentence-transformer embeddings.
- Any other pre-trained language representation that violates the assignment restriction.

When in doubt about whether a library/model is permitted, prefer the simpler classical NLP solution and document the decision rather than silently introducing a prohibited model.

## Target Formulation

The default project formulation is **binary classification of next-day USD direction**:

\[
y_t =
\begin{cases}
1, & \text{if } Close_{t+1} > Close_t \\
0, & \text{otherwise}
\end{cases}
\]

Interpretation:

- `1` = UP.
- `0` = DOWN / non-increase.

Do not change the target definition casually. Any alternative formulation (regression, volatility classification, etc.) must be explicitly justified and documented in the report.

## Core Pipeline

The intended architecture is:

```text
                    ┌──────────────────────┐
                    │ Historical USD Data  │
                    └──────────┬───────────┘
                               │
                        Market Feature
                         Engineering
                               │
                               ▼
                       Market Features
                               │
                               │
                               ├──────────────┐
                               │              │
                               │              │
                    ┌──────────▼──────────┐   │
                    │ Geopolitical News  │   │
                    └──────────┬──────────┘   │
                               │              │
                         Text Cleaning        │
                               │              │
                    ┌──────────┴──────────┐   │
                    │                     │   │
                    ▼                     ▼   │
               Sentiment              TF-IDF │
                    │                     │   │
                    └──────────┬──────────┘   │
                               │              │
                       Daily Aggregation      │
                               │              │
                               ▼              │
                          NLP Features        │
                               │              │
                               └──────┬───────┘
                                      ▼
                             Feature Integration
                                      │
                                      ▼
                                  ML Model
                                      │
                                      ▼
                                UP / DOWN
                                      │
                                      ▼
                                  Evaluation
```

## Implementation Strategy

Implement the project in the following order. Do not jump directly to model tuning before data integrity and target construction are verified.

### Phase 1 — Dataset Audit

Before modeling, inspect all available data and document:

- File names and formats.
- Row counts.
- Date ranges.
- Column names and types.
- Missing values.
- Duplicate rows/articles.
- Duplicate dates where relevant.
- Number of news articles per day.
- Coverage gaps in exchange-rate data.
- Coverage gaps in news data.
- Timestamp/date conventions and timezone assumptions.
- Relationship between news dates and market dates.

Create a clean, reproducible preprocessing path instead of editing the raw files manually.

### Phase 2 — Target Construction

Construct the next-day direction target from the exchange-rate series.

Important:

- Calculate the target from future price information only after the chronological split logic is established.
- Never allow `Close_{t+1}` or any future-derived value to appear in the feature set for day `t`.
- Clearly document how non-trading days, missing dates, and repeated dates are handled.

### Phase 3 — Market Feature Engineering

Create historical features using information available up to day `t` only.

Candidate features include:

- Previous-day return.
- Multi-day lagged returns.
- Rolling mean / moving average.
- Rolling volatility.
- Previous closing value.
- Other simple lag-based features justified by EDA.

Avoid feature engineering that uses future observations or full-dataset statistics without fitting them only on the training period.

### Phase 4 — NLP Feature Extraction

Build at least one practical classical NLP representation and preferably evaluate two complementary families:

#### Sentiment features

Possible daily features:

- Article count.
- Mean sentiment score.
- Median sentiment score.
- Positive ratio.
- Negative ratio.
- Neutral ratio.
- Optional sentiment dispersion.

#### TF-IDF features

Use TF-IDF on the selected textual field.

The exact text input must be an explicit experiment decision:

- title only,
- article body only,
- or title + article body.

Do not assume full article text is automatically better. Compare or justify the choice using the available dataset and EDA.

### Phase 5 — Temporal Aggregation

News is article-level, while the prediction target is daily. Convert article-level NLP outputs into daily features before joining with market data.

Conceptually:

```text
many articles on date t
        ↓
article-level NLP extraction
        ↓
daily aggregation
        ↓
one NLP feature vector for date t
```

The join key should be an explicitly normalized date.

Be careful with news published after the relevant market observation. The implementation must define the information cutoff used for prediction and avoid using news that would not have been available at prediction time.

### Phase 6 — Chronological Split

Use a time-ordered Train / Validation / Test split.

Example structure:

```text
TRAIN          VALIDATION        TEST
2021–2024      2025              2026
```

The exact dates must be chosen from the actual dataset and documented.

Never use a random split for the final time-series experiment.

For preprocessing methods with learned parameters (for example TF-IDF vocabulary/IDF values or scaling), fit them on the training period only and transform validation/test afterward.

### Phase 7 — Baseline Modeling

Build a market-only baseline first.

Required conceptual comparison:

```text
Model 0: Historical USD features only
Model 1: Historical USD + sentiment
Model 2: Historical USD + TF-IDF
Model 3: Historical USD + sentiment + TF-IDF
```

A simple naive baseline may also be included when useful for context.

Keep the first experiment intentionally simple. Baselines exist to establish a trustworthy reference point, not to maximize score through aggressive hyperparameter tuning.

### Phase 8 — Evaluation

For the binary direction task, report at minimum:

- Directional Accuracy.
- F1-score.

Also report a confusion matrix when useful for interpretation.

Do not compare models solely by one metric if class imbalance or other evaluation issues make the comparison misleading.

Every reported metric must be computed on the correct held-out split and must not use test data for model selection.

### Phase 9 — Experiment Analysis

The key comparison is the incremental value of NLP features.

At minimum, answer:

1. How well does the market-only baseline perform?
2. Does sentiment change performance?
3. Does TF-IDF change performance?
4. What happens when sentiment and TF-IDF are combined?
5. Are there signs of overfitting, leakage, or unstable performance?

Do not claim that NLP "causes" better predictions. The experiments only establish predictive differences under the tested setup.

## Repository Structure

Use a modular structure consistent with the assignment:

```text
project/
├── data/
│   ├── raw/
│   │   ├── exchange_rate.*
│   │   └── news.*
│   └── processed/
│       ├── train.*
│       ├── validation.*
│       └── test.*
│
├── src/
│   ├── data_preparation.py
│   ├── feature_market.py
│   ├── feature_sentiment.py
│   ├── feature_tfidf.py
│   ├── baseline.py
│   ├── train.py
│   └── evaluate.py
│
├── notebook/
│   ├── 01_eda.ipynb
│   └── 02_baseline_experiment.ipynb
│
├── reports/
│   └── pipeline.pdf
│
├── README.md
└── AGENTS.md
```

The exact filenames may be adjusted to match the existing repository, but responsibilities should remain modular and clear.

## Code Organization Rules

- Keep raw-data ingestion separate from feature engineering.
- Keep market features separate from NLP features.
- Keep sentiment and TF-IDF implementations separate so experiments can be enabled/disabled independently.
- Keep training and evaluation logic reusable from both notebooks and scripts.
- Avoid putting the entire pipeline into one notebook.
- Notebooks are for EDA, visualization, experiment orchestration, and interpretation; reusable logic belongs in `src/`.
- Add concise comments where logic is non-obvious, especially around temporal joins and leakage prevention.
- Prefer deterministic behavior by fixing random seeds where supported.
- Make important configuration values explicit rather than hard-coded in multiple files.

## Data Leakage Rules

Data leakage is a first-class concern in this project.

Never:

- Randomly shuffle time-series observations before final splitting.
- Use future USD values as model features.
- Compute rolling features using future rows.
- Fit TF-IDF on validation/test text before evaluation.
- Compute scaling parameters from the whole dataset before splitting.
- Aggregate or normalize information using future periods when predicting an earlier period.
- Use news published after the defined prediction cutoff.
- Tune hyperparameters repeatedly against the test set.

When a preprocessing step has a fitted state, treat that state as training-only.

## Experiment Configuration

Keep experiments easy to reproduce. Prefer an explicit configuration such as:

```text
TARGET = next_day_direction
TEXT_SOURCE = title + article_body
SENTIMENT = enabled/disabled
TFIDF = enabled/disabled
MODEL = chosen_baseline_model
RANDOM_SEED = fixed_value
TRAIN_PERIOD = documented_date_range
VALIDATION_PERIOD = documented_date_range
TEST_PERIOD = documented_date_range
```

Record the configuration associated with every reported experiment.

## Expected Deliverables

The implementation is complete only when the repository contains:

### 1. Data

Separated train, validation, and test datasets under `data/` or the project's equivalent structure.

### 2. Source Code

Modular code for:

- Data preparation.
- Market feature extraction.
- NLP sentiment extraction.
- TF-IDF extraction.
- Baseline/model training.
- Evaluation.

### 3. Notebooks

At minimum:

- EDA notebook.
- Initial baseline/experiment notebook.

### 4. Report

The PDF must explain:

- Task formulation, including target and metrics.
- Rationale for the selected NLP feature extraction approach.
- Detailed architecture/pipeline diagram.
- Initial experimental results and interpretation.

## Definition of Done

Consider the implementation ready for submission only when all of the following are true:

- [ ] Dataset structure has been audited and documented.
- [ ] Target definition is implemented and verified.
- [ ] Historical market features are leakage-safe.
- [ ] News text is cleaned consistently.
- [ ] At least one allowed classical NLP feature pipeline is implemented.
- [ ] Any use of TF-IDF is fit on training data only.
- [ ] News is aggregated to the appropriate temporal granularity.
- [ ] News/market joining uses a documented information cutoff.
- [ ] Train/validation/test are chronological.
- [ ] Market-only baseline is trained.
- [ ] NLP-augmented experiment is trained.
- [ ] Directional Accuracy and F1 are reported.
- [ ] Results are reproducible.
- [ ] No prohibited pretrained embeddings or transformer models are used.
- [ ] Notebook outputs correspond to the current source code.
- [ ] Repository structure matches the assignment deliverables.
- [ ] PDF report includes formulation, NLP rationale, and pipeline architecture.

## Agent Working Rules

When implementing changes in this repository:

1. **Inspect before modifying.** Understand the current dataset, existing scripts, and repository structure before introducing new code.
2. **Preserve temporal integrity.** Any new feature must be checked for future information leakage.
3. **Prefer minimal, modular changes.** Do not rewrite unrelated components merely to fit a preferred structure.
4. **Document assumptions.** Especially assumptions involving dates, market availability, article timestamps, and missing values.
5. **Keep experiments comparable.** When comparing models, hold the split, target, evaluation protocol, and relevant preprocessing rules constant.
6. **Do not silently change the research question.** Ask for or document a deliberate change when altering the target or experimental design.
7. **Do not add prohibited NLP models.** Classical approaches are the default unless the assignment specification is explicitly changed.
8. **Validate after implementation.** Run the relevant preprocessing, training, and evaluation path after code changes.
9. **Make failures explicit.** Do not fabricate metrics or report successful experiments that were not actually executed.
10. **Update documentation when behavior changes.** README, experiment configuration, and report text should remain consistent with the code.

## Preferred Development Sequence

```text
1. Inspect repository + dataset
2. Run EDA
3. Define target + information cutoff
4. Build chronological split
5. Build market-only baseline
6. Build sentiment features
7. Add TF-IDF features
8. Run controlled experiments
9. Evaluate on validation/test appropriately
10. Analyze results
11. Finalize pipeline diagram
12. Finalize PDF + README
```

## Final Principle

The goal is not to produce the most complicated model. The goal is to produce a **methodologically sound, reproducible, classical-NLP baseline pipeline** where the contribution of geopolitical news can be compared fairly against historical USD information.
