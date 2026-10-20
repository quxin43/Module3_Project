# Rerun recipe (frozen before the rerun starts)

Written and committed **before** any rerun result exists. The rerun replaces the tuning rule and the Sell cut-off rule; everything else is unchanged from the first run.

**Why the rerun:** the first run picked settings with randomized search (12–20 random draws) and kept the single highest practice score. The validation curves showed that this can pick a narrow spike (for example KNN n_neighbors = 93 in 2007–2011) or a point at the edge of a plateau, and random draws can miss parts of the range. The rerun was prompted by those validation charts, after the first test result had already been seen. Both results will be reported, whichever is better.

## 1. How the best setting of a model is defined (every model, every predicted year)

1. **Fixed grid.** Every combination in the grid below is scored. Settings not in the grid use the fixed value listed (library default unless stated).
2. **Score.** Each setting is trained and scored on the same 5 practice blocks as before (blocks of 252 trading days inside the training window, 5-day gap, `TimeSeriesSplit(n_splits=5, test_size=252, gap=5)`). Score = mean macro-F1 over the 5 blocks.
3. **Smooth.** Each setting's score is replaced by the average of its own score and the scores of its grid neighbours one step away on each numeric setting. Categorical settings (weights, penalty, max_features, hidden layers) are never averaged across. A point on the edge of the grid has fewer neighbours; that is accepted as is.
4. **Pick.** The setting with the highest smoothed score wins. If several tie to 3 decimals, the simpler one wins (order below).
5. **Edge flag.** If the winner is on the edge of a numeric range, the report flags it. Ranges are not widened during the run.
6. **Use.** The winner is refitted on the whole training window (minus the 5-day gap) and predicts the year.

| Model | Grid | Fixed (not tuned) | Size | Simpler means (tie-break) |
|---|---|---|---|---|
| Dummy | strategy most_frequent / stratified | — | 2 | most_frequent |
| KNN | n_neighbors 5–200 step 5 × weights uniform / distance | — | 80 | more neighbours, then uniform |
| Logistic regression | C 0.001 → 10, 17 log-spaced × L2 / L1 | — | 34 | smaller C, then L1 |
| Decision tree | max_depth 2–8 × min_samples_leaf 20, 40, 60, 80, 100, 125, 150, 175, 200 | — | 63 | shallower, then larger leaves |
| Random forest | max_depth 3–10 × min_samples_leaf 20, 50, 100, 150, 200 × max_features sqrt / 0.5 | n_estimators = 300 (more trees only reduce noise) | 80 | shallower, larger leaves, sqrt |
| HistGradientBoosting | learning_rate 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2 × max_iter 100–500 step 100 × l2_regularization 0 / 0.5 / 1 | max_leaf_nodes = 31 (default) | 105 | fewer rounds, smaller learning rate, more l2 |
| XGBoost | learning_rate (same 7) × n_estimators 100–500 step 100 × colsample_bytree 0.6 / 0.8 / 1.0 | max_depth = 6, subsample = 1.0 (defaults) | 105 | fewer trees, smaller learning rate, smaller colsample |
| Perceptron | alpha 0.00001 → 0.01, 13 log-spaced × L2 / L1 / elasticnet | — | 39 | larger alpha, then L2 |
| MLP | learning_rate_init 0.0001 → 0.01, 9 log-spaced × hidden layers (16), (32), (32, 16) | alpha = 0.0001 (default) | 27 | smaller network, then smaller learning rate |

Numeric (smoothed) settings: n_neighbors; C; max_depth and min_samples_leaf; learning_rate, max_iter / n_estimators, l2_regularization / colsample_bytree; alpha; learning_rate_init.

Computation shortcut for the two boosting models (does not change any result): a boosting model with 200 rounds is exactly the first 200 rounds of the same model trained for 500 rounds (no early stopping, no row subsampling), so each learning rate × third-setting combination is trained once for 500 rounds and scored after 100, 200, 300, 400 and 500 rounds.

## 2. How the Sell cut-off is defined (shortlisted models, once, on 2005–2020)

Signal rule unchanged: Sell if P(Sell) ≥ c; otherwise Buy if P(Buy) is the highest; otherwise Hold. Candidates: c = 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, plus the plain "highest probability wins" rule.

1. Backtest each shortlisted model with each rule as one account over 2005–2020, with costs, against Buy & Hold.
2. **Smooth:** each cut-off's total excess return is replaced by the average over itself and its neighbours (c − 0.05, c, c + 0.05; the ends have one neighbour). The "highest probability wins" rule has no neighbours and keeps its own value.
3. **Pick** the rule with the highest smoothed excess return (ties: higher Sharpe).
4. The model ranking then uses that rule's actual (unsmoothed) results: total excess return, years won, median yearly Sharpe; add the ranks, lowest total wins, tie to the higher excess return; worst-year guard −20 percentage points.

## 3. Unchanged from the first run

Data, 14 features, labels (±1% over 5 days), walk-forward folds (validation 2005–2020, test 2021–2026), 5-row gaps, pipelines and class balancing, shortlist = top 3 by pooled macro-F1 on 2005–2020, trading rules (next day's open, all-in / all-out, 0.1% fee + 0.05% slippage, cash 0%), and **one** test run of the frozen recipe.

## 4. Display-only additions (do not affect any choice)

- **Train-score line** on each validation curve (score on the training part of each practice split), to show underfitting / overfitting.
- **MLP training curves** for each year's winning MLP: training loss and internal validation accuracy per epoch (what scikit-learn records).
- **Sell cut-off chart:** raw and smoothed total excess return against c for the shortlisted models.
- **Comparison on 2005–2020:** old rule vs new rule vs library default settings, macro-F1 per year.
