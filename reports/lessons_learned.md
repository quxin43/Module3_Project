# Lessons Learned: SPY Buy / Sell / Hold classifier

## The result in one paragraph

The project trained 9 classifiers to label each trading day Buy (SPY rises more than 1% over the next 5 days), Sell (falls more than 1%) or Hold. They were evaluated by walk-forward validation on 2005–2020, and the frozen recipe was then tested once on 2021 to Oct 2026.

- **Prediction skill is real but small.** The chosen model (L1-regularised logistic regression) scored a macro-F1 of **0.399** on the test, against **0.341** for a dummy.
- **It did not beat holding SPY.** With the asymmetric trading rule (go to cash only when P(Sell) ≥ 0.50), $10,000 grew to **$21,272**, against **$22,787** for Buy & Hold. That is 15 percentage points behind, after 0.1% fees and 0.05% slippage per trade.
- **This is the outcome the brief anticipates.** The method holds up: the look-ahead tests pass, and the test years were opened only once.

## What worked

| Result | Evidence |
|---|---|
| Models learned something real | Every model beat the dummy's macro-F1 (0.340) on 2005–2020, the perceptron only barely (0.357). The best three scored 0.42. The test score (0.399) stayed above the dummy, with non-overlapping 95% intervals |
| Simple models were enough | Logistic regression tied the random forest and MLP and beat both boosting models. Extra flexibility found no extra signal |
| Asymmetric trading rule | Going to cash only on a confident Sell turned large losses into near-Buy-&-Hold results. The plain "highest probability wins" rule lost heavily in both validation and test ($12,562 vs $22,787 on the test) |
| Walk-forward instead of one split | Yearly folds showed how unstable the edge is: yearly macro-F1 ranged from about 0.25 to 0.46, and no model was best every year |
| Leakage controls | Automated tests confirm features don't change when future rows are removed, a 5-row gap separates training and predicted years, and trades happen the day after the signal. A plain-Python backtest matches vectorbt to the cent |

## What did not work, and why

1. **The skill is in "calm vs wild", not "up vs down".** Hold recall was about 0.7, but Buy and Sell recall only about 0.3. The EDA showed why: volatility predicts how *big* the next week's move will be, not its *direction*. Buy weeks and Sell weeks start from almost identical indicator readings (median RSI 53.4 vs 53.3).
2. **Accuracy is not money.** On the test, the classifier was right 43% of the time, yet the strategy still lost to Buy & Hold. It was invested 98.7% of the time and traded only 7 times. Its three exits came right after drops, in June 2022 and April–May 2025, and the market rebounded without it. Missing a few strong recovery days costs more than many correct "Hold" calls earn.
3. **The validation win was flattered by selection.** In 2005–2020 the chosen rule beat Buy & Hold by 159 percentage points. But most of that came from one event (stepping aside in late 2008), and it was the best of 21 model-and-rule combinations. Picking the best of many backtests on the same data overstates the edge. The untouched test gave the honest answer.
4. **Costs were not the problem here, timing was.** The chosen strategy paid only $135 in costs on the test, and the zero-cost version still trails Buy & Hold. For the plain rule, though, costs were large: about $2,500 on the test and $4,000–8,000 in validation.
5. **Feature importance is weak and unstable.** Shuffled together, the volatility family (20-day volatility, ATR) helped all 8 real models over 2005–2020, and the momentum family (RSI, MACD) helped most. The moving-average trend ratios slightly hurt 6 of the 8 models on new years. Shuffling any one feature changed macro-F1 by at most about 0.04. Several features had negative importance in 2017–2020, meaning patterns learned earlier did not hold. Many indicators are near-duplicates (12 pairs with |correlation| ≥ 0.8), so importance is shared and hard to read.

## Why markets are hard to predict this way

- **Public prices are already reflected.** SPY is one of the most traded assets in the world. Any simple pattern in public price and volume data that reliably predicted next week would be traded away. This is the efficient-market idea, and the results are consistent with it.
- **Regimes change.** A rule that helped in the 2008 crash added nothing in calm years and hurt in fast V-shaped recoveries (2022, 2025).
- **The upward drift is a strong opponent.** SPY rose in most years, so any time spent in cash has to be repaid by avoiding a larger fall later.

## Simplifications to keep in mind

- **Dividends:** reinvested immediately at no cost, through adjusted prices. A Singapore investor would normally pay 30% US withholding tax on SPY dividends, which this backtest ignores. It applies to both the strategy and Buy & Hold.
- **Execution:** fills at the next day's open plus 0.05% slippage, and cash earns 0%. Interest on cash would have helped the strategy slightly in 2023–2026.
- **Missing final week:** the last 5 trading days of data have no label yet, so the backtest ends on 9 Oct 2026.

## What I would try next

1. **Predict volatility instead of direction.** The models are good at spotting calm weeks, so use them to size positions (for example, hold less SPY when predicted risk is high) rather than to go all-in or all-out.
2. **Fewer, less overlapping features.** Use one indicator per idea (trend, momentum, volatility, volume) to make importance readable and reduce noise.
3. **Treat the Sell cut-off as part of the model.** Tune it inside each walk-forward fold, rather than once on the whole validation period, to reduce the selection bias seen here.
4. **Add information prices don't contain.** For example, interest rates, credit spreads or the VIX. These would have to be lagged in the same careful way.
5. **Longer horizon.** A 20-day label gives fewer trades and less noise per label, at the cost of fewer independent examples.
