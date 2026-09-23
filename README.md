# Multi-Country African Market Event Study: COVID-19 Impact

An event study measuring how COVID-19 news shocks affected stock returns on
the **Ghana Stock Exchange**, extended to a cross-country comparison with
**Nigeria, Egypt, and Mauritius**.

## What this does

For each country and each of three event dates — the WHO pandemic
declaration, the first confirmed local case, and the national lockdown
announcement — the analysis:

1. Estimates a **market model** (OLS: firm return ~ market return) per firm
   over a clean pre-event estimation window, to get each firm's Alpha/Beta.
2. Uses Alpha/Beta to compute each firm's **expected return** on every day
   around the event, and the **abnormal return (AR)** as actual − expected.
3. Aggregates AR into **cumulative abnormal returns (CAR)** per firm and
   **cumulative average abnormal returns (CAAR)** across the market, over an
   11-day event window (t = −5 to +5).
4. Runs cross-sectional t-tests to flag which days/windows show
   statistically significant abnormal reaction (5% level).
5. Compares CAAR paths across the four countries for each event.

## Repo structure

```
src/
  data_processing.py   # per-country loaders that standardize raw exports
                        # into a common schema (Date, Firm, Stock_Return,
                        # Market_Return, Country) + event date registry
  market_model.py       # OLS market model estimation + abnormal return calc
  event_window.py        # calendar time -> event time, market-level CAR
  calculations.py       # significance tests + final report table
notebooks/
  01_event_study_analysis.ipynb   # full walkthrough: load -> model -> test
                                   # -> compare -> export, with charts
data/
  raw/        Data Individual Stocks.xlsx        # firm + market daily returns, 4 countries
  processed/  Multi_Country_Event_Study.xlsx      # final results workbook (all events, all countries, cross-country comparison sheets + charts)
```

## Method notes

- Estimation window: t = −135 to −16 (trading days relative to the event),
  minimum 30 observations per firm.
- Event window: t = −5 to +5, plus fixed (−1,+1) and (0,+1) windows for
  tighter significance checks.
- Significance: two-tailed t-test, 5% level (`*` 10%, `**` 5%, `***` 1%).

## Running it

```bash
pip install pandas numpy statsmodels scipy matplotlib xlsxwriter openpyxl
jupyter notebook notebooks/01_event_study_analysis.ipynb
```

The notebook currently runs the full pipeline inline; `src/` holds the same
logic factored into reusable functions for anyone who wants to script the
analysis (e.g. add a new country) without going through the notebook.

## Data

`data/raw/Data Individual Stocks.xlsx` contains daily firm-level and
market-index returns for Ghana, Nigeria, Egypt, and Mauritius around the
COVID-19 period. `data/processed/Multi_Country_Event_Study.xlsx` is the
final output: per-country/per-event result tables, cross-country CAAR
comparison sheets, and charts.
