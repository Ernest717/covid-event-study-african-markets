"""
Market model estimation.

Standard event-study market model: for each firm, regress its return on the
market return over a clean "estimation window" (before the event window) to
get Alpha/Beta, then use those to predict an expected return and derive the
abnormal return (actual - expected) for every day around the event.
"""

import pandas as pd
import statsmodels.api as sm


def run_market_model(data: pd.DataFrame, est_start: int = -135, est_end: int = -16) -> pd.DataFrame:
    """
    Fit OLS (Stock_Return ~ Market_Return) per firm over the estimation
    window [est_start, est_end] (in event time 't'), skipping firms with
    fewer than 30 estimation-window observations.

    Returns a DataFrame with one row per firm: ['Firm', 'Alpha', 'Beta'].
    """
    results = []

    for firm, group in data.groupby('Firm'):
        estimation = group[(group['t'] >= est_start) & (group['t'] <= est_end)]

        if len(estimation) < 30:
            continue

        X = sm.add_constant(estimation['Market_Return'])
        y = estimation['Stock_Return']
        model = sm.OLS(y, X).fit()

        results.append({
            'Firm': firm,
            'Alpha': model.params['const'],
            'Beta': model.params['Market_Return'],
        })

    return pd.DataFrame(results)


def compute_abnormal_returns(event_time_data: pd.DataFrame, model_params: pd.DataFrame) -> pd.DataFrame:
    """
    Merge each firm's Alpha/Beta onto its event-time data and compute:
      Expected_Return = Alpha + Beta * Market_Return
      AR              = Stock_Return - Expected_Return
      CAR             = firm-level cumulative sum of AR over event time

    Parameters
    ----------
    event_time_data : output of event_window.create_event_time
    model_params : output of run_market_model (['Firm', 'Alpha', 'Beta'])
    """
    data = event_time_data.merge(model_params, on='Firm', how='left')

    data['Expected_Return'] = data['Alpha'] + data['Beta'] * data['Market_Return']
    data['AR'] = data['Stock_Return'] - data['Expected_Return']

    data = data.sort_values(['Firm', 't'])
    data['CAR'] = data.groupby('Firm')['AR'].cumsum()

    return data
