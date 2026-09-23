"""
Event-window construction.

Turns calendar-time firm/market return data into "event time" (t = 0 on the
event date, negative/positive integers for trading days before/after it),
and adds market-wide cumulative abnormal return (CAR) tracking.
"""

import pandas as pd


def create_event_time(data: pd.DataFrame, event_date: pd.Timestamp) -> pd.DataFrame:
    """
    Re-index each firm's trading days around ``event_date`` so that the event
    date is t = 0, the prior trading day is t = -1, the next is t = +1, etc.

    Firms that never traded on ``event_date`` are dropped (there is no t = 0
    to anchor to).

    Parameters
    ----------
    data : DataFrame with at least ['Firm', 'Date'] columns, one row per
        firm/trading day.
    event_date : the calendar date the event occurred on.

    Returns
    -------
    DataFrame identical to ``data`` plus an integer 't' column.
    """
    event_data = []

    for firm, group in data.groupby('Firm'):
        group = group.sort_values('Date').reset_index(drop=True)

        if event_date not in group['Date'].values:
            continue

        event_index = group[group['Date'] == event_date].index[0]
        group['t'] = group.index - event_index

        event_data.append(group)

    return pd.concat(event_data, ignore_index=True)


def add_market_level_car(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add market-wide AAR (average abnormal return across firms, per event day)
    and Market_CAR (its running cumulative sum).

    Requires ``data`` to already have 't' (event time) and 'AR' (abnormal
    return) columns, e.g. from market_model.compute_abnormal_returns.
    """
    data = data.copy()

    # Average abnormal return for each event day
    data['AAR'] = data.groupby('t')['AR'].transform('mean')

    # One row per event day, cumulated
    market_car_map = (
        data[['t', 'AAR']]
        .drop_duplicates()
        .sort_values('t')
        .copy()
    )
    market_car_map['Market_CAR'] = market_car_map['AAR'].cumsum()

    return data.merge(market_car_map[['t', 'Market_CAR']], on='t', how='left')


def extract_window(data: pd.DataFrame, start_t: int = -5, end_t: int = 5) -> pd.DataFrame:
    """Restrict event-time data to the [start_t, end_t] event window."""
    return data[(data['t'] >= start_t) & (data['t'] <= end_t)].copy()
