"""
Raw data loading and cleaning.

Each stock exchange's raw export uses different sheet names and column
labels, so each country gets a small dedicated loader that standardizes
everything down to a common schema:

    ['Date', 'Firm', 'Stock_Return', 'Market_Return', 'Country']

so the rest of the pipeline (event_window / market_model / calculations)
can treat every country identically.
"""

import pandas as pd


def load_ghana(file_path: str) -> pd.DataFrame:
    stocks = pd.read_excel(file_path, sheet_name="GSE_STOCKS_SORTED")
    market = pd.read_excel(file_path, sheet_name="GSE_MARKET")

    stocks = stocks[['Daily Date', 'Share Code', 'Share_Return']].copy()
    stocks = stocks.rename(columns={
        'Daily Date': 'Date',
        'Share Code': 'Firm',
        'Share_Return': 'Stock_Return',
    })

    market = market[['Date', 'GSE_MRKT_RET']].copy()
    market = market.rename(columns={'GSE_MRKT_RET': 'Market_Return'})

    stocks['Date'] = pd.to_datetime(stocks['Date'], dayfirst=True, errors='coerce')
    market['Date'] = pd.to_datetime(market['Date'], dayfirst=True, errors='coerce')
    stocks['Stock_Return'] = pd.to_numeric(stocks['Stock_Return'], errors='coerce')
    market['Market_Return'] = pd.to_numeric(market['Market_Return'], errors='coerce')

    data = pd.merge(stocks, market, on='Date', how='inner')
    data = data.sort_values(['Firm', 'Date']).reset_index(drop=True)
    data['Country'] = 'Ghana'
    return data


def load_nigeria(file_path: str) -> pd.DataFrame:
    stocks = pd.read_excel(file_path, sheet_name="Nigeria")
    market = pd.read_excel(file_path, sheet_name="Nigeria_Market")

    stocks = stocks.rename(columns={'Company': 'Firm', 'Share_Return': 'Stock_Return'})
    stocks['Country'] = 'Nigeria'

    market = market.rename(columns={'Change %': 'Market_Return'})
    market = market[['Date', 'Market_Return']]

    stocks['Date'] = pd.to_datetime(stocks['Date'])
    market['Date'] = pd.to_datetime(market['Date'])

    return pd.merge(stocks, market, on='Date', how='inner')


def load_egypt(file_path: str) -> pd.DataFrame:
    stocks = pd.read_excel(file_path, sheet_name="Egypt_EGX")
    market = pd.read_excel(file_path, sheet_name="Egypt_Market")

    stocks = stocks.rename(columns={'Stock': 'Firm'})
    stocks['Country'] = 'Egypt'

    market = market.rename(columns={'Change %': 'Market_Return'})
    market = market[['Date', 'Market_Return']]

    stocks['Date'] = pd.to_datetime(stocks['Date'])
    market['Date'] = pd.to_datetime(market['Date'])

    return pd.merge(stocks, market, on='Date', how='inner')


def load_mauritius(file_path: str) -> pd.DataFrame:
    stocks = pd.read_excel(file_path, sheet_name="Mauritius")
    market = pd.read_excel(file_path, sheet_name="Mauritius_Market")

    stocks = stocks.rename(columns={'Company': 'Firm', 'Stock_Returns': 'Stock_Return'})
    stocks['Country'] = 'Mauritius'

    market = market.rename(columns={'Change %': 'Market_Return'})
    market = market[['Date', 'Market_Return']]

    stocks['Date'] = pd.to_datetime(stocks['Date'])
    market['Date'] = pd.to_datetime(market['Date'])

    return pd.merge(stocks, market, on='Date', how='inner')


# Registry so callers can loop over countries instead of hardcoding branches
COUNTRY_LOADERS = {
    'Ghana': load_ghana,
    'Nigeria': load_nigeria,
    'Egypt': load_egypt,
    'Mauritius': load_mauritius,
}

# COVID-19 event dates per country (WHO pandemic declaration is shared;
# first confirmed case and lockdown dates are country-specific)
COUNTRY_EVENTS = {
    'Ghana': {
        'WHO_Declaration': pd.Timestamp('2020-03-11'),
        'First_Case': pd.Timestamp('2020-03-12'),
        'Lockdown': pd.Timestamp('2020-03-30'),
    },
    'Nigeria': {
        'WHO_Declaration': pd.Timestamp('2020-03-11'),
        'First_Case': pd.Timestamp('2020-02-27'),
        'Lockdown': pd.Timestamp('2020-03-30'),
    },
    'Egypt': {
        'WHO_Declaration': pd.Timestamp('2020-03-11'),
        'First_Case': pd.Timestamp('2020-02-16'),
        'Lockdown': pd.Timestamp('2020-03-25'),
    },
    'Mauritius': {
        'WHO_Declaration': pd.Timestamp('2020-03-11'),
        'First_Case': pd.Timestamp('2020-03-18'),
        'Lockdown': pd.Timestamp('2020-03-20'),
    },
}
