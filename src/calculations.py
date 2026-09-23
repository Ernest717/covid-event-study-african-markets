"""
Event-study statistics.

Cross-sectional significance testing for abnormal returns (AR) and
cumulative abnormal returns (CAR/CAAR), plus the formatted output table
matching the project's original Excel layout.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _stars(p: float) -> str:
    """Significance stars: * p<0.10, ** p<0.05, *** p<0.01."""
    if p < 0.01:
        return '***'
    if p < 0.05:
        return '**'
    if p < 0.10:
        return '*'
    return ''


def build_event_summary(event_window_data: pd.DataFrame) -> pd.DataFrame:
    """
    Per-event-day statistics with cross-sectional t-tests:
      AAR, t_stat_AAR, p_val_AAR, Sig_AAR     -- average abnormal return
      CAAR, t_stat_CAAR, p_val_CAAR, Sig_CAAR -- cumulative average abnormal return
      Avg_Firm_CAR                             -- mean of firm-level CARs
    """
    aar_stats = (
        event_window_data
        .groupby('t')
        .agg(
            N=('AR', 'count'),
            AAR=('AR', 'mean'),
            AR_std=('AR', 'std'),
            CAAR=('Market_CAR', 'first'),
            Avg_Firm_CAR=('CAR', 'mean'),
        )
        .reset_index()
    )

    caar_std = (
        event_window_data
        .groupby('t')['CAR']
        .std()
        .reset_index()
        .rename(columns={'CAR': 'CAR_std'})
    )
    summary = aar_stats.merge(caar_std, on='t')

    summary['AR_se'] = summary['AR_std'] / np.sqrt(summary['N'])
    summary['CAR_se'] = summary['CAR_std'] / np.sqrt(summary['N'])

    summary['t_stat_AAR'] = summary['AAR'] / summary['AR_se']
    summary['t_stat_CAAR'] = summary['CAAR'] / summary['CAR_se']

    dof = summary['N'] - 1
    summary['p_val_AAR'] = 2 * (1 - stats.t.cdf(summary['t_stat_AAR'].abs(), dof))
    summary['p_val_CAAR'] = 2 * (1 - stats.t.cdf(summary['t_stat_CAAR'].abs(), dof))

    summary['Sig_AAR'] = summary['p_val_AAR'].apply(_stars)
    summary['Sig_CAAR'] = summary['p_val_CAAR'].apply(_stars)

    summary = summary.drop(columns=['AR_std', 'AR_se', 'CAR_std', 'CAR_se'])

    for col in ['AAR', 'CAAR', 'Avg_Firm_CAR', 't_stat_AAR', 't_stat_CAAR',
                'p_val_AAR', 'p_val_CAAR']:
        summary[col] = summary[col].round(6)

    return summary.sort_values('t').reset_index(drop=True)


def fixed_window_stats(data: pd.DataFrame, start_t: int, end_t: int):
    """
    Mean CAR, t-stat and significance ('Yes'/'No' at 5%) for firms' summed
    abnormal returns over a fixed [start_t, end_t] window.
    """
    temp = data[(data['t'] >= start_t) & (data['t'] <= end_t)].copy()

    firm_car = (
        temp.groupby('Firm')['AR']
        .sum()
        .reset_index(name='CAR_window')
    )

    mean_car = firm_car['CAR_window'].mean()
    std_car = firm_car['CAR_window'].std()
    n = firm_car['CAR_window'].count()

    if std_car == 0 or pd.isna(std_car):
        t_stat = np.nan
    else:
        t_stat = mean_car / (std_car / np.sqrt(n))

    sig = 'Yes' if pd.notna(t_stat) and abs(t_stat) > 1.96 else 'No'
    return mean_car, t_stat, sig


def build_final_output(event_window_data: pd.DataFrame) -> pd.DataFrame:
    """
    Day-by-day output table: exchange/benchmark returns, AR with t-stat and
    significance, the (-5,+5) market CAR, and the (-1,+1) / (0,+1) fixed
    window CARs (reported on the t = +1 row), matching the project's
    original Excel report format.
    """
    daily = (
        event_window_data
        .groupby(['Date', 't'])
        .agg(
            Exchange_Return=('Stock_Return', 'mean'),
            Benchmark_Return=('Expected_Return', 'mean'),
            AR=('AR', 'mean'),
            AR_std=('AR', 'std'),
            N=('AR', 'count'),
            Market_CAR=('Market_CAR', 'first'),
            Firm_CAR_std=('CAR', 'std'),
        )
        .reset_index()
        .sort_values('t')
    )

    daily['AR_se'] = daily['AR_std'] / np.sqrt(daily['N'])
    daily['T-Stat_AR'] = daily['AR'] / daily['AR_se']
    daily['Significant?(5% level)'] = np.where(daily['T-Stat_AR'].abs() > 1.96, 'Yes', 'No')

    daily['t-stat [-5, +5]'] = daily['Market_CAR'] / (daily['Firm_CAR_std'] / np.sqrt(daily['N']))
    daily['Sig? [-5, +5]'] = np.where(daily['t-stat [-5, +5]'].abs() > 1.96, 'Yes', 'No')

    car_m1_p1, t_m1_p1, sig_m1_p1 = fixed_window_stats(event_window_data, -1, 1)
    car_0_p1, t_0_p1, sig_0_p1 = fixed_window_stats(event_window_data, 0, 1)

    daily['CAR (-5, +5)%'] = daily['Market_CAR']
    daily['CAR (-1, +1)%'] = np.nan
    daily['t-stat [-1, +1]'] = np.nan
    daily['Sig? [-1, +1]'] = ''
    daily['CAR (0, +1)%'] = np.nan
    daily['t-stat [0, +1]'] = np.nan
    daily['Sig? [0, +1]'] = ''

    daily.loc[daily['t'] == 1, 'CAR (-1, +1)%'] = car_m1_p1
    daily.loc[daily['t'] == 1, 't-stat [-1, +1]'] = t_m1_p1
    daily.loc[daily['t'] == 1, 'Sig? [-1, +1]'] = sig_m1_p1

    daily.loc[daily['t'] == 1, 'CAR (0, +1)%'] = car_0_p1
    daily.loc[daily['t'] == 1, 't-stat [0, +1]'] = t_0_p1
    daily.loc[daily['t'] == 1, 'Sig? [0, +1]'] = sig_0_p1

    final_output = daily.rename(columns={
        'Exchange_Return': 'Exchange_Return(%)',
        'Benchmark_Return': 'Benchmark_Return(%)',
        'AR': 'AR (%)',
    })

    return final_output[[
        'Date', 't',
        'Exchange_Return(%)', 'Benchmark_Return(%)', 'AR (%)',
        'T-Stat_AR', 'Significant?(5% level)',
        'CAR (-5, +5)%', 't-stat [-5, +5]', 'Sig? [-5, +5]',
        'CAR (-1, +1)%', 't-stat [-1, +1]', 'Sig? [-1, +1]',
        'CAR (0, +1)%', 't-stat [0, +1]', 'Sig? [0, +1]',
    ]]
