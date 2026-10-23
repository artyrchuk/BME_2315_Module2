from scipy.interpolate import interp1d
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = '/Users/ellepickard/Documents/Computational BME/Module 2/Untitled/BME_2315_Module2/DataSet/Module2_FSHD_Dataset.csv'
df = pd.read_csv(path)

# Keep complete three-visit patients
cols = ['lag_months_y1', 'lag_months_y2',
        'ff_baseline_R', 'ff_y1_R', 'ff_y2_R',
        'ff_baseline_L', 'ff_y1_L', 'ff_y2_L',
        'wr_10_baseline', 'wr_10_y1', 'wr_10_y2']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=cols).copy()
df = df[(df.lag_months_y1 > 0) &
        (df.lag_months_y2 > df.lag_months_y1)]
df = df[(df.ff_baseline_R > 0) & (df.ff_baseline_L > 0) &
        (df.wr_10_baseline > 0)].copy()

T1, T2 = df.lag_months_y1.min(), df.lag_months_y2.min()

# Interpolate each patient's measurements
def interp(row, col, t):
    side = col[-1] if col.startswith('ff_') else None

    if side:
        times = [0, row['lag_months_y1'], row['lag_months_y2']]
        vals = [
            row[f'ff_baseline_{side}'],
            row[f'ff_y1_{side}'],
            row[f'ff_y2_{side}']
        ]
    else:
        times = [0, row['lag_months_y1'], row['lag_months_y2']]
        vals = [
            row['wr_10_baseline'],
            row['wr_10_y1'],
            row['wr_10_y2']
        ]

    return float(interp1d(times, vals)(t))

# Individual percent changes from baseline
for side in ['R', 'L']:
    for label, t in [('T1', T1), ('T2', T2)]:
        df[f'ff_{label}_{side}'] = df.apply(
            lambda r: interp(r, f'ff_{side}', t), axis=1
        )

for label, t in [('T1', T1), ('T2', T2)]:
    df[f'wr_10_{label}'] = df.apply(
        lambda r: interp(r, 'wr_10', t), axis=1
    )

df['tug_time_T2'] = df.apply(
    lambda r: interp(r, 'tug_time', T2), axis=1
)


# Individual patient bar graphs
out = os.path.join(os.path.dirname(path), 'FSHD_Results')
os.makedirs(out, exist_ok=True)

for side in ['R', 'L']:
    for t in ['T1', 'T2']:
        df[f'pct_{side}_{t}'] = (
            (df[f'ff_{t}_{side}'] - df[f'ff_baseline_{side}'])
            / df[f'ff_baseline_{side}'] * 100
        )

for i, r in df.iterrows():
    fig, ax = plt.subplots()
    x = np.arange(2)
    ax.bar(x - .2, [r.pct_R_T1, r.pct_R_T2], .4, label='Right')
    ax.bar(x + .2, [r.pct_L_T1, r.pct_L_T2], .4, label='Left')
    ax.axhline(0, color='gray')
    ax.set_xticks(x, [f'T1 ({T1:.1f} mo)', f'T2 ({T2:.1f} mo)'])
    ax.set_ylabel('Fat fraction change (%)')
    ax.set_title(f'Patient {r.get("patient_id", i)}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{out}/patient_{i}.png')
    plt.close(fig)


# Statistical significance: change from baseline and between time points
for side in ['R', 'L']:
    for t in ['T1', 'T2']:
        x = df[f'pct_{side}_{t}'].dropna()
        if len(x) >= 2:
            print(side, t, 'change vs baseline:',
                  stats.ttest_1samp(x, 0))

    pair = df[[f'pct_{side}_T1', f'pct_{side}_T2']].dropna()
    if len(pair) >= 3:
        print(side, 'T1 vs T2:',
              stats.ttest_rel(pair.iloc[:, 0], pair.iloc[:, 1]))


# Calculate percent change in 10-meter walk/run time
for t in ['T1', 'T2']:
    df[f'wr_pct_{t}'] = (
        (df[f'wr_10_{t}'] - df['wr_10_baseline'])
        / df['wr_10_baseline'] * 100
    )

# Linear regression, statistical significance, and scatter plots
for side in ['R', 'L']:
    for t in ['T1', 'T2']:
        pair = df[[f'pct_{side}_{t}', f'wr_pct_{t}']].dropna()

        if len(pair) >= 3:
            x = pair[f'pct_{side}_{t}'].to_numpy()
            y = pair[f'wr_pct_{t}'].to_numpy()

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            print(f'\n{side} side, {t}')
            print(f'N = {len(pair)}')
            print(f'Equation: y = {slope:.4f}x + {intercept:.4f}')
            print(f'R-squared = {r_value**2:.4f}')
            print(f'Slope p-value = {p_value:.4f}')
            print(f'Pearson r = {r_value:.4f}')

            # Plot data and regression line
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(x, y, label='Patients')

            line_x = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                line_x,
                slope * line_x + intercept,
                color='red',
                label=f'Linear fit ($R^2$ = {r_value**2:.3f})'
            )

            ax.axhline(0, color='gray', linewidth=0.8)
            ax.axvline(0, color='gray', linewidth=0.8)
            ax.set_xlabel('Fat fraction change from baseline (%)')
            ax.set_ylabel('Walk/run time change from baseline (%)')
            ax.set_title(
                f'{side} Side, {t}: Fat Fraction vs Walk/Run\n'
                f'Slope p = {p_value:.4f}'
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(f'{out}/correlation_{side}_{t}.png')
            plt.show()

# Individual patient percent change from baseline to T2

# Fat fraction: one bar per patient, right and left sides
# Individual patient percent-change bar graphs, sorted lowest to highest

# Fat fraction: one bar per patient for each side at T2
for side in ['R', 'L']:
    values = df[f'pct_{side}_T2'].dropna().sort_values()
    subjects = df.loc[values.index, 'subject_id'].astype(str)

    plt.figure(figsize=(12, 5))
    plt.bar(subjects, values)
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.xlabel('Patient ID')
    plt.ylabel('Fat fraction change from baseline (%)')
    plt.title(f'Individual Fat Fraction Change at T2 - {side} Side')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{out}/individual_fat_fraction_T2_{side}.png')
    plt.show()


# Walk/run time: one bar per patient at T2, sorted lowest to highest
values = df['wr_pct_T2'].dropna().sort_values()
subjects = df.loc[values.index, 'subject_id'].astype(str)

plt.figure(figsize=(12, 5))
plt.bar(subjects, values, color='orange')
plt.axhline(0, color='gray', linewidth=0.8)
plt.xlabel('Patient ID')
plt.ylabel('Walk/run time change from baseline (%)')
plt.title('Individual 10-Meter Walk/Run Time Change at T2')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{out}/individual_walk_run_T2.png')
plt.show()

# Calculate TUG time percent change from baseline
df['tug_pct_T2'] = (
    (df['tug_time_T2'] - df['tug_time_baseline'])
    / df['tug_time_baseline'] * 100
)

# Individual patient TUG graph, sorted lowest to highest
values = df.loc[df['tug_pct_T2'].notna(), 'tug_pct_T2']
values = values.loc[values.abs().sort_values().index]
subjects = df.loc[values.index, 'subject_id'].astype(str)

x = np.arange(len(values))

plt.figure(figsize=(12, 5))
plt.bar(x, values, color='green')


plt.axhline(0, color='gray', linewidth=0.8)
plt.xticks(x, subjects, rotation=90)
plt.xlabel('Patient ID')
plt.ylabel('TUG time change from baseline (%)')
plt.title('Individual TUG Time Change at T2')
plt.tight_layout()
plt.savefig(f'{out}/individual_tug_T2.png')
plt.show()