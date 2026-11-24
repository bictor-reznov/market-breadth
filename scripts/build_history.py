"""One-time: convert eod2_data per-symbol CSVs (split/bonus adjusted) into yearly parquet files.
Usage: python scripts/build_history.py <eod2_daily_dir> <out_dir>"""
import os, sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
frames = []
for f in sorted(os.listdir(src)):
    if not f.endswith('.csv'):
        continue
    try:
        df = pd.read_csv(os.path.join(src, f), usecols=lambda c: c in
                         ('Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'P/E'))
    except Exception as e:
        print('skip', f, e); continue
    if df.empty:
        continue
    df['symbol'] = f[:-4].upper()
    frames.append(df)
d = pd.concat(frames, ignore_index=True)
d = d.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                      'Close': 'close', 'Volume': 'volume', 'P/E': 'pe'})
d['date'] = pd.to_datetime(d['date'])
for c in ('open', 'high', 'low', 'close', 'pe'):
    d[c] = pd.to_numeric(d[c], errors='coerce').astype('float32')
d['volume'] = pd.to_numeric(d['volume'], errors='coerce').fillna(0).astype('int64')
d['symbol'] = d['symbol'].astype('category')
os.makedirs(out, exist_ok=True)
for y, g in d.groupby(d['date'].dt.year):
    g = g.sort_values(['symbol', 'date'])
    g['symbol'] = g['symbol'].astype(str)
    g.to_parquet(os.path.join(out, f'{y}.parquet'), index=False, compression='zstd')
print('rows', len(d), 'symbols', d['symbol'].nunique(), 'last', d['date'].max().date())
