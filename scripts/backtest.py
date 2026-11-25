"""Backtest breadth thresholds against what the Nifty did next (2012 onward).

Historical breadth is measured on a point-in-time universe: each day, the 1,400 NSE stocks
with the highest 50-day average traded value (needs 200 sessions of history). That keeps the
universe comparable across years. The free archive holds only stocks still listed today, so
older readings carry some survivorship bias (delisted laggards are missing).

Writes public/data/signals.json: for every signal, its rule, each past trigger with the
Nifty 50 return 1, 3, 6 and 12 months later, summary stats vs all days, and today's status.
"""
import datetime as dt, glob, json, math, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'public', 'data')
START = '2010-01-01'          # load from here (EMA warm-up)
TEST_FROM = '2012-03-01'      # first date with Nifty 50 history + settled EMAs
TOP_N = 1400                  # matches the dashboard's ₹1,000 cr list within ~2 pts
H = {'1m': 21, '3m': 63, '6m': 126, '12m': 252}
DEBOUNCE = 40                 # sessions before the same signal can fire again


def r(x, n=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, n)


def load():
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(DATA, 'prices', '*.parquet')))
              if int(os.path.basename(p)[:4]) >= int(START[:4])]
    d = pd.concat(frames, ignore_index=True)
    extra = []
    for p in sorted(glob.glob(os.path.join(DATA, 'daily', '20*.csv'))):
        day = pd.Timestamp(os.path.basename(p)[:10])
        x = pd.read_csv(p, usecols=['symbol', 'open', 'high', 'low', 'close', 'volume']); x['date'] = day
        extra.append(x)
        ip = os.path.join(DATA, 'daily', f'idx-{day.date()}.csv')
        if os.path.exists(ip):
            i = pd.read_csv(ip, usecols=['symbol', 'close']); i['date'] = day; i['volume'] = 0
            extra.append(i)
    if extra:
        d = pd.concat([d] + extra, ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    d = d[d.date >= START].drop_duplicates(['symbol', 'date'], keep='last')
    ap = os.path.join(DATA, 'adjustments.csv')
    if os.path.exists(ap):
        for a in pd.read_csv(ap).itertuples():
            m = (d.symbol == a.symbol) & (d.date < pd.Timestamp(a.ex_date))
            d.loc[m, ['open', 'high', 'low', 'close']] *= a.factor
    return d


def breadth(d):
    is_idx = d.symbol.str.contains(' ') | d.symbol.str.startswith('NIFTY')
    eq = d[~is_idx]
    C = eq.pivot(index='date', columns='symbol', values='close').sort_index().astype('float64')
    V = eq.pivot(index='date', columns='symbol', values='volume').reindex_like(C).astype('float64')
    traded = C.notna()
    Cf = C.ffill()
    nobs = traded.cumsum()
    tv = (C * V).where(traded).rolling(50, min_periods=30).mean()
    elig = traded & (nobs >= 200)
    rank = tv.where(elig).rank(axis=1, ascending=False)
    U = rank <= TOP_N
    out = pd.DataFrame(index=C.index)
    for n in (20, 50, 200):
        e = Cf.ewm(span=n, adjust=False, min_periods=n).mean()
        out[f'p{n}'] = ((Cf > e) & U).sum(axis=1) / U.sum(axis=1).replace(0, np.nan) * 100
    h252 = C.rolling(252, min_periods=20).max()
    l252 = C.rolling(252, min_periods=20).min()
    out['nh'] = ((C >= h252 * 0.95) & U).sum(axis=1) / U.sum(axis=1) * 100
    out['nl'] = ((C <= l252 * 1.05) & U).sum(axis=1) / U.sum(axis=1) * 100
    idx = d[is_idx].pivot(index='date', columns='symbol', values='close').sort_index().ffill()
    out['nifty'] = idx['NIFTY 50'].reindex(out.index).ffill()
    return out[out.index >= TEST_FROM]


def fwd(nifty):
    return {k: (nifty.shift(-n) / nifty - 1) * 100 for k, n in H.items()}


def stats(dates, F):
    res = {}
    for k in H:
        v = F[k].reindex(dates).dropna()
        res[k] = {'n': int(len(v)), 'median': r(v.median()), 'mean': r(v.mean()),
                  'win': r((v > 0).mean() * 100, 0) if len(v) else None,
                  'worst': r(v.min()), 'best': r(v.max())}
    return res


ZONES = {
    'p200': [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 101)],
    'p20': [(0, 15), (15, 30), (30, 50), (50, 70), (70, 101)],
    'nh': [(0, 10), (10, 20), (20, 30), (30, 101)],
}


def zones(b, F, base):
    out = {}
    for col, bands in ZONES.items():
        rows = []
        for lo, hi in bands:
            m = (b[col] >= lo) & (b[col] < hi)
            runs = int((m & ~m.shift(1, fill_value=False)).sum())
            st = stats(b.index[m], F)
            rows.append({'lo': lo, 'hi': min(hi, 100), 'days': int(m.sum()), 'episodes': runs,
                         'share': r(m.mean() * 100, 0),
                         **{f'm{k}': st[k]['median'] for k in ('3m', '6m', '12m')},
                         **{f'w{k}': st[k]['win'] for k in ('3m', '6m', '12m')},
                         **{f'x{k}': r((st[k]['median'] or 0) - (base[k]['median'] or 0)) for k in ('3m', '6m', '12m')},
                         'now': bool(lo <= b[col].iloc[-1] < hi)})
        out[col] = rows
    return out


def debounce(mask, gap=DEBOUNCE):
    hits, last = [], -10 ** 9
    for i, (d, v) in enumerate(mask.items()):
        if v and i - last > gap:
            hits.append(d); last = i
    return hits


def cross_up(s, t, below=10):
    """first close above t after at least `below` sessions under it"""
    under = (s < t).astype(int).rolling(below).sum() == below
    return (s >= t) & under.shift(1, fill_value=False)


def cross_down(s, t, above=10):
    over = (s > t).astype(int).rolling(above).sum() == above
    return (s <= t) & over.shift(1, fill_value=False)


def thrust(p20, lo=25, hi=60, within=10):
    was_low = p20.rolling(within).min().shift(1) < lo
    return (p20 >= hi) & was_low


def main():
    b = breadth(load())
    F = fwd(b['nifty'])
    base = stats(b.index, F)
    asof = b.index[-1]

    # --- threshold grid for long-term breadth (200-day) crossing up, and 20-day washouts
    grid200 = []
    for t in range(30, 80, 5):
        ev = debounce(cross_up(b['p200'], t))
        s = stats(ev, F)
        grid200.append({'t': t, 'n': s['6m']['n'], 'm3': s['3m']['median'], 'm6': s['6m']['median'],
                        'w6': s['6m']['win'], 'm12': s['12m']['median'], 'w12': s['12m']['win']})
    grid20 = []
    for t in (5, 10, 15, 20, 25):
        ev = debounce(cross_down(b['p20'], t))
        s = stats(ev, F)
        grid20.append({'t': t, 'n': s['3m']['n'], 'm1': s['1m']['median'], 'm3': s['3m']['median'],
                       'w3': s['3m']['win'], 'm6': s['6m']['median'], 'w6': s['6m']['win']})
    gridnh = []
    for t in (10, 15, 20, 25, 30, 35):
        ev = debounce(cross_up(b['nh'], t))
        s = stats(ev, F)
        gridnh.append({'t': t, 'n': s['6m']['n'], 'm3': s['3m']['median'], 'm6': s['6m']['median'],
                       'w6': s['6m']['win'], 'm12': s['12m']['median'], 'w12': s['12m']['win']})

    # Fixed, round thresholds chosen up front (not optimised on this data, to avoid curve-fitting).
    defs = [
        ('washout', 'Washout', 'p20', cross_down(b['p20'], 15),
         '% of stocks above the 20-day EMA falls to 15% or lower', 15, 'below'),
        ('lt_oversold', 'Long-term oversold', 'p200', cross_down(b['p200'], 30),
         '% of stocks above the 200-day EMA falls below 30%', 30, 'below'),
        ('new_highs', 'Broad new highs', 'nh', cross_up(b['nh'], 30),
         '30%+ of stocks within 5% of their 52-week high', 30, 'above'),
        ('bull_zone', 'Bull line (SOIC)', 'p200', cross_up(b['p200'], 60),
         '% above the 200-day EMA climbs above 60%', 60, 'above'),
        ('thrust', 'Breadth thrust', 'p20', thrust(b['p20']),
         '% above the 20-day EMA jumps from under 25% to 60%+ within 10 sessions', 60, 'above'),
    ]
    signals = []
    for key, name, col, mask, rule, level, side in defs:
        ev = debounce(mask)
        st = stats(ev, F)
        events = [{'d': d.strftime('%Y-%m-%d'), 'v': r(b.at[d, col]),
                   **{k: r(F[k].get(d)) for k in H}} for d in ev]
        cur = b[col].iloc[-1]
        last = ev[-1] if ev else None
        since = int((b.index > last).sum()) if last is not None else None
        live = since is not None and since <= 63
        signals.append({
            'key': key, 'name': name, 'series': col, 'rule': rule, 'level': level, 'side': side,
            'now': r(cur), 'gap': r(cur - level),
            'last': last.strftime('%Y-%m-%d') if last is not None else None, 'since': since,
            'active': bool(live),
            'nifty_since': r((b['nifty'].iloc[-1] / b.at[last, 'nifty'] - 1) * 100) if last is not None else None,
            'stats': st, 'events': events,
            # edge vs an average day, judged on 6-month odds and median
            'edge': ('better' if (st['6m']['win'] or 0) >= (base['6m']['win'] or 0) + 8 and (st['6m']['median'] or 0) > (base['6m']['median'] or 0)
                     else 'worse' if (st['6m']['win'] or 0) <= (base['6m']['win'] or 0) - 8 else 'none'),
        })

    zs = zones(b, F, base)
    out = {
        'zones': zs,
        'asof': asof.strftime('%Y-%m-%d'), 'from': b.index[0].strftime('%Y-%m-%d'),
        'universe': f'Each day, the {TOP_N:,} NSE stocks with the highest 50-day average traded value (tracks this dashboard’s ₹1,000 cr list within ~2 points)',
        'benchmark': 'Nifty 50 price return',
        'baseline': base, 'signals': signals,
        'grids': {'p200_up': grid200, 'p20_down': grid20, 'nh_up': gridnh},
        'levels': {'p20_washout': 15, 'p200_oversold': 30, 'p200_bull': 60, 'nh_broad': 30},
        'now': {k: r(b[k].iloc[-1]) for k in ('p20', 'p50', 'p200', 'nh', 'nl')},
        'note': 'Past signals are a guide to odds, not a forecast. Built from stocks listed today, '
                'so older readings carry some survivorship bias.',
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'signals.json'), 'w') as fh:
        json.dump(out, fh, separators=(',', ':'), allow_nan=False)
    print('now', out['now'])
    for s in signals:
        st = s['stats']
        print(f"{s['name']:20} {s['edge']:6} n={st['6m']['n']:3}  3M med {st['3m']['median']}% win {st['3m']['win']}% | "
              f"6M med {st['6m']['median']}% win {st['6m']['win']}% | 12M med {st['12m']['median']}% "
              f"win {st['12m']['win']}% | last {s['last']} ({s['since']} sessions ago)")
    print('baseline', {k: (v['median'], v['win']) for k, v in base.items()})
    for col, rows in zs.items():
        print(col, [(f"{z['lo']}-{z['hi']}", z['days'], z['episodes'], z['m6m'], z['w6m'], z['m12m'], z['w12m']) for z in rows])
    print('p200 grid', [(g['t'], g['n'], g['m6'], g['w6']) for g in grid200])
    print('p20 grid', [(g['t'], g['n'], g['m3'], g['w3']) for g in grid20])
    print('nh grid', [(g['t'], g['n'], g['m6'], g['w6']) for g in gridnh])


if __name__ == '__main__':
    main()
