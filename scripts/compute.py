"""Compute every dashboard metric from the stored prices and write JSON for the site.

Universe: StockScans ₹1,000 cr+ list (data/universe.csv), NSE main board only.
Outputs (public/data/): summary.json, breadth.json, industries.json, indices.json,
stocks.json, flows.json
"""
import datetime as dt, glob, json, math, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'public', 'data')

NEAR_HIGH = 0.05      # "near 52-week high": close within 5% of the 52-week high
NEAR_LOW = 0.05
HISTORY_YEARS = 3.4   # enough for a settled 200-day EMA plus two years of breadth history
CHART_SESSIONS = 500  # breadth history sent to the page (~2 years)

# Index scoreboard: (data symbol, label, group)
INDICES = [
    ('NIFTY 50', 'Nifty 50', 'Size'), ('NIFTY NEXT 50', 'Nifty Next 50', 'Size'),
    ('NIFTY 500', 'Nifty 500', 'Size'), ('NIFTY MIDCAP 150', 'Nifty Midcap 150', 'Size'),
    ('NIFTY SMALLCAP 250', 'Nifty Smallcap 250', 'Size'), ('NIFTY MICROCAP 250', 'Nifty Microcap 250', 'Size'),
    ('NIFTY BANK', 'Nifty Bank', 'Sector'), ('NIFTY PSU BANK', 'Nifty PSU Bank', 'Sector'),
    ('NIFTY PRIVATE BANK', 'Nifty Private Bank', 'Sector'),
    ('NIFTY FINANCIAL SERVICES', 'Nifty Financial Services', 'Sector'),
    ('NIFTY CAPITAL MARKETS', 'Nifty Capital Markets', 'Sector'), ('NIFTY IT', 'Nifty IT', 'Sector'),
    ('NIFTY PHARMA', 'Nifty Pharma', 'Sector'), ('NIFTY HEALTHCARE INDEX', 'Nifty Healthcare', 'Sector'),
    ('NIFTY FMCG', 'Nifty FMCG', 'Sector'), ('NIFTY AUTO', 'Nifty Auto', 'Sector'),
    ('NIFTY METAL', 'Nifty Metal', 'Sector'), ('NIFTY REALTY', 'Nifty Realty', 'Sector'),
    ('NIFTY ENERGY', 'Nifty Energy', 'Sector'), ('NIFTY OIL & GAS', 'Nifty Oil & Gas', 'Sector'),
    ('NIFTY INFRASTRUCTURE', 'Nifty Infrastructure', 'Sector'), ('NIFTY MEDIA', 'Nifty Media', 'Sector'),
    ('NIFTY CONSUMER DURABLES', 'Nifty Consumer Durables', 'Sector'),
    ('NIFTY CAPITAL GOODS', 'Nifty Capital Goods', 'Sector'), ('NIFTY CHEMICALS', 'Nifty Chemicals', 'Sector'),
    ('NIFTY INDIA DEFENCE', 'Nifty India Defence', 'Sector'), ('NIFTY CPSE', 'Nifty CPSE', 'Sector'),
    ('NIFTY POWER', 'Nifty Power', 'Sector'), ('NIFTY HOSPITALS', 'Nifty Hospitals', 'Sector'),
]
VALUATION = ['NIFTY 50', 'NIFTY NEXT 50', 'NIFTY 500', 'NIFTY MIDCAP 150', 'NIFTY SMALLCAP 250',
             'NIFTY BANK', 'NIFTY IT', 'NIFTY FMCG', 'NIFTY PHARMA', 'NIFTY AUTO', 'NIFTY METAL']


# ---------------------------------------------------------------- loading
def load_prices():
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(365.25 * HISTORY_YEARS))
    frames = []
    for p in sorted(glob.glob(os.path.join(DATA, 'prices', '*.parquet'))):
        if int(os.path.basename(p)[:4]) >= start.year:
            frames.append(pd.read_parquet(p))
    base = pd.concat(frames, ignore_index=True)
    base = base[base.date >= start]
    base_end = base.date.max()
    names, extra = {}, []
    for p in sorted(glob.glob(os.path.join(DATA, 'daily', '20*.csv'))):
        day = pd.Timestamp(os.path.basename(p)[:10])
        if day <= base_end:
            continue
        d = pd.read_csv(p)
        names.update(zip(d.symbol, d.name))
        d['date'] = day
        extra.append(d[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
        ip = os.path.join(DATA, 'daily', f'idx-{day.date()}.csv')
        if os.path.exists(ip):
            i = pd.read_csv(ip)
            i['date'] = day
            i['volume'] = 0
            extra.append(i[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'pe']])
    allp = pd.concat([base] + extra, ignore_index=True) if extra else base
    adj_path = os.path.join(DATA, 'adjustments.csv')
    if os.path.exists(adj_path):
        adj = pd.read_csv(adj_path)
        for r in adj.itertuples():
            m = (allp.symbol == r.symbol) & (allp.date < pd.Timestamp(r.ex_date))
            allp.loc[m, ['open', 'high', 'low', 'close']] *= r.factor
    allp = allp.drop_duplicates(['symbol', 'date'], keep='last')
    return allp, names


def wide(df, col, symbols):
    w = df[df.symbol.isin(symbols)].pivot(index='date', columns='symbol', values=col).sort_index()
    return w.reindex(columns=[s for s in symbols if s in w.columns])


# ---------------------------------------------------------------- helpers
def r1(x, n=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, n)


def pct_change(series, n):
    s = series.dropna()
    if len(s) <= n:
        return None
    return r1((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


def status_path(values, lo=30, hi=70):
    """StockScans-style status: >=hi outperforming, <=lo underperforming; in between it is
    'accumulating' if it last came from below, 'consolidating' if it last came from above."""
    state, out = None, []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(state); continue
        if v >= hi:
            state = 'outperforming'
        elif v <= lo:
            state = 'underperforming'
        elif state in ('outperforming', 'consolidating'):
            state = 'consolidating'
        elif state in ('underperforming', 'accumulating'):
            state = 'accumulating'
        else:
            state = 'accumulating' if v >= 50 else 'consolidating'
        out.append(state)
    return out


def status_matrix(score):
    """Vectorised status_path over a dates x names frame."""
    vals = score.values
    st = np.full(vals.shape, '', dtype=object)
    prev = np.full(vals.shape[1], '', dtype=object)
    for i in range(vals.shape[0]):
        v = vals[i]
        cur = prev.copy()
        nan = np.isnan(v)
        hi, lo = v >= 70, v <= 30
        mid = ~nan & ~hi & ~lo
        from_above = np.isin(prev, ['outperforming', 'consolidating'])
        from_below = np.isin(prev, ['underperforming', 'accumulating'])
        cur[hi] = 'outperforming'
        cur[lo] = 'underperforming'
        cur[mid & from_above] = 'consolidating'
        cur[mid & from_below] = 'accumulating'
        fresh = mid & ~from_above & ~from_below
        cur[fresh & (v >= 50)] = 'accumulating'
        cur[fresh & (v < 50)] = 'consolidating'
        st[i] = cur
        prev = cur
    return pd.DataFrame(st, index=score.index, columns=score.columns)


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    prices, names = load_prices()
    uni = pd.read_csv(os.path.join(DATA, 'universe.csv'))
    uni = uni[uni.symbol.isin(set(prices.symbol))]
    syms = list(uni.symbol)
    ind_of = dict(zip(uni.symbol, uni.industry))

    C = wide(prices, 'close', syms)
    H = wide(prices, 'high', syms).reindex_like(C)
    L = wide(prices, 'low', syms).reindex_like(C)
    dates = C.index
    asof = dates[-1]
    traded = C.notna()
    Cf = C.ffill()
    nobs = traded.cumsum()

    # --- EMAs and breadth
    ema = {n: Cf.ewm(span=n, adjust=False, min_periods=n).mean() for n in (20, 50, 200)}
    above, pct_above = {}, {}
    for n in (20, 50, 200):
        elig = traded & (nobs >= n)
        above[n] = (Cf > ema[n]) & elig
        pct_above[n] = above[n].sum(axis=1) / elig.sum(axis=1).replace(0, np.nan) * 100

    Hf, Lf = H.where(traded), L.where(traded)
    # "near" counts use closing prices (matches StockScans/SOIC levels best); new-high flags use intraday
    h252 = C.rolling(252, min_periods=20).max()
    l252 = C.rolling(252, min_periods=20).min()
    near_high = (C >= h252 * (1 - NEAR_HIGH)) & traded
    near_low = (C <= l252 * (1 + NEAR_LOW)) & traded
    prior_h = {k: Hf.shift(1).rolling(w, min_periods=int(w * 0.8)).max() for k, w in
               (('1m', 21), ('3m', 63), ('6m', 126), ('52w', 251))}
    prior_l = {k: Lf.shift(1).rolling(w, min_periods=int(w * 0.8)).min() for k, w in
               (('1m', 21), ('6m', 126), ('52w', 251))}
    new_high = {k: (Hf > v) & traded for k, v in prior_h.items()}
    new_low = {k: (Lf < v) & traded for k, v in prior_l.items()}
    ret1 = Cf.pct_change(fill_method=None) * 100
    ret1 = ret1.where(traded)
    up4, dn4 = (ret1 >= 4).sum(axis=1), (ret1 <= -4).sum(axis=1)
    adv, dec = (ret1 > 0).sum(axis=1), (ret1 < 0).sum(axis=1)
    r20 = (Cf / Cf.shift(20) - 1) * 100
    thrust_m = ((r20 >= 25) & traded).sum(axis=1) - ((r20 <= -25) & traded).sum(axis=1)

    # --- RS rating 1-99 (12 months, latest quarter double-weighted)
    raw = 2 * Cf / Cf.shift(63) + Cf / Cf.shift(126) + Cf / Cf.shift(189) + Cf / Cf.shift(252)
    raw = raw.where(traded)
    rs = (raw.rank(axis=1, pct=True) * 98 + 1).round()

    # --- stock strength score and status
    trend = 25 * (above[20].astype(int) + above[50].astype(int) + above[200].astype(int)
                  + ((ema[50] > ema[200]) & traded & (nobs >= 200)).astype(int))
    pos52 = ((C - l252) / (h252 - l252) * 100).clip(0, 100)
    score = (trend.where(nobs >= 200) + rs + pos52) / 3
    tail = score.iloc[-260:]
    stock_status = status_matrix(tail)

    # --- industry score: score an equal-weight index of each industry's stocks with the same
    #     recipe (EMA trend + RS vs other industries + 52-week position)
    ind_cols = pd.Series(ind_of).reindex(C.columns)
    dret = (Cf.pct_change(fill_method=None)).where(traded).clip(-0.2, 0.2)
    ind_ret = dret.T.groupby(ind_cols).mean().T
    imembers = traded.T.groupby(ind_cols).sum().T
    II = (1 + ind_ret.fillna(0)).cumprod().where(imembers.cumsum() > 0)
    i_ema = {n: II.ewm(span=n, adjust=False, min_periods=n).mean() for n in (20, 50, 200)}
    i_trend = 25 * sum((II > i_ema[n]).astype(int) for n in (20, 50, 200)) \
        + 25 * (i_ema[50] > i_ema[200]).astype(int)
    i_raw = 2 * II / II.shift(63) + II / II.shift(126) + II / II.shift(189) + II / II.shift(252)
    i_rs = i_raw.rank(axis=1, pct=True) * 98 + 1
    i_h, i_l = II.rolling(252, min_periods=60).max(), II.rolling(252, min_periods=60).min()
    i_pos = ((II - i_l) / (i_h - i_l) * 100).clip(0, 100)
    iscore = ((i_trend + i_rs + i_pos) / 3).where(i_ema[200].notna()).iloc[-260:]
    ist = status_matrix(iscore)

    # --- index data
    idx_syms = sorted({s for s, _, _ in INDICES} | set(VALUATION))
    IC = wide(prices, 'close', idx_syms).ffill()
    IH = wide(prices, 'high', idx_syms).ffill()
    IPE = wide(prices, 'pe', idx_syms) if 'pe' in prices.columns else pd.DataFrame()
    nifty = IC['NIFTY 50'] if 'NIFTY 50' in IC else None

    # ================================================================ breadth.json
    ch = dates[-CHART_SESSIONS:]
    series = []
    for d in ch:
        series.append({
            'd': d.strftime('%Y-%m-%d'),
            'p20': r1(pct_above[20][d]), 'p50': r1(pct_above[50][d]), 'p200': r1(pct_above[200][d]),
            'nh': int(near_high.loc[d].sum()), 'nl': int(near_low.loc[d].sum()),
            'h52': int(new_high['52w'].loc[d].sum()), 'l52': int(new_low['52w'].loc[d].sum()),
            'up4': int(up4[d]), 'dn4': int(dn4[d]), 'adv': int(adv[d]), 'dec': int(dec[d]),
            'thr': int(thrust_m[d]),
            'nifty': r1(IC['NIFTY 50'].get(d), 2) if nifty is not None else None,
        })
    # annotations: washouts (p20 local lows under 15) and thrusts (p20 from <25 to >60 in <=10 sessions)
    p20 = pct_above[20].loc[ch]
    cands = []
    for i in range(len(p20)):
        v = p20.iloc[i]
        if v < 15 and v == p20.iloc[max(0, i - 15): i + 16].min():
            cands.append((i, 'washout', v))
        if i >= 10 and v > 60 and p20.iloc[i - 10: i].min() < 25:
            cands.append((i, 'thrust', v))
    notes = []
    for kind, better in (('washout', min), ('thrust', max)):
        pts = [c for c in cands if c[1] == kind]
        # keep one point per episode: the extreme within 20 sessions
        while pts:
            pick = better(pts, key=lambda c: c[2])
            notes.append({'d': p20.index[pick[0]].strftime('%Y-%m-%d'), 'kind': kind, 'v': r1(pick[2])})
            pts = [c for c in pts if abs(c[0] - pick[0]) > 20]
    notes.sort(key=lambda n: n['d'])
    p200 = pct_above[200].loc[ch]
    best = p200.idxmax()
    notes.append({'d': best.strftime('%Y-%m-%d'), 'kind': 'best200', 'v': r1(p200.max())})
    breadth = {'asof': asof.strftime('%Y-%m-%d'), 'series': series, 'notes': notes}

    # ================================================================ tiles
    def at(series_, k):
        return series_.iloc[-1 - k] if len(series_) > k else np.nan
    look = dates[-60:]
    wash_d = pct_above[20].loc[look].idxmin()
    def tile(key, s, digits=1, is_pct=True):
        return {'key': key, 'now': r1(at(s, 0), digits), 'w1': r1(at(s, 5), digits),
                'm1': r1(at(s, 21), digits), 'wash': r1(s.get(wash_d), digits), 'pct': is_pct}
    nh_s, nl_s = near_high.sum(axis=1), near_low.sum(axis=1)
    tiles = [tile('Above 20-day EMA', pct_above[20]), tile('Above 50-day EMA', pct_above[50]),
             tile('Above 200-day EMA', pct_above[200]),
             tile('Near 52-week high', nh_s, 0, False), tile('Near 52-week low', nl_s, 0, False)]

    # ================================================================ industries.json
    last, m1 = iscore.index[-1], iscore.index[-22] if len(iscore) > 22 else iscore.index[0]
    stock_last = stock_status.iloc[-1]
    rows = []
    for ind in iscore.columns:
        members = [s for s in tail.columns if ind_of.get(s) == ind and traded[s].iloc[-1]]
        if not members or pd.isna(iscore.at[last, ind]):
            continue
        st_counts = stock_last[members].value_counts()
        r1m = (Cf[members].iloc[-1] / Cf[members].iloc[-22] - 1) * 100
        rows.append({
            'industry': ind, 'n': len(members),
            'score': r1(iscore.at[last, ind]), 'chg1m': r1(iscore.at[last, ind] - iscore.at[m1, ind]),
            'status': ist.at[last, ind] or None,
            'out': int(st_counts.get('outperforming', 0)), 'acc': int(st_counts.get('accumulating', 0)),
            'con': int(st_counts.get('consolidating', 0)), 'under': int(st_counts.get('underperforming', 0)),
            'a200': r1(above[200][members].iloc[-1].mean() * 100, 0),
            'rs': r1(rs[members].iloc[-1].median(), 0), 'ret1m': r1(r1m.median()),
            'nh': int(near_high[members].iloc[-1].sum()),
            'spark': [r1(v) for v in iscore[ind].iloc[-60:].tolist()],
        })
    rows.sort(key=lambda r: (r['score'] is None, -(r['score'] or 0)))
    counts = pd.Series([r['status'] for r in rows if r['n'] >= 3]).value_counts()
    industries = {'asof': asof.strftime('%Y-%m-%d'), 'rows': rows,
                  'counts': {k: int(counts.get(k, 0)) for k in
                             ('outperforming', 'accumulating', 'consolidating', 'underperforming')}}

    # ================================================================ indices.json
    irows = []
    ic_rs = {}
    for sym, label, grp in INDICES:
        if sym not in IC:
            continue
        s = IC[sym].dropna()
        ic_rs[sym] = (s.iloc[-1] / s.iloc[-64] if len(s) > 64 else np.nan)
    rs_rank = pd.Series(ic_rs).rank(pct=True) * 100
    ytd_base = IC.loc[IC.index < pd.Timestamp(asof.year, 1, 1)].iloc[-1] if (IC.index < pd.Timestamp(asof.year, 1, 1)).any() else None
    for sym, label, grp in INDICES:
        if sym not in IC:
            continue
        s = IC[sym].dropna()
        e = {n: s.ewm(span=n, adjust=False).mean().iloc[-1] for n in (20, 50, 200)}
        hi252 = IH[sym].dropna().iloc[-252:].max()
        lo252 = s.iloc[-252:].min()
        hi3y = IH[sym].dropna().iloc[-756:].max()
        trend_i = 25 * ((s.iloc[-1] > e[20]) + (s.iloc[-1] > e[50]) + (s.iloc[-1] > e[200]) + (e[50] > e[200]))
        pos = (s.iloc[-1] - lo252) / (hi252 - lo252) * 100 if hi252 > lo252 else 50
        sc = (trend_i + pos + rs_rank.get(sym, 50)) / 3
        irows.append({
            'sym': sym, 'name': label, 'group': grp, 'close': r1(s.iloc[-1], 2),
            'r1d': pct_change(s, 1), 'r1w': pct_change(s, 5), 'r1m': pct_change(s, 21),
            'r3m': pct_change(s, 63), 'r1y': pct_change(s, 252),
            'ytd': r1((s.iloc[-1] / ytd_base[sym] - 1) * 100) if ytd_base is not None else None,
            'from_high': r1((s.iloc[-1] / hi252 - 1) * 100),
            'from_3y_high': r1((s.iloc[-1] / hi3y - 1) * 100),
            'score': r1(sc), 'a20': bool(s.iloc[-1] > e[20]), 'a50': bool(s.iloc[-1] > e[50]),
            'a200': bool(s.iloc[-1] > e[200]),
        })
    vrows = []
    for sym in VALUATION:
        if IPE.empty or sym not in IPE:
            continue
        pe = IPE[sym].dropna()
        pe10 = pe[pe.index >= asof - pd.Timedelta(days=3653)]
        if pe10.empty:
            continue
        cur = pe.iloc[-1]
        vrows.append({'sym': sym, 'name': dict((s, l) for s, l, _ in INDICES).get(sym, sym.title()),
                      'pe': r1(cur), 'median10y': r1(pe10.median()),
                      'pctile10y': r1((pe10 < cur).mean() * 100, 0),
                      'min10y': r1(pe10.min()), 'max10y': r1(pe10.max()),
                      'since': pe10.index[0].strftime('%Y-%m-%d'),
                      'spark': [r1(v) for v in pe10.iloc[::5].tolist()]})
    indices = {'asof': asof.strftime('%Y-%m-%d'), 'rows': irows, 'valuation': vrows}

    # ================================================================ flows.json
    fp = os.path.join(DATA, 'flows.csv')
    flows = {'rows': [], 'source': 'NSE provisional cash-market data (₹ crore)'}
    if os.path.exists(fp):
        f = pd.read_csv(fp).sort_values('date')
        f['date'] = pd.to_datetime(f['date'])
        recent = f[f.date >= f.date.max() - pd.Timedelta(days=120)]
        flows['rows'] = [{'d': r.date.strftime('%Y-%m-%d'), 'fii': r1(r.fii_net, 0), 'dii': r1(r.dii_net, 0)}
                         for r in recent.itertuples()]
        if len(f):
            mtd = f[(f.date.dt.year == f.date.max().year) & (f.date.dt.month == f.date.max().month)]
            wk = f[f.date > f.date.max() - pd.Timedelta(days=7)]
            flows.update({'last': flows['rows'][-1],
                          'mtd': {'fii': r1(mtd.fii_net.sum(), 0), 'dii': r1(mtd.dii_net.sum(), 0),
                                  'days': int(len(mtd)), 'fii_buy_days': int((mtd.fii_net > 0).sum())},
                          'week': {'fii': r1(wk.fii_net.sum(), 0), 'dii': r1(wk.dii_net.sum(), 0)}})

    # ================================================================ regime checklist
    t_nh_hi, t_nh_mid = 300, 150  # SOIC's handoff signpost; universe and definition are close to theirs
    p200_now, p20_now = at(pct_above[200], 0), at(pct_above[20], 0)
    nh_now = int(at(nh_s, 0))
    oc, uc = industries['counts']['outperforming'], industries['counts']['underperforming']
    thrust_recent = [n for n in notes if n['kind'] == 'thrust'
                     and pd.Timestamp(n['d']) >= dates[-42]]
    wash_v = pct_above[20].get(wash_d)
    nifty_w = pct_change(IC['NIFTY 50'], 5) if nifty is not None else None
    p20_w = at(pct_above[20], 0) - at(pct_above[20], 5)
    mid = next((r for r in irows if r['sym'] == 'NIFTY MIDCAP 150'), None)
    nval = next((v for v in vrows if v['sym'] == 'NIFTY 50'), None)

    def v3(cond_in, cond_mixed):
        return 'in place' if cond_in else ('mixed' if cond_mixed else 'not yet')
    checks = [
        {'signal': 'Long-term breadth', 'reading': f'{p200_now:.1f}% of stocks above the 200-day EMA',
         'verdict': v3(p200_now >= 60, p200_now >= 50), 'rule': 'In place at 60%+, mixed 50–60%'},
        {'signal': 'Breadth thrust off a washout',
         'reading': (f'{wash_v:.1f}% above 20-day EMA on {wash_d:%d %b}; ' +
                     (f'thrust to {thrust_recent[-1]["v"]}% on {pd.Timestamp(thrust_recent[-1]["d"]):%d %b}'
                      if thrust_recent else 'no thrust above 60% since')),
         'verdict': v3(bool(thrust_recent), p20_now >= 50),
         'rule': 'In place if % above 20-day EMA jumped from under 25% to over 60% within 10 sessions (last 2 months)'},
        {'signal': 'Stocks near 52-week highs', 'reading': f'{nh_now} stocks within {NEAR_HIGH:.0%} of a 52-week high',
         'verdict': v3(nh_now >= t_nh_hi, nh_now >= t_nh_mid),
         'rule': f'In place at {t_nh_hi}+ (SOIC’s handoff level), mixed {t_nh_mid}+'},
        {'signal': 'Industry participation', 'reading': f'{oc} outperforming vs {uc} underperforming industries',
         'verdict': v3(oc > uc, oc >= 0.6 * uc), 'rule': 'In place when outperformers outnumber underperformers'},
        {'signal': 'Breadth confirming the index',
         'reading': (f'Nifty {nifty_w:+.1f}% this week; % above 20-day EMA {p20_w:+.1f} pts'
                     if nifty_w is not None else 'n/a'),
         'verdict': v3(nifty_w is not None and nifty_w > 0 and p20_w > 0,
                       nifty_w is not None and (nifty_w <= 0) == (p20_w <= 0)),
         'rule': 'In place when the index and breadth rise together over the week'},
    ]
    if mid:
        checks.append({'signal': 'Midcaps vs 3-year high',
                       'reading': f'Nifty Midcap 150 is {mid["from_3y_high"]:+.1f}% from its 3-year high',
                       'verdict': v3(mid['from_3y_high'] >= -0.5, mid['from_3y_high'] >= -5),
                       'rule': 'In place at a new high, mixed within 5%'})
    if nval:
        checks.append({'signal': 'Valuation cushion',
                       'reading': f'Nifty 50 at {nval["pe"]}x, {nval["pctile10y"]:.0f}th percentile of 10 years (median {nval["median10y"]}x)',
                       'verdict': v3(nval['pctile10y'] <= 30, nval['pctile10y'] <= 70),
                       'rule': 'In place below the 30th percentile, mixed up to the 70th'})
    if flows.get('mtd'):
        fm, dm = flows['mtd']['fii'], flows['mtd']['dii']
        few = flows['mtd']['days'] < 5
        note_ = f" — only {flows['mtd']['days']} session(s) collected so far" if few else ''
        checks.append({'signal': 'Foreign buyer (FII)', 'reading': f'FII net ₹{fm:+,.0f} cr this month ({flows["mtd"]["fii_buy_days"]} of {flows["mtd"]["days"]} sessions net buying){note_}',
                       'verdict': v3(not few and fm > 0 and flows['mtd']['fii_buy_days'] >= 0.6 * flows['mtd']['days'], fm > 0),
                       'rule': 'In place when FIIs are net buyers on most sessions this month'})
        checks.append({'signal': 'Domestic bid (DII)', 'reading': f'DII net ₹{dm:+,.0f} cr this month{note_}',
                       'verdict': v3(not few and dm > 0, dm > -0.25 * abs(fm or 1)), 'rule': 'In place when DIIs are net buyers this month'})
    tally = pd.Series([c['verdict'] for c in checks]).value_counts()

    # ================================================================ stocks.json
    i_last = -1
    srows = []
    pe_ss = dict(zip(uni.symbol, uni.pe))
    mcap = dict(zip(uni.symbol, uni.mcap))
    for s in syms:
        if not traded[s].iloc[i_last]:
            continue
        c = Cf[s]
        h = h252[s].iloc[i_last]
        srows.append({
            's': s, 'n': names.get(s, ''), 'i': ind_of.get(s), 'mc': r1(mcap.get(s), 0),
            'pe': r1(pe_ss.get(s)), 'c': r1(c.iloc[i_last], 2),
            'r1d': pct_change(c, 1), 'r1w': pct_change(c, 5), 'r1m': pct_change(c, 21),
            'r3m': pct_change(c, 63), 'r6m': pct_change(c, 126), 'r1y': pct_change(c, 252),
            'dh': r1((c.iloc[i_last] / h - 1) * 100) if h == h else None,
            'h52': bool(new_high['52w'][s].iloc[i_last]), 'h6m': bool(new_high['6m'][s].iloc[i_last]),
            'h3m': bool(new_high['3m'][s].iloc[i_last]), 'h1m': bool(new_high['1m'][s].iloc[i_last]),
            'rs': r1(rs[s].iloc[i_last], 0), 'rsw': r1(rs[s].iloc[-6], 0) if len(rs) > 6 else None,
            'sc': r1(score[s].iloc[i_last]), 'st': stock_status[s].iloc[-1] or None,
            'a20': bool(above[20][s].iloc[i_last]), 'a50': bool(above[50][s].iloc[i_last]),
            'a200': bool(above[200][s].iloc[i_last]),
        })

    # ================================================================ summary.json
    latest_log = {}
    lp = os.path.join(DATA, 'update_log.json')
    if os.path.exists(lp):
        latest_log = json.load(open(lp))
    summary = {
        'asof': asof.strftime('%Y-%m-%d'), 'generated': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
        'universe': {'stocks': int(traded.iloc[-1].sum()), 'listed': len(syms),
                     'rule': 'NSE main-board stocks above ₹1,000 cr market cap (StockScans list)',
                     'list_asof': str(uni['as_of'].iloc[0]) if 'as_of' in uni else None},
        'washout_date': wash_d.strftime('%Y-%m-%d'), 'tiles': tiles,
        'checks': checks, 'tally': {k: int(tally.get(k, 0)) for k in ('in place', 'mixed', 'not yet')},
        'today': {'adv': int(adv.iloc[-1]), 'dec': int(dec.iloc[-1]), 'up4': int(up4.iloc[-1]),
                  'dn4': int(dn4.iloc[-1]), 'h52': int(new_high['52w'].iloc[-1].sum()),
                  'l52': int(new_low['52w'].iloc[-1].sum()), 'h6m': int(new_high['6m'].iloc[-1].sum()),
                  'l6m': int(new_low['6m'].iloc[-1].sum()), 'h1m': int(new_high['1m'].iloc[-1].sum()),
                  'l1m': int(new_low['1m'].iloc[-1].sum())},
        'params': {'near_high': NEAR_HIGH, 'near_low': NEAR_LOW},
        'update': latest_log,
    }

    for name, obj in (('summary', summary), ('breadth', breadth), ('industries', industries),
                      ('indices', indices), ('stocks', {'asof': summary['asof'], 'rows': srows}),
                      ('flows', flows)):
        with open(os.path.join(OUT, f'{name}.json'), 'w') as fh:
            json.dump(obj, fh, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    print('as of', summary['asof'], '| stocks', summary['universe']['stocks'],
          '| >20/50/200 EMA', [t['now'] for t in tiles[:3]], '| near H/L', tiles[3]['now'], tiles[4]['now'])


if __name__ == '__main__':
    main()
