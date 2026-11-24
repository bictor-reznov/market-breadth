"""Daily data fetch from NSE (runs in GitHub Actions after market close).

Downloads, for every trading day missing from the store:
  * CM bhavcopy (UDiFF)           -> data/daily/YYYY-MM-DD.csv     (main-board equity rows)
  * all-index close file          -> data/daily/idx-YYYY-MM-DD.csv (index OHLC + P/E, P/B, DY)
  * split/bonus adjustment factors, detected from NSE's adjusted previous close -> data/adjustments.csv
and appends today's FII/DII provisional cash figures to data/flows.csv.

Every step is best-effort: a failure is logged in data/update_log.json and the job carries on,
so one blocked request never stops the rest of the pipeline.
"""
import csv, datetime as dt, glob, io, json, os, sys, time, zipfile
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
DAILY = os.path.join(DATA, 'daily')
PRICES = os.path.join(DATA, 'prices')
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SERIES = {'EQ', 'BE', 'BZ'}
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/124.0 Safari/537.36')
log = {'run_at': dt.datetime.now(IST).isoformat(timespec='seconds'), 'steps': []}


def note(step, ok, detail=''):
    log['steps'].append({'step': step, 'ok': ok, 'detail': str(detail)[:300]})
    print(('OK  ' if ok else 'FAIL'), step, detail)


def session():
    s = requests.Session()
    s.headers.update({'User-Agent': UA, 'Accept': '*/*', 'Accept-Language': 'en-US,en;q=0.9',
                      'Referer': 'https://www.nseindia.com/'})
    try:  # cookies needed for the www.nseindia.com/api endpoints
        s.get('https://www.nseindia.com/', timeout=20)
    except Exception as e:
        note('nse homepage', False, e)
    return s


def get(s, url, tries=3):
    last = None
    for i in range(tries):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 200 and r.content:
                return r
            last = f'HTTP {r.status_code}'
            if r.status_code == 404:
                return None
        except Exception as e:
            last = e
        time.sleep(2 + 3 * i)
    raise RuntimeError(f'{url}: {last}')


def last_stored_date():
    dates = [os.path.basename(p)[:10] for p in glob.glob(os.path.join(DAILY, '20*.csv'))]
    if dates:
        return dt.date.fromisoformat(max(dates))
    years = sorted(glob.glob(os.path.join(PRICES, '*.parquet')))
    d = pd.read_parquet(years[-1], columns=['date'])
    return d['date'].max().date()


def last_close_map(before):
    """Latest stored close per symbol strictly before `before` (for adjustment detection)."""
    files = sorted(p for p in glob.glob(os.path.join(DAILY, '20*.csv'))
                   if os.path.basename(p)[:10] < before.isoformat())
    if files:
        d = pd.read_csv(files[-1], usecols=['symbol', 'close'])
        return dict(zip(d.symbol, d.close))
    years = sorted(glob.glob(os.path.join(PRICES, '*.parquet')))
    d = pd.read_parquet(years[-1], columns=['date', 'symbol', 'close'])
    d = d[d.date == d.date.max()]
    return dict(zip(d.symbol, d.close))


def fetch_bhavcopy(s, day):
    url = ('https://nsearchives.nseindia.com/content/cm/'
           f'BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip')
    r = get(s, url)
    if r is None:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]))
    raw = raw[raw['SctySrs'].isin(SERIES)]
    out = pd.DataFrame({
        'symbol': raw['TckrSymb'].astype(str).str.upper(), 'series': raw['SctySrs'],
        'name': raw['FinInstrmNm'], 'open': raw['OpnPric'], 'high': raw['HghPric'],
        'low': raw['LwPric'], 'close': raw['ClsPric'], 'prevclose': raw['PrvsClsgPric'],
        'volume': raw['TtlTradgVol'], 'turnover': raw['TtlTrfVal']})
    # one row per symbol: prefer EQ over BE/BZ
    out['rank'] = out['series'].map({'EQ': 0, 'BE': 1, 'BZ': 2})
    return out.sort_values('rank').drop_duplicates('symbol').drop(columns='rank')


def fetch_index_file(s, day):
    url = f'https://nsearchives.nseindia.com/content/indices/ind_close_all_{day:%d%m%Y}.csv'
    r = get(s, url)
    if r is None:
        return None
    d = pd.read_csv(io.StringIO(r.text))
    d.columns = [c.strip() for c in d.columns]
    num = lambda c: pd.to_numeric(d[c], errors='coerce')
    return pd.DataFrame({'symbol': d['Index Name'].astype(str).str.strip().str.upper(),
                         'open': num('Open Index Value'), 'high': num('High Index Value'),
                         'low': num('Low Index Value'), 'close': num('Closing Index Value'),
                         'pe': num('P/E'), 'pb': num('P/B'), 'dy': num('Div Yield')})


def detect_adjustments(day, bhav):
    prev = last_close_map(day)
    rows = []
    for sym, pc in zip(bhav.symbol, bhav.prevclose):
        old = prev.get(sym)
        if not old or not pc or old <= 0:
            continue
        f = pc / old
        if f < 0.97 or f > 1.03:  # NSE adjusts prev close on split / bonus / rights ex-dates
            rows.append({'symbol': sym, 'ex_date': day.isoformat(), 'factor': round(f, 6)})
    if rows:
        path = os.path.join(DATA, 'adjustments.csv')
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['symbol', 'ex_date', 'factor'])
            if new:
                w.writeheader()
            w.writerows(rows)
    return rows


def fetch_flows(s):
    r = get(s, 'https://www.nseindia.com/api/fiidiiTradeReact')
    data = r.json() if r is not None else []
    rec = {}
    for row in data:
        cat = row.get('category', '').upper()
        key = 'fii' if 'FII' in cat or 'FPI' in cat else 'dii' if 'DII' in cat else None
        if not key:
            continue
        d = dt.datetime.strptime(row['date'], '%d-%b-%Y').date().isoformat()
        rec['date'] = d
        for k in ('buyValue', 'sellValue', 'netValue'):
            rec[f'{key}_{k[:-5]}'] = float(str(row[k]).replace(',', ''))
    if 'fii_net' not in rec or 'dii_net' not in rec:
        raise RuntimeError(f'unexpected payload: {str(data)[:200]}')
    path = os.path.join(DATA, 'flows.csv')
    cols = ['date', 'fii_buy', 'fii_sell', 'fii_net', 'dii_buy', 'dii_sell', 'dii_net']
    have = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)
    if rec['date'] in set(have['date'].astype(str)):
        return f"{rec['date']} already stored"
    have = pd.concat([have, pd.DataFrame([rec])[cols]], ignore_index=True)
    have.sort_values('date').to_csv(path, index=False)
    return f"{rec['date']} FII {rec['fii_net']:+,.0f} cr, DII {rec['dii_net']:+,.0f} cr"


def main():
    os.makedirs(DAILY, exist_ok=True)
    s = session()
    today = dt.datetime.now(IST).date()
    # before ~18:30 IST today's final file is not out yet
    if dt.datetime.now(IST).hour < 18:
        today -= dt.timedelta(days=1)
    day = last_stored_date() + dt.timedelta(days=1)
    added = []
    while day <= today:
        if day.weekday() < 5:
            try:
                bhav = fetch_bhavcopy(s, day)
                if bhav is None:
                    note(f'bhavcopy {day}', True, 'no file (holiday)')
                else:
                    adj = detect_adjustments(day, bhav)
                    bhav.to_csv(os.path.join(DAILY, f'{day}.csv'), index=False)
                    added.append(day.isoformat())
                    note(f'bhavcopy {day}', True, f'{len(bhav)} rows, {len(adj)} adjustments')
                    try:
                        idx = fetch_index_file(s, day)
                        if idx is not None:
                            idx.to_csv(os.path.join(DAILY, f'idx-{day}.csv'), index=False)
                        note(f'indices {day}', idx is not None, '' if idx is not None else 'missing')
                    except Exception as e:
                        note(f'indices {day}', False, e)
            except Exception as e:
                note(f'bhavcopy {day}', False, e)
                break  # keep dates contiguous; retry next run
        day += dt.timedelta(days=1)
    try:
        note('fii/dii', True, fetch_flows(s))
    except Exception as e:
        note('fii/dii', False, e)
    log['added'] = added
    with open(os.path.join(DATA, 'update_log.json'), 'w') as fh:
        json.dump(log, fh, indent=1)


if __name__ == '__main__':
    main()
