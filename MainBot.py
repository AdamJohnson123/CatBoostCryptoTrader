
import os
import json
import time
import math
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import ccxt
from catboost import CatBoostClassifier ,CatBoostRegressor
import math
# ------------------------ Configuration ------------------------
DEFAULT_CONFIG = {
    "exchange": "binance",
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "history_bars": 43800,
    "poll_interval_seconds": 60,
    "mode": "SIMULATE",  # SIMULATE or TESTNET
    
    "lookback": 100,
    "rsi_len": 14,
    "adx_len": 14,
    "atr_len": 14,
    "vwap_len": 20,
    "norm_len": 50,
    "min_return_threshold": 0.0001,
    # SMC params
    "swing_lookback": 5,
    "order_block_lookback": 20,
    # risk
    "fee": 0.0004,
    "slippage": 0.0005,
    "pos_pct": 1.0,  # full position by default
    "initial_capital": 1.0,
    "atr_mult_stop": 2.0,
    "breakeven_at_pct": 0.05,  # 5%
    "trailing_atr_mult": 1.0,
    "max_drawdown_pct": 35.0,
}

CONFIG_PATH = os.environ.get('PAPER_TRADE_CONFIG', 'paper_trade_config.json')


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                user = json.load(f)
            cfg.update(user)
        except Exception as e:
            print(f"Failed to read config {path}: {e}")
    else:
        # write default to help user
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"Wrote default config to {path} - edit as needed and re-run")
    return cfg


# ------------------------ Logging & DB ------------------------

def setup_logging(logfile: str = 'smc_bot.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )


class TradeDB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                symbol TEXT,
                side TEXT,
                price REAL,
                size REAL,
                notional REAL,
                fee REAL,
                pnl REAL,
                note TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        self.conn.commit()

    def record_trade(self, symbol, side, price, size, notional, fee, pnl, note=''):
        cur = self.conn.cursor()
        cur.execute('''INSERT INTO trades (ts,symbol,side,price,size,notional,fee,pnl,note) VALUES (?,?,?,?,?,?,?,?,?)''',
                    (datetime.now(timezone.utc).isoformat(), symbol, side, price, size, notional, fee, pnl, note))
        self.conn.commit()
        logging.info(f"Recorded trade {side} {symbol} price={price} size={size} pnl={pnl}")

    def save_state(self, key, value):
        cur = self.conn.cursor()
        cur.execute('''INSERT OR REPLACE INTO state (key,value) VALUES (?,?)''', (key, json.dumps(value)))
        self.conn.commit()

    def load_state(self, key):
        cur = self.conn.cursor()
        cur.execute('''SELECT value FROM state WHERE key=?''', (key,))
        r = cur.fetchone()
        return json.loads(r[0]) if r else None


# ------------------------ Exchange helpers ------------------------

def init_exchange(config: Dict[str, Any]) -> ccxt.Exchange:
    api_key = os.environ.get('EXCHANGE_APIKEY') or config.get('apiKey')
    api_secret = os.environ.get('EXCHANGE_SECRET') or config.get('secret')
    api_pass = os.environ.get('EXCHANGE_PASSWORD') or config.get('password')

    exchange_args = {
        'enableRateLimit': True,
    }
    if api_key and api_secret:
        exchange_args.update({'apiKey': api_key, 'secret': api_secret})
        if api_pass:
            exchange_args['password'] = api_pass

    exchange = getattr(ccxt, config.get('exchange', 'binance'))(exchange_args)

    if config.get('mode', 'SIMULATE').upper() == 'TESTNET':
        try:
            exchange.set_sandbox_mode(True)
            logging.info('Enabled ccxt sandbox mode (if supported)')
        except Exception:
            logging.warning('ccxt sandbox mode not supported - ensure testnet endpoint is configured')

    return exchange


def fetch_ohlcv_df(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    return df


# ------------------------ Indicators & CatBoost prediction (from your Pine) ------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - 100 / (1 + rs)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up = df['high'] - df['high'].shift(1)
    down = df['low'].shift(1) - df['low']
    plusDM = up.where((up > down) & (up > 0), 0.0)
    minusDM = down.where((down > up) & (down > 0), 0.0)
    tr = (pd.concat([(df['high'] - df['low']).abs(), (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1))
    trur = tr.rolling(length).mean()
    plusDI = 100 * plusDM.rolling(length).mean() / (trur + 1e-12)
    minusDI = 100 * minusDM.rolling(length).mean() / (trur + 1e-12)
    dx = 100 * (plusDI - minusDI).abs() / (plusDI + minusDI + 1e-12)
    adx = dx.rolling(length).mean()
    return adx


def calc_rolling_vwap(df: pd.DataFrame, length: int = 20) -> pd.Series:
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = (typical_price * df['volume']).rolling(length).sum()
    v = df['volume'].rolling(length).sum()
    return (pv / v).fillna(typical_price)


def taker_ratio(df: pd.DataFrame, period: int = 10) -> pd.Series:
    is_buy = df['close'] > df['open']
    buy_vol = df['volume'].where(is_buy, 0.0)
    sell_vol = df['volume'].where(~is_buy, 0.0)
    total_buy = buy_vol.rolling(period).sum()
    total_sell = sell_vol.rolling(period).sum()
    total_vol = df['volume'].rolling(period).sum()
    return ((total_buy - total_sell) / total_vol).fillna(0.0)


def zscore(series: pd.Series, length: int = 50) -> pd.Series:
    return ((series - series.rolling(length).mean()) / (series.rolling(length).std() + 1e-12)).fillna(0.0)





def compute_prediction_series(df: pd.DataFrame, model, cfg: Dict[str, Any]) -> float:
    if model is None:
        return 0.0
    
    # 1. 仅计算最新的特征
    features = extract_features_from_df(df, cfg) # 建议抽离出的函数
    last_feature = features[-1].reshape(1, -1)
    
    # 2. 直接推理
    prediction = model.predict(last_feature)[0]
    return prediction

# compute features
    # RSI
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta>0, delta, 0.0)
    loss = np.where(delta<0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(rsi_len).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(rsi_len).mean().to_numpy()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi_v = 100 - 100/(1+rs)

    # ADX
    up = high - np.roll(high,1)
    down = np.roll(low,1) - low
    up[0]=0; down[0]=0
    plusDM = np.where((up>down) & (up>0), up, 0.0)
    minusDM = np.where((down>up) & (down>0), down, 0.0)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close,1))
    tr3 = np.abs(low - np.roll(close,1))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    trur = pd.Series(tr).rolling(adx_len).mean().to_numpy()
    plusDI = 100 * pd.Series(plusDM).rolling(adx_len).mean().to_numpy() / (trur + 1e-12)
    minusDI = 100 * pd.Series(minusDM).rolling(adx_len).mean().to_numpy() / (trur + 1e-12)
    dx = 100 * np.abs(plusDI - minusDI) / (plusDI + minusDI + 1e-12)
    adx_v = pd.Series(dx).rolling(adx_len).mean().to_numpy()

    # VWAP
    typical = (high + low + close)/3.0
    pv = pd.Series(typical * volume).rolling(vwap_len).sum().to_numpy()
    v = pd.Series(volume).rolling(vwap_len).sum().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap_v = np.where(v>0, pv / v, typical)

    # ATR
    tr = tr
    atr_v = pd.Series(tr).rolling(atr_len).mean().to_numpy()

    # taker ratio
    is_buy = close > open_
    buy_vol = np.where(is_buy, volume, 0.0)
    sell_vol = np.where(~is_buy, volume, 0.0)
    total_buy = pd.Series(buy_vol).rolling(10).sum().to_numpy()
    total_sell = pd.Series(sell_vol).rolling(10).sum().to_numpy()
    total_vol = pd.Series(volume).rolling(10).sum().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        taker_v = np.where(total_vol>0, (total_buy - total_sell) / total_vol, 0.0)

    # zscore normalization (FIXED: Use x.iloc[-1] for scalar return)
    nrsi = pd.Series(rsi_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    nadx = pd.Series(adx_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    nvwap = pd.Series(vwap_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    natr = pd.Series(atr_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    nvol = pd.Series(volume).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    ntaker = pd.Series(taker_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()

    # features matrix
    features = np.vstack([np.nan_to_num(nrsi), np.nan_to_num(nadx), np.nan_to_num(nvwap), np.nan_to_num(natr), np.nan_to_num(nvol), np.nan_to_num(ntaker)]).T
    prediction = np.full(N, 0.5, dtype=float)
    labels = np.zeros(N, dtype=int)
    for g in range(N):
        fidx = g + n_neighbors
        if fidx < N:
            labels[g] = 1 if (close[fidx] - close[g]) > 0 else 0

    for idx in range(lookback, N):
        hist_start = idx - lookback
        hist_features = features[hist_start:idx]
        hist_labels = labels[hist_start:idx]
        valid_mask = ~np.isnan(hist_features).any(axis=1)
        if not valid_mask.any():
            continue
        train_features = hist_features[valid_mask]
        train_labels = hist_labels[valid_mask]
        if len(train_features) < 10:  # Minimum samples to train reliably
            continue
        model = CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, verbose=False, random_seed=42)
        model.fit(train_features, train_labels)
        cur = features[idx].reshape(1, -1)
        if np.isnan(cur).any():
            continue
        pred_prob = model.predict_proba(cur)[0, 1]  # Probability of class 1 (price up)
        prediction[idx] = pred_prob

    return prediction


# ------------------------ SMC structure & zones (simplified faithful port) ------------------------

def detect_swings(df: pd.DataFrame, swing_lookback: int = 5):
    # detect simple swing highs/lows similar to pine's pivot logic
    highs = df['high']
    lows = df['low']
    swing_high_idx = []
    swing_low_idx = []
    for i in range(swing_lookback, len(df)-swing_lookback):
        window_high = highs[i-swing_lookback:i+swing_lookback+1]
        if highs[i] == window_high.max():
            swing_high_idx.append(i)
        window_low = lows[i-swing_lookback:i+swing_lookback+1]
        if lows[i] == window_low.min():
            swing_low_idx.append(i)
    df['is_swing_high'] = False
    df['is_swing_low'] = False
    df.loc[swing_high_idx, 'is_swing_high'] = True
    df.loc[swing_low_idx, 'is_swing_low'] = True
    return df


def compute_structure(df: pd.DataFrame, lookback: int = 50):
    # build a lightweight HH/HL/LH/LL structure tracker
    swings = []
    # parse detected swing points into sequence of highs/lows
    for i, row in df.iterrows():
        if row.get('is_swing_high'):
            swings.append((i, 'H', row['high']))
        if row.get('is_swing_low'):
            swings.append((i, 'L', row['low']))
    # reduce to most recent swings
    swings = swings[-(lookback*2):]
    # derive simple trend bias
    bias = 0  # 1 bullish, -1 bearish, 0 neutral
    if len(swings) >= 4:
        types = [s[1] for s in swings]
        # naive HH/HL detection
        highs = [s for s in swings if s[1]=='H']
        lows = [s for s in swings if s[1]=='L']
        if len(highs)>=2 and highs[-1][2] > highs[-2][2] and len(lows)>=2 and lows[-1][2] > lows[-2][2]:
            bias = 1
        elif len(highs)>=2 and highs[-1][2] < highs[-2][2] and len(lows)>=2 and lows[-1][2] < lows[-2][2]:
            bias = -1
    return bias


def detect_order_blocks(df: pd.DataFrame, window: int = 20):
    # simplified order block detection: last bearish/bullish imbalanced candle block
    obs = []
    for i in range(window, len(df)):
        slice_df = df.iloc[i-window:i]
        # bullish OB candidate: last significant bullish engulfing area
        # we store an OB as (start_idx, end_idx, high, low, bias)
        # simple heuristic: if slice close is higher than open overall -> bullish zone
        if slice_df['close'].iloc[-1] > slice_df['open'].iloc[0]:
            high = slice_df['high'].max()
            low = slice_df['low'].min()
            obs.append({'time': df['datetime'].iloc[i], 'high': high, 'low': low, 'bias': 'bullish', 'start': i-window, 'end': i-1})
        else:
            high = slice_df['high'].max()
            low = slice_df['low'].min()
            obs.append({'time': df['datetime'].iloc[i], 'high': high, 'low': low, 'bias': 'bearish', 'start': i-window, 'end': i-1})
    return pd.DataFrame(obs)


def detect_fvg(df: pd.DataFrame, extend: int = 1):
    # Fair value gap (three-bar gap) simplistic detection
    fvg_list = []
    for i in range(2, len(df)):
        # bullish FVG: middle bar has no overlap with prior/next bodies
        b1_open, b1_close = df['open'].iat[i-2], df['close'].iat[i-2]
        b2_open, b2_close = df['open'].iat[i-1], df['close'].iat[i-1]
        b3_open, b3_close = df['open'].iat[i], df['close'].iat[i]
        # bullish gap condition
        top_b1 = max(b1_open, b1_close)
        bottom_b2 = min(b2_open, b2_close)
        if bottom_b2 > top_b1:
            fvg_list.append({'idx': i-1, 'type': 'bull', 'top': bottom_b2, 'bottom': top_b1})
        # bearish gap
        bottom_b1 = min(b1_open, b1_close)
        top_b2 = max(b2_open, b2_close)
        if top_b2 < bottom_b1:
            fvg_list.append({'idx': i-1, 'type': 'bear', 'top': top_b1, 'bottom': bottom_b2})
    return pd.DataFrame(fvg_list)


# ------------------------ SMC-based position sizing & order decision ------------------------

def calculate_size_from_risk(capital: float, price: float, stop_price: float, risk_pct: float = 0.02):
    # risk_pct: fraction of capital willing to lose per trade
    risk_amount = capital * risk_pct
    # for spot coin amount = risk_amount / abs(entry-stop)
    unit_risk = abs(price - stop_price)
    if unit_risk <= 0:
        return 0.0
    size = risk_amount / unit_risk
    return size


# ------------------------ Execution & Simulation ------------------------

def simulate_entry(capital: float, price: float, cfg: Dict[str, Any], stop_price: float):
    pos_notional = capital * cfg['pos_pct']
    size = pos_notional / price
    # account for slippage on entry
    entry_price = price * (1.0 + cfg['slippage'])
    return {'entry_price': entry_price, 'size': size, 'notional': pos_notional, 'stop_price': stop_price}


def simulate_exit(entry, price_exit: float, cfg: Dict[str, Any]):
    exit_price = price_exit * (1.0 - cfg['slippage'])
    gross_pnl = (exit_price - entry['entry_price']) * entry['size']
    fee_total = entry['notional'] * cfg['fee'] * 2
    net_pnl = gross_pnl - fee_total
    return {'exit_price': exit_price, 'gross_pnl': gross_pnl, 'fee': fee_total, 'net_pnl': net_pnl}


# ------------------------ ART Stoploss (ATR-based with structure) ------------------------

def compute_art_stop(entry_price: float, direction: str, last_swing_price: float, atr_val: float, atr_mult: float):
    # direction: 'long' or 'short'
    if direction == 'long':
        # stop placed below last swing low minus ATR buffer
        stop = min(last_swing_price - atr_val * atr_mult, entry_price * (1 - 0.01))
    else:
        stop = max(last_swing_price + atr_val * atr_mult, entry_price * (1 + 0.01))
    return stop


# ------------------------ Main engine ------------------------
class SMCTrader:
    def __init__(self, cfg: Dict[str, Any]):
        # ... (keep existing)
        self.model_path = "catboost_model3.cbm"
        if os.path.exists(self.model_path):
            self.model = CatBoostRegressor().load_model(self.model_path)
            self.last_retrain_ts = os.path.getmtime(self.model_path)
        else:
            self.model = None
            logging.warning("No model found—will train on first step")
        self.cached_df = None  # For efficiency
    def shutdown(self):
        logging.info('Shutting down gracefully')
        self.running = False


def step(self):
    try:
        # Cache recent data for efficiency
        new_bars = 100  # Fetch only recent
        new_df = fetch_ohlcv_df(self.exchange, self.cfg['symbol'], self.cfg['timeframe'], new_bars)
        if not new_df.empty:
            if self.cached_df is not None:
                self.cached_df = pd.concat([self.cached_df, new_df]).drop_duplicates('datetime').tail(self.cfg['history_bars'])
            else:
                self.cached_df = new_df
        df = self.cached_df or new_df
        if df.empty:
            logging.warning('No OHLCV returned - skipping this cycle')
            return

        # Compute indicators
        df['ema20'] = ema(df['close'], 20)
        df['ema100'] = ema(df['close'], 100)
        df['rsi'] = rsi(df['close'], self.cfg['rsi_len'])
        df['atr'] = calc_atr(df, self.cfg['atr_len'])
        df = detect_swings(df, self.cfg['swing_lookback'])
        ob_df = detect_order_blocks(df, self.cfg['order_block_lookback'])
        fvg_df = detect_fvg(df)
        bias = compute_structure(df)

        # Periodic retrain
        if self.model is None or (time.time() - getattr(self, 'last_retrain_ts', 0) > 86400):
            self.model = train_initial_model(self.cfg, self.model_path)
            self.last_retrain_ts = time.time()

        # Prediction (inference only)
        pred = compute_prediction_series(df, self.model, self.cfg)
        last_pred_return = pred[-1]  # Expected return
        price = float(df['close'].iat[-1])

        # Find last swing (for stops)
        last_swing_low = None
        last_swing_high = None
        swings_idx = df.index[df['is_swing_low'] | df['is_swing_high']].tolist()
        if swings_idx:
            for si in reversed(swings_idx):
                if df.at[si, 'is_swing_low'] and last_swing_low is None:
                    last_swing_low = df.at[si, 'low']
                if df.at[si, 'is_swing_high'] and last_swing_high is None:
                    last_swing_high = df.at[si, 'high']
                if last_swing_low is not None and last_swing_high is not None:
                    break

        # New signals (regression-gated SMC)
        buy_signal = False
        sell_signal = False
        latest_ob = ob_df.iloc[-1] if not ob_df.empty else None
        ml_buy = last_pred_return > self.cfg['min_return_threshold']
        ml_sell = last_pred_return < -self.cfg['min_return_threshold']

        if latest_ob is not None:
            if latest_ob['bias'] == 'bullish' and price <= latest_ob['high'] and ml_buy:
                buy_signal = True
            if latest_ob['bias'] == 'bearish' and price >= latest_ob['low'] and ml_sell:
                sell_signal = True
        else:
            if ml_buy and bias == 1:
                buy_signal = True
            if ml_sell and bias == -1:
                sell_signal = True

        logging.info(f"Signals: buy={buy_signal} sell={sell_signal} pred_return={last_pred_return:.4f} bias={bias}")

        # Execute (unchanged, but uses last_swing_low)
        if self.position is None and buy_signal:
            atr_val = float(df['atr'].iat[-1]) if not np.isnan(df['atr'].iat[-1]) else 0.0
            last_swing = last_swing_low if last_swing_low is not None else price * 0.98
            stop_price = compute_art_stop(price, 'long', last_swing, atr_val, self.cfg['atr_mult_stop'])
            entry = simulate_entry(self.capital, price, self.cfg, stop_price)
            self.position = {'side': 'LONG', 'entry': entry}
            self.db.record_trade(self.cfg['symbol'], 'BUY', entry['entry_price'], entry['size'], entry['notional'], 0.0, 0.0, note='SIMULATED SMC ENTRY')
            logging.info(f"SIMULATED ENTRY at {entry['entry_price']:.2f} stop={stop_price:.2f}")

        elif self.position is not None and sell_signal and self.position['side'] == 'LONG':
            exit_res = simulate_exit(self.position['entry'], price, self.cfg)
            pnl = exit_res['net_pnl']
            self.capital += pnl  # Fixed: no / capital (it's absolute)
            self.db.record_trade(self.cfg['symbol'], 'SELL', exit_res['exit_price'], self.position['entry']['size'], self.position['entry']['notional'], exit_res['fee'], exit_res['net_pnl'], note='SIMULATED SMC EXIT')
            logging.info(f"SIMULATED EXIT net_pnl={pnl:.6f}, new capital={self.capital:.6f}")
            self.position = None

        # Trailing/breakeven/stop (unchanged, but fixed trailing_stop calc for long)
        if self.position is not None:
            entry = self.position['entry']
            current_price = price
            entry_price = entry['entry_price']
            size = entry['size']
            unreal_pct = (current_price - entry_price) / (entry_price + 1e-12)
            if unreal_pct >= self.cfg['breakeven_at_pct']:
                entry['stop_price'] = entry_price
                logging.info('Moved stop to breakeven')
            atr_val = float(df['atr'].iat[-1]) if not np.isnan(df['atr'].iat[-1]) else 0.0
            # Fixed trailing for long: below current
            if unreal_pct > 0 and current_price - (self.cfg['trailing_atr_mult'] * atr_val) > entry.get('stop_price', -np.inf):
                entry['stop_price'] = current_price - (self.cfg['trailing_atr_mult'] * atr_val)
                logging.info(f"Adjusted trailing stop to {entry['stop_price']:.6f}")
            if entry.get('stop_price') is not None and current_price <= entry['stop_price']:
                exit_res = simulate_exit(entry, current_price, self.cfg)
                pnl = exit_res['net_pnl']
                self.capital += pnl
                self.db.record_trade(self.cfg['symbol'], 'SELL', exit_res['exit_price'], size, entry['notional'], exit_res['fee'], exit_res['net_pnl'], note='SIMULATED STOP EXIT')
                logging.info(f"STOP EXIT net_pnl={pnl:.6f}, new capital={self.capital:.6f}")
                self.position = None

    except Exception as e:
        logging.exception(f"Error in step: {e}")

            # Execute simulated orders
            if self.position is None and buy_signal:
                # compute stop using ART + swing low
                atr_val = float(df['atr'].iat[-1]) if not np.isnan(df['atr'].iat[-1]) else 0.0
                last_swing = last_swing_low if last_swing_low is not None else price * 0.98
                stop_price = compute_art_stop(price, 'long', last_swing, atr_val, self.cfg['atr_mult_stop'])
                entry = simulate_entry(self.capital, price, self.cfg, stop_price)
                self.position = {'side': 'LONG', 'entry': entry}
                self.db.record_trade(self.cfg['symbol'], 'BUY', entry['entry_price'], entry['size'], entry['notional'], 0.0, 0.0, note='SIMULATED SMC ENTRY')
                logging.info(f"SIMULATED ENTRY at {entry['entry_price']:.2f} stop={stop_price:.2f}")

            elif self.position is not None and sell_signal and self.position['side'] == 'LONG':
                exit_res = simulate_exit(self.position['entry'], price, self.cfg)
                pnl = exit_res['net_pnl']
                self.capital += pnl
                self.db.record_trade(self.cfg['symbol'], 'SELL', exit_res['exit_price'], self.position['entry']['size'], self.position['entry']['notional'], exit_res['fee'], exit_res['net_pnl'], note='SIMULATED SMC EXIT')
                logging.info(f"SIMULATED EXIT net_pnl={pnl:.6f}, new capital approx={self.capital:.6f}")
                self.position = None

            # if in position, apply breakeven / trailing stop rules based on ATR and floating PnL
            if self.position is not None:
                entry = self.position['entry']
                current_price = price
                entry_price = entry['entry_price']
                size = entry['size']
                # current unrealized return for position
                unreal_pct = (current_price - entry_price) / (entry_price + 1e-12)
                # move stop to breakeven
                if unreal_pct >= self.cfg['breakeven_at_pct']:
                    # set stop to entry_price
                    entry['stop_price'] = entry_price
                    logging.info('Moved stop to breakeven')
                # trailing by ATR
                atr_val = float(df['atr'].iat[-1]) if not np.isnan(df['atr'].iat[-1]) else 0.0
                trailing_stop = entry_price + (self.cfg['trailing_atr_mult'] * atr_val)
                if unreal_pct > 0 and current_price - (self.cfg['trailing_atr_mult'] * atr_val) > entry.get('stop_price', -1e9):
                    entry['stop_price'] = current_price - (self.cfg['trailing_atr_mult'] * atr_val)
                    logging.info(f"Adjusted trailing stop to {entry['stop_price']:.6f}")
                # check stop hit
                if entry.get('stop_price') is not None and current_price <= entry['stop_price']:
                    exit_res = simulate_exit(entry, current_price, self.cfg)
                    pnl = exit_res['net_pnl']
                    self.capital += pnl
                    self.db.record_trade(self.cfg['symbol'], 'SELL', exit_res['exit_price'], size, entry['notional'], exit_res['fee'], exit_res['net_pnl'], note='SIMULATED STOP EXIT')
                    logging.info(f"STOP EXIT net_pnl={pnl:.6f}, new capital approx={self.capital:.6f}")
                    self.position = None

        except Exception as e:
            logging.exception(f"Error in step: {e}")

    def run(self):
        logging.info('SMCTrader started in %s mode', self.cfg.get('mode'))
        while self.running:
            start = time.time()
            self.step()
            elapsed = time.time() - start
            wait = max(1.0, self.cfg.get('poll_interval_seconds', 60) - elapsed)
            time.sleep(wait)



#debud

def generate_features_and_labels(df: pd.DataFrame, cfg: Dict[str, Any]):
    prediction_horizon = cfg.get('prediction_horizon', 7)
    rsi_len = cfg.get('rsi_len', 14)
    adx_len = cfg.get('adx_len', 14)
    atr_len = cfg.get('atr_len', 14)
    vwap_len = cfg.get('vwap_len', 20)
    norm_len = cfg.get('norm_len', 50)

    close = df['close'].to_numpy(dtype=float)
    high = df['high'].to_numpy(dtype=float)
    low = df['low'].to_numpy(dtype=float)
    open_ = df['open'].to_numpy(dtype=float)
    volume = df['volume'].to_numpy(dtype=float)
    timestamps = df['datetime'].to_numpy(dtype='datetime64[ns]')
    N = len(df)

    # === Indicators (unchanged) ===
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(rsi_len).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(rsi_len).mean().to_numpy()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi_v = 100 - 100 / (1 + rs)

    up = high - np.roll(high, 1)
    down = np.roll(low, 1) - low
    up[0] = down[0] = 0
    plusDM = np.where((up > down) & (up > 0), up, 0.0)
    minusDM = np.where((down > up) & (down > 0), down, 0.0)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    trur = pd.Series(tr).rolling(adx_len).mean().to_numpy()
    plusDI = 100 * pd.Series(plusDM).rolling(adx_len).mean().to_numpy() / (trur + 1e-12)
    minusDI = 100 * pd.Series(minusDM).rolling(adx_len).mean().to_numpy() / (trur + 1e-12)
    dx = 100 * np.abs(plusDI - minusDI) / (plusDI + minusDI + 1e-12)
    adx_v = pd.Series(dx).rolling(adx_len).mean().to_numpy()

    typical = (high + low + close) / 3.0
    pv = pd.Series(typical * volume).rolling(vwap_len).sum().to_numpy()
    v = pd.Series(volume).rolling(vwap_len).sum().to_numpy()
    vwap_v = np.where(v > 0, pv / v, typical)

    atr_v = pd.Series(tr).rolling(atr_len).mean().to_numpy()

    is_buy = close > open_
    buy_vol = np.where(is_buy, volume, 0.0)
    sell_vol = np.where(~is_buy, volume, 0.0)
    total_buy = pd.Series(buy_vol).rolling(10).sum().to_numpy()
    total_sell = pd.Series(sell_vol).rolling(10).sum().to_numpy()
    total_vol = pd.Series(volume).rolling(10).sum().to_numpy()
    taker_v = np.where(total_vol > 0, (total_buy - total_sell) / total_vol, 0.0)

    # === Normalization ===
    nrsi = pd.Series(rsi_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    nadx = pd.Series(adx_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    nvwap = pd.Series(vwap_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    natr = pd.Series(atr_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    nvol = pd.Series(volume).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()
    ntaker = pd.Series(taker_v).rolling(norm_len).apply(lambda x: (x.iloc[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-12)).to_numpy()

    # === Features Matrix (use normalized for consistency) ===
    features = np.vstack([
        np.nan_to_num(nrsi), np.nan_to_num(nadx), np.nan_to_num(nvwap),
        np.nan_to_num(natr), np.nan_to_num(nvol), np.nan_to_num(ntaker)
    ]).T

    # === Regression Labels: Future relative return (float) ===
    labels = np.zeros(N, dtype=float)
    for g in range(N - prediction_horizon):
        fidx = g + prediction_horizon
        if fidx < N:
            labels[g] = (close[fidx] - close[g]) / close[g]  # e.g., 0.012 for +1.2%

    # === Weights ===
    time_diff = (timestamps - timestamps[0]).astype('timedelta64[s]').astype(float)
    weights = np.exp(-time_diff / (24 * 60 * 60))  # 1-day half-life

    # === Clean NaNs ===
    valid_mask = ~np.isnan(features).any(axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask]
    weights = weights[valid_mask]

    return features, labels, weights

def train_initial_model(cfg, model_path="catboost_model3.cbm"):  # Match CLI path
    df = fetch_ohlcv_df(init_exchange(cfg), cfg['symbol'], cfg['timeframe'], cfg['history_bars'] * 2)
    features, labels, weights = generate_features_and_labels(df, cfg)
    
    split_idx = int(len(features) * 0.8)
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    train_weights, val_weights = weights[:split_idx], weights[split_idx:]
    
    model = CatBoostRegressor(
        iterations=100000,
        learning_rate=0.03,
        depth=5,
        loss_function='RMSE',
        l2_leaf_reg=5,
        verbose=20,
        random_seed=42,
        early_stopping_rounds=100,
        task_type='CPU'
    )
    model.fit(
        train_features, train_labels, sample_weight=train_weights,
        eval_set=(val_features, val_labels),  # Fixed: no weights in eval_set
        use_best_model=True
    )
    print(f"Best iteration: {model.get_best_iteration()}")
    if model.get_evals_result():
        print(f"Val RMSE: {model.get_evals_result()['validation']['RMSE'][-1]:.4f}")
    model.save_model(model_path)
    return model



# ------------------------ CLI Entrypoint ------------------------
if __name__ == '__main__':
    cfg = load_config()
    cfg['mode'] = os.environ.get('PAPER_MODE', cfg.get('mode', 'SIMULATE'))
    cfg['apiKey'] = os.environ.get('EXCHANGE_APIKEY') or cfg.get('apiKey')
    cfg['secret'] = os.environ.get('EXCHANGE_SECRET') or cfg.get('secret')
    cfg['password'] = os.environ.get('EXCHANGE_PASSWORD') or cfg.get('password')

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'train':
        # Train mode: Auto fetch and train initial model
        train_initial_model(cfg)
        logging.info("Training process finished.")
    else:
        # Run mode: Auto train if no model, then start trader
        model_path = "catboost_model3.cbm"
        if not os.path.exists(model_path):
            train_initial_model(cfg)
        trader = SMCTrader(cfg)
        try:
            trader.run()
        except KeyboardInterrupt:
            logging.info('Keyboard interrupt received - exiting')
        finally:
            logging.info('Exited')


