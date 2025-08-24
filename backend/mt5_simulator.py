"""
MetaTrader 5 Simulator
Simulates MT5 API for development and testing purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import random
import time
import logging

# Constants mimicking MT5
TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 60
TIMEFRAME_H4 = 240
TIMEFRAME_D1 = 1440

ORDER_BUY = 0
ORDER_SELL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_RETURN = 0
TRADE_ACTION_DEAL = 1

COPY_TICKS_ALL = 0

class MT5Simulator:
    """Simulator for MetaTrader 5 API"""
    
    def __init__(self):
        self.initialized = False
        self.connected = False
        self.account_info_data = {
            'login': 1296306,
            'server': 'FreshForex-MT5',
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'margin_free': 10000.0,
            'profit': 0.0,
            'currency': 'USD',
            'leverage': 100
        }
        self.positions = []
        self.orders = []
        self.trade_history = []
        self.symbols_data = {}
        self.last_error = None
        
        # Load historical data for main forex pairs
        self.forex_symbols = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X'
        }
        
        # Initialize symbol data
        self._initialize_symbols_data()
        
        logging.info("MT5 Simulator initialized")
    
    def _initialize_symbols_data(self):
        """Initialize historical data for forex symbols"""
        for symbol in self.forex_symbols.keys():
            # Generate synthetic data for demonstration
            self._generate_synthetic_data(symbol)
            logging.info(f"Initialized data for {symbol}")
    
    def _generate_synthetic_data(self, symbol: str):
        """Generate synthetic forex data"""
        logging.info(f"Generating synthetic data for {symbol}")
        
        # Base prices for different symbols
        base_prices = {
            'EURUSD': 1.0800,
            'GBPUSD': 1.2700,
            'USDJPY': 150.00,
            'USDCHF': 0.9000,
            'AUDUSD': 0.6500,
            'USDCAD': 1.3500
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate 7 days of 5-minute data
        periods = 7 * 24 * 12  # 7 days * 24 hours * 12 (5-min intervals)
        start_time = datetime.now() - timedelta(days=7)
        
        # Generate realistic price movements
        prices = [base_price]
        volumes = []
        times = []
        
        for i in range(periods):
            current_time = start_time + timedelta(minutes=i*5)
            times.append(current_time)
            
            # Generate price movement
            change = np.random.normal(0, 0.0005)  # Small changes for forex
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
            # Generate volume
            volumes.append(np.random.randint(100, 1000))
        
        # Create OHLC data
        data = []
        for i in range(len(times)):
            price = prices[i]
            volatility = abs(np.random.normal(0, 0.0003))
            
            high = price + volatility
            low = price - volatility
            open_price = price + np.random.normal(0, 0.0001)
            close_price = price + np.random.normal(0, 0.0001)
            
            # Ensure high is highest and low is lowest
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            data.append({
                'time': times[i],
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'tick_volume': volumes[i]
            })
        
        df = pd.DataFrame(data)
        self.symbols_data[symbol] = df
        logging.info(f"Generated {len(df)} candles for {symbol}")
    
    def _initialize_symbols_data_original(self):
        """Initialize historical data for forex symbols - Original method with yfinance"""
        try:
            for symbol, yahoo_symbol in self.forex_symbols.items():
                # Get last 30 days of data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                # Download data from Yahoo Finance
                data = yf.download(yahoo_symbol, start=start_date, end=end_date, interval='5m')
                
                if not data.empty:
                    # Convert to MT5-like format
                    data['time'] = data.index
                    data['open'] = data['Open']
                    data['high'] = data['High']
                    data['low'] = data['Low']
                    data['close'] = data['Close']
                    data['tick_volume'] = data['Volume'] if 'Volume' in data else np.random.randint(100, 1000, len(data))
                    
                    self.symbols_data[symbol] = data
                    logging.info(f"Loaded {len(data)} candles for {symbol}")
                else:
                    # Generate synthetic data if Yahoo Finance fails
                    self._generate_synthetic_data(symbol)
                    
        except Exception as e:
            logging.error(f"Error loading symbol data: {e}")
            # Generate synthetic data for all symbols
            for symbol in self.forex_symbols.keys():
                self._generate_synthetic_data(symbol)
    
    def initialize(self) -> bool:
        """Initialize MT5 connection"""
        self.initialized = True
        self.last_error = None
        return True
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        self.initialized = False
        self.connected = False
        logging.info("MT5 Simulator shut down")
    
    def login(self, login: int, password: str, server: str) -> bool:
        """Login to MT5 account"""
        if not self.initialized:
            self.last_error = "MT5 not initialized"
            return False
        
        # Simple validation
        if login == 1296306 and password == "@0JxCmJa" and server == "FreshForex-MT5":
            self.connected = True
            self.account_info_data['login'] = login
            self.account_info_data['server'] = server
            self.last_error = None
            logging.info(f"Successfully logged in to account {login}")
            return True
        else:
            self.last_error = "Invalid credentials"
            return False
    
    def account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.connected:
            return None
        
        # Simple namedtuple-like object
        class AccountInfo:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
            
            def _asdict(self):
                return self.__dict__
        
        return AccountInfo(self.account_info_data)
    
    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> Optional[np.ndarray]:
        """Get historical rates"""
        if not self.connected or symbol not in self.symbols_data:
            return None
        
        data = self.symbols_data[symbol]
        
        # Handle different timeframes (for now, we'll use 5-minute data)
        if timeframe != TIMEFRAME_M5:
            # For simplicity, we'll resample to requested timeframe
            data = self._resample_data(data, timeframe)
        
        # Get the requested slice
        end_pos = start_pos + count
        if end_pos > len(data):
            end_pos = len(data)
        
        if start_pos >= len(data):
            return None
        
        slice_data = data.iloc[-(end_pos-start_pos):]
        
        # Convert to numpy array with MT5-like structure
        rates = []
        for i, row in slice_data.iterrows():
            # Handle timestamp conversion properly
            try:
                time_val = row['time']
                if pd.isna(time_val):
                    continue
                    
                if hasattr(time_val, 'timestamp'):
                    timestamp = int(time_val.timestamp())
                else:
                    # Convert to timestamp using pandas
                    timestamp = int(pd.Timestamp(time_val).timestamp())
                    
                rates.append((
                    timestamp,                     # time
                    float(row['open']),            # open
                    float(row['high']),            # high
                    float(row['low']),             # low
                    float(row['close']),           # close
                    int(row['tick_volume']),       # tick_volume
                    0,                             # spread
                    0                              # real_volume
                ))
            except Exception as e:
                # Skip problematic rows
                continue
        
        if not rates:
            return None
        
        return np.array(rates, dtype=[
            ('time', 'i8'),
            ('open', 'f8'),
            ('high', 'f8'),
            ('low', 'f8'),
            ('close', 'f8'),
            ('tick_volume', 'i8'),
            ('spread', 'i4'),
            ('real_volume', 'i8')
        ])
    
    def _resample_data(self, data: pd.DataFrame, timeframe: int) -> pd.DataFrame:
        """Resample data to different timeframe"""
        if timeframe == TIMEFRAME_M5:
            return data
        
        # Convert timeframe to pandas frequency
        freq_map = {
            TIMEFRAME_M1: '1T',
            TIMEFRAME_M15: '15T',
            TIMEFRAME_M30: '30T',
            TIMEFRAME_H1: '1H',
            TIMEFRAME_H4: '4H',
            TIMEFRAME_D1: '1D'
        }
        
        freq = freq_map.get(timeframe, '5T')
        
        # Set time as index for resampling
        data_copy = data.copy()
        data_copy.set_index('time', inplace=True)
        
        # Resample OHLC data
        resampled = data_copy.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tick_volume': 'sum'
        }).dropna()
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        return resampled
    
    def symbol_info_tick(self, symbol: str) -> Optional[Dict]:
        """Get current tick information"""
        if not self.connected or symbol not in self.symbols_data:
            return None
        
        data = self.symbols_data[symbol]
        if data.empty:
            return None
        
        # Get the latest price
        latest = data.iloc[-1]
        
        # Simulate bid/ask spread
        spread = 0.00002 if 'JPY' not in symbol else 0.002
        mid_price = latest['close']
        
        class TickInfo:
            def __init__(self):
                self.time = int(datetime.now().timestamp())
                self.bid = mid_price - spread/2
                self.ask = mid_price + spread/2
                self.last = mid_price
                self.volume = int(latest['tick_volume'])
        
        return TickInfo()
    
    def copy_ticks_from(self, symbol: str, from_time: datetime, count: int, flags: int) -> Optional[np.ndarray]:
        """Get tick data"""
        if not self.connected or symbol not in self.symbols_data:
            return None
        
        # For simplicity, generate synthetic tick data
        ticks = []
        current_time = from_time
        
        # Get current price
        tick_info = self.symbol_info_tick(symbol)
        if not tick_info:
            return None
        
        for i in range(count):
            # Small price variations
            variation = np.random.normal(0, 0.00001)
            bid = tick_info.bid + variation
            ask = tick_info.ask + variation
            
            ticks.append((
                int(current_time.timestamp()),
                bid,
                ask,
                (bid + ask) / 2,
                np.random.randint(1, 10),  # volume
                int(current_time.timestamp() * 1000) % 1000,  # time_msc
                0,  # flags
                0   # volume_real
            ))
            
            current_time += timedelta(seconds=1)
        
        return np.array(ticks, dtype=[
            ('time', 'i8'),
            ('bid', 'f8'),
            ('ask', 'f8'),
            ('last', 'f8'),
            ('volume', 'i8'),
            ('time_msc', 'i8'),
            ('flags', 'i4'),
            ('volume_real', 'f8')
        ])
    
    def order_send(self, request: Dict) -> Dict:
        """Send trading order"""
        if not self.connected:
            return {'retcode': 10013, 'comment': 'Not connected'}
        
        # Simulate order execution
        symbol = request.get('symbol', 'EURUSD')
        volume = request.get('volume', 0.1)
        order_type = request.get('type', ORDER_BUY)
        price = request.get('price', 0)
        
        # Get current price
        tick_info = self.symbol_info_tick(symbol)
        if not tick_info:
            return {'retcode': 10014, 'comment': 'Symbol not found'}
        
        # Use current price if not specified
        if price == 0:
            price = tick_info.ask if order_type == ORDER_BUY else tick_info.bid
        
        # Create order
        order_id = len(self.orders) + 1
        order = {
            'ticket': order_id,
            'symbol': symbol,
            'volume': volume,
            'type': order_type,
            'open_price': price,
            'open_time': datetime.now(),
            'sl': request.get('sl', 0),
            'tp': request.get('tp', 0),
            'profit': 0,
            'commission': -0.7,  # Simulate commission
            'swap': 0,
            'comment': request.get('comment', '')
        }
        
        self.orders.append(order)
        
        # Add to positions
        self.positions.append(order.copy())
        
        # Update account balance (subtract commission)
        self.account_info_data['balance'] -= 0.7
        
        logging.info(f"Order executed: {order_type} {volume} {symbol} at {price}")
        
        return {
            'retcode': 10009,  # TRADE_RETCODE_DONE
            'deal': order_id,
            'order': order_id,
            'volume': volume,
            'price': price,
            'comment': 'Order executed'
        }
    
    def positions_get(self, symbol: str = None) -> List[Dict]:
        """Get open positions"""
        if not self.connected:
            return []
        
        positions = self.positions.copy()
        
        if symbol:
            positions = [p for p in positions if p['symbol'] == symbol]
        
        # Update profits
        for position in positions:
            self._update_position_profit(position)
        
        return positions
    
    def _update_position_profit(self, position: Dict):
        """Update position profit"""
        symbol = position['symbol']
        tick_info = self.symbol_info_tick(symbol)
        
        if not tick_info:
            return
        
        # Calculate profit
        if position['type'] == ORDER_BUY:
            current_price = tick_info.bid
            profit = (current_price - position['open_price']) * position['volume'] * 100000
        else:  # SELL
            current_price = tick_info.ask
            profit = (position['open_price'] - current_price) * position['volume'] * 100000
        
        position['profit'] = profit
        
        # Update account equity
        total_profit = sum(p['profit'] for p in self.positions)
        self.account_info_data['equity'] = self.account_info_data['balance'] + total_profit
        self.account_info_data['profit'] = total_profit
    
    def history_deals_get(self, from_date: datetime, to_date: datetime) -> List[Dict]:
        """Get trade history"""
        if not self.connected:
            return []
        
        # Filter history by date range
        history = [
            deal for deal in self.trade_history
            if from_date <= deal['time'] <= to_date
        ]
        
        return history
    
    def last_error(self) -> Optional[str]:
        """Get last error"""
        return self.last_error


# Global instance
mt5_sim = MT5Simulator()

# Export functions to match MT5 API
def initialize():
    return mt5_sim.initialize()

def shutdown():
    return mt5_sim.shutdown()

def login(login, password, server):
    return mt5_sim.login(login, password, server)

def account_info():
    return mt5_sim.account_info()

def copy_rates_from_pos(symbol, timeframe, start_pos, count):
    return mt5_sim.copy_rates_from_pos(symbol, timeframe, start_pos, count)

def symbol_info_tick(symbol):
    return mt5_sim.symbol_info_tick(symbol)

def copy_ticks_from(symbol, from_time, count, flags):
    return mt5_sim.copy_ticks_from(symbol, from_time, count, flags)

def order_send(request):
    return mt5_sim.order_send(request)

def positions_get(symbol=None):
    return mt5_sim.positions_get(symbol)

def history_deals_get(from_date, to_date):
    return mt5_sim.history_deals_get(from_date, to_date)

def last_error():
    return mt5_sim.last_error