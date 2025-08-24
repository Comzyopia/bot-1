"""
Indicateurs techniques avancés pour le Bot de Trading
Version: 2.0 - Professional Edition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import talib
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Classe pour calculer tous les indicateurs techniques"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcule tous les indicateurs techniques"""
        if df.empty or len(df) < 50:
            return {}
        
        indicators = {}
        
        # Indicateurs de tendance
        indicators.update(self._calculate_trend_indicators(df))
        
        # Indicateurs de momentum
        indicators.update(self._calculate_momentum_indicators(df))
        
        # Indicateurs de volume
        indicators.update(self._calculate_volume_indicators(df))
        
        # Indicateurs de volatilité
        indicators.update(self._calculate_volatility_indicators(df))
        
        # Support et résistance
        indicators.update(self._calculate_support_resistance(df))
        
        # Patterns de chandeliers
        indicators.update(self._calculate_candlestick_patterns(df))
        
        return indicators
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcule les indicateurs de tendance"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        try:
            # Moyennes mobiles
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['sma_200'] = talib.SMA(close, timeperiod=200)
            
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)
            indicators['ema_50'] = talib.EMA(close, timeperiod=50)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
            
            # ADX
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
            indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle * 100
            
            # Parabolic SAR
            indicators['sar'] = talib.SAR(high, low)
            
            # Ichimoku
            indicators.update(self._calculate_ichimoku(df))
            
        except Exception as e:
            print(f"Erreur dans le calcul des indicateurs de tendance: {e}")
        
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcule les indicateurs de momentum"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        try:
            # RSI
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd
            
            # CCI
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=20)
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # ROC
            indicators['roc'] = talib.ROC(close, timeperiod=10)
            
            # CMO
            indicators['cmo'] = talib.CMO(close, timeperiod=14)
            
            # MFI
            if 'tick_volume' in df.columns:
                indicators['mfi'] = talib.MFI(high, low, close, df['tick_volume'].values, timeperiod=14)
            
        except Exception as e:
            print(f"Erreur dans le calcul des indicateurs de momentum: {e}")
        
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcule les indicateurs de volume"""
        close = df['close'].values
        volume = df.get('tick_volume', df.get('volume', np.ones(len(df)))).values
        
        indicators = {}
        
        try:
            # Volume SMA
            indicators['volume_sma'] = talib.SMA(volume, timeperiod=20)
            
            # On Balance Volume
            indicators['obv'] = talib.OBV(close, volume)
            
            # Accumulation/Distribution Line
            indicators['ad'] = talib.AD(df['high'].values, df['low'].values, close, volume)
            
            # Chaikin A/D Oscillator
            indicators['chaikin_ad'] = talib.ADOSC(df['high'].values, df['low'].values, close, volume)
            
            # Volume Price Trend
            vpt = np.zeros(len(close))
            for i in range(1, len(close)):
                vpt[i] = vpt[i-1] + volume[i] * (close[i] - close[i-1]) / close[i-1]
            indicators['vpt'] = vpt
            
        except Exception as e:
            print(f"Erreur dans le calcul des indicateurs de volume: {e}")
        
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcule les indicateurs de volatilité"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        try:
            # ATR
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Volatilité historique
            returns = np.diff(np.log(close))
            indicators['historical_volatility'] = np.sqrt(252) * np.std(returns) * 100
            
            # True Range
            indicators['trange'] = talib.TRANGE(high, low, close)
            
            # Normalized ATR
            indicators['natr'] = talib.NATR(high, low, close, timeperiod=14)
            
        except Exception as e:
            print(f"Erreur dans le calcul des indicateurs de volatilité: {e}")
        
        return indicators
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calcule les niveaux de support et résistance"""
        indicators = {}
        
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Points pivots
            pivot_points = self._calculate_pivot_points(df)
            indicators.update(pivot_points)
            
            # Support et résistance par extrema
            sr_levels = self._find_support_resistance_levels(high, low, close)
            indicators.update(sr_levels)
            
            # Fibonacci retracements
            fib_levels = self._calculate_fibonacci_levels(high, low)
            indicators.update(fib_levels)
            
        except Exception as e:
            print(f"Erreur dans le calcul des support/résistance: {e}")
        
        return indicators
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calcule les points pivots"""
        if len(df) < 2:
            return {}
        
        # Utilise les données du jour précédent
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        
        # Point pivot principal
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Résistances
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        
        # Supports
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    def _find_support_resistance_levels(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """Trouve les niveaux de support et résistance par extrema"""
        # Trouve les extrema locaux
        high_peaks = argrelextrema(high, np.greater, order=5)[0]
        low_valleys = argrelextrema(low, np.less, order=5)[0]
        
        # Récupère les niveaux
        resistance_levels = high[high_peaks]
        support_levels = low[low_valleys]
        
        # Trie et filtre
        resistance_levels = sorted(resistance_levels, reverse=True)[:5]
        support_levels = sorted(support_levels)[:5]
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels
        }
    
    def _calculate_fibonacci_levels(self, high: np.ndarray, low: np.ndarray) -> Dict:
        """Calcule les niveaux de Fibonacci"""
        # Trouve le plus haut et plus bas récents
        recent_high = np.max(high[-100:])
        recent_low = np.min(low[-100:])
        
        diff = recent_high - recent_low
        
        # Niveaux de Fibonacci
        fib_levels = {
            'fib_0': recent_high,
            'fib_23.6': recent_high - 0.236 * diff,
            'fib_38.2': recent_high - 0.382 * diff,
            'fib_50': recent_high - 0.5 * diff,
            'fib_61.8': recent_high - 0.618 * diff,
            'fib_100': recent_low
        }
        
        return fib_levels
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calcule l'indicateur Ichimoku"""
        if len(df) < 52:
            return {}
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        span_a = ((tenkan + kijun) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou = close.shift(-26)
        
        return {
            'ichimoku_tenkan': tenkan.values,
            'ichimoku_kijun': kijun.values,
            'ichimoku_span_a': span_a.values,
            'ichimoku_span_b': span_b.values,
            'ichimoku_chikou': chikou.values
        }
    
    def _calculate_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Calcule les patterns de chandeliers"""
        if len(df) < 10:
            return {}
        
        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        patterns = {}
        
        try:
            # Patterns de retournement
            patterns['doji'] = talib.CDLDOJI(open_price, high, low, close)
            patterns['hammer'] = talib.CDLHAMMER(open_price, high, low, close)
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
            patterns['engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
            patterns['harami'] = talib.CDLHARAMI(open_price, high, low, close)
            patterns['morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
            patterns['evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
            
            # Patterns de continuation
            patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)
            patterns['three_black_crows'] = talib.CDL3BLACKCROWS(open_price, high, low, close)
            patterns['rising_three'] = talib.CDLRISEFALL3METHODS(open_price, high, low, close)
            
            # Patterns additionnels
            patterns['spinning_top'] = talib.CDLSPINNINGTOP(open_price, high, low, close)
            patterns['marubozu'] = talib.CDLMARUBOZU(open_price, high, low, close)
            patterns['dragonfly_doji'] = talib.CDLDRAGONFLYDOJI(open_price, high, low, close)
            patterns['gravestone_doji'] = talib.CDLGRAVESTONEDOJI(open_price, high, low, close)
            
        except Exception as e:
            print(f"Erreur dans le calcul des patterns de chandeliers: {e}")
        
        return patterns
    
    def get_signal_strength(self, indicators: Dict, current_price: float) -> Dict:
        """Calcule la force du signal basée sur les indicateurs"""
        if not indicators:
            return {"strength": 0, "direction": "NEUTRAL", "details": []}
        
        signals = []
        total_weight = 0
        total_score = 0
        
        try:
            # Analyse des moyennes mobiles
            sma_20 = indicators.get('sma_20')
            sma_50 = indicators.get('sma_50')
            ema_12 = indicators.get('ema_12')
            ema_26 = indicators.get('ema_26')
            
            if sma_20 is not None and sma_50 is not None and not np.isnan(sma_20[-1]) and not np.isnan(sma_50[-1]):
                if sma_20[-1] > sma_50[-1]:
                    signals.append({"name": "SMA Crossover", "score": 1, "weight": 0.3})
                    total_score += 1 * 0.3
                else:
                    signals.append({"name": "SMA Crossover", "score": -1, "weight": 0.3})
                    total_score -= 1 * 0.3
                total_weight += 0.3
            
            # Analyse RSI
            rsi = indicators.get('rsi')
            if rsi is not None and not np.isnan(rsi[-1]):
                if rsi[-1] < 30:
                    signals.append({"name": "RSI Oversold", "score": 1, "weight": 0.4})
                    total_score += 1 * 0.4
                elif rsi[-1] > 70:
                    signals.append({"name": "RSI Overbought", "score": -1, "weight": 0.4})
                    total_score -= 1 * 0.4
                total_weight += 0.4
            
            # Analyse MACD
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            if macd is not None and macd_signal is not None and not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]):
                if macd[-1] > macd_signal[-1]:
                    signals.append({"name": "MACD Bullish", "score": 1, "weight": 0.35})
                    total_score += 1 * 0.35
                else:
                    signals.append({"name": "MACD Bearish", "score": -1, "weight": 0.35})
                    total_score -= 1 * 0.35
                total_weight += 0.35
            
            # Analyse des Bollinger Bands
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            if bb_upper is not None and bb_lower is not None and not np.isnan(bb_upper[-1]) and not np.isnan(bb_lower[-1]):
                if current_price <= bb_lower[-1]:
                    signals.append({"name": "BB Oversold", "score": 1, "weight": 0.25})
                    total_score += 1 * 0.25
                elif current_price >= bb_upper[-1]:
                    signals.append({"name": "BB Overbought", "score": -1, "weight": 0.25})
                    total_score -= 1 * 0.25
                total_weight += 0.25
            
            # Calcul de la force finale
            if total_weight > 0:
                strength = total_score / total_weight
                strength = max(-1, min(1, strength))  # Normalise entre -1 et 1
            else:
                strength = 0
            
            # Détermination de la direction
            if strength > 0.3:
                direction = "BULLISH"
            elif strength < -0.3:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"
            
            return {
                "strength": abs(strength),
                "direction": direction,
                "details": signals,
                "score": strength
            }
            
        except Exception as e:
            print(f"Erreur dans le calcul de la force du signal: {e}")
            return {"strength": 0, "direction": "NEUTRAL", "details": []}
    
    def detect_divergences(self, df: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """Détecte les divergences entre prix et indicateurs"""
        divergences = []
        
        if len(df) < 50:
            return divergences
        
        try:
            price = df['close'].values
            rsi = indicators.get('rsi')
            macd = indicators.get('macd')
            
            # Divergence RSI
            if rsi is not None:
                price_peaks = argrelextrema(price, np.greater, order=10)[0]
                rsi_peaks = argrelextrema(rsi, np.greater, order=10)[0]
                
                if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                    # Divergence baissière
                    if price[-1] > price[price_peaks[-2]] and rsi[-1] < rsi[rsi_peaks[-2]]:
                        divergences.append({
                            "type": "bearish_divergence",
                            "indicator": "RSI",
                            "strength": 0.7
                        })
                    
                    # Divergence haussière
                    if price[-1] < price[price_peaks[-2]] and rsi[-1] > rsi[rsi_peaks[-2]]:
                        divergences.append({
                            "type": "bullish_divergence",
                            "indicator": "RSI",
                            "strength": 0.7
                        })
            
        except Exception as e:
            print(f"Erreur dans la détection des divergences: {e}")
        
        return divergences