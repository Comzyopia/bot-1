"""
🧠 ULTRA PRICE ACTION AI SYSTEM
═══════════════════════════════════

Système d'IA révolutionnaire spécialisé dans l'analyse du Price Action
pour des décisions de trading ultra-précises et performantes.

Version: 3.0 - Ultra Performance Edition
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import cv2
from scipy.signal import find_peaks, argrelextrema
from ultra_reward_system import ultra_reward_system, TradeContext, MarketRegime

class PriceActionPattern(Enum):
    """Patterns de Price Action détectés"""
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"
    BREAKOUT_BULLISH = "breakout_bullish"
    BREAKOUT_BEARISH = "breakout_bearish"
    FALSE_BREAKOUT = "false_breakout"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT = "pennant"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"

@dataclass
class PriceActionSignal:
    """Signal de Price Action avec contexte complet"""
    pattern: PriceActionPattern
    confidence: float
    strength: float
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    market_regime: MarketRegime
    timeframe_confluence: Dict[str, bool]
    volume_confirmation: bool
    momentum_alignment: bool
    support_resistance_level: float
    pattern_completion_bar: int
    expected_move_pips: float

class UltraPriceActionAI:
    """🚀 IA Ultra-Performante spécialisée dans le Price Action"""
    
    def __init__(self, state_size: int = 100, action_size: int = 5):
        self.state_size = state_size
        self.action_size = action_size  # 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE_ALL, 4=SCALE_IN
        
        # 🧠 ARCHITECTURE NEURONAL ULTRA-AVANCÉE
        self.price_action_model = self._build_price_action_model()
        self.pattern_recognition_model = self._build_pattern_recognition_model()
        self.regime_detection_model = self._build_regime_detection_model()
        
        # 📊 MÉMOIRE ET EXPÉRIENCE
        self.memory = deque(maxlen=100000)  # Plus grande mémoire
        self.price_action_memory = deque(maxlen=50000)
        self.pattern_memory = deque(maxlen=25000)
        
        # 🎯 PARAMÈTRES D'APPRENTISSAGE OPTIMISÉS
        self.gamma = 0.99  # Facteur de discount élevé
        self.epsilon = 0.8  # Exploration initiale
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9999  # Décroissance très lente
        self.learning_rate = 0.0003
        
        # 📈 ANALYSE DE MARCHÉ
        self.support_resistance_levels = {}
        self.current_market_regime = MarketRegime.RANGING
        self.volatility_regime = "normal"
        self.session_type = "london"  # london, ny, asian
        
        # 🏆 PERFORMANCE TRACKING
        self.performance_metrics = {
            "pattern_accuracy": 0.0,
            "support_resistance_accuracy": 0.0,
            "breakout_success_rate": 0.0,
            "false_signal_rate": 0.0,
            "price_action_score": 0.0
        }
        
        # 🎛️ CONFIGURATION AVANCÉE
        self.config = {
            "min_pattern_confidence": 0.75,
            "min_breakout_volume": 1.5,
            "max_risk_per_trade": 0.015,
            "preferred_rr_ratio": 2.5,
            "max_correlation_trades": 2,
            "session_filters": True,
            "news_filter": True,
            "volatility_filter": True
        }
        
        logging.info("🚀 Ultra Price Action AI initialisé avec succès!")
    
    def _build_price_action_model(self):
        """🏗️ Construit le modèle principal de Price Action"""
        model = keras.Sequential([
            # Couche d'entrée avec normalisation
            layers.Input(shape=(self.state_size,)),
            layers.BatchNormalization(),
            
            # Couches convolutives pour patterns temporels
            layers.Reshape((self.state_size, 1)),
            layers.Conv1D(128, 7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(256, 5, activation='relu', padding='same'),  
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.GlobalMaxPooling1D(),
            
            # Couches denses avec attention
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(), 
            layers.Dropout(0.2),
            
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            
            # Sortie avec différentes têtes
            layers.Dense(self.action_size, activation='linear', name='q_values')
        ])
        
        # Optimiseur avancé
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Plus robuste que MSE
            metrics=['mae']
        )
        
        return model
    
    def _build_pattern_recognition_model(self):
        """🔍 Modèle spécialisé dans la reconnaissance de patterns"""
        inputs = keras.Input(shape=(50, 4))  # 50 barres OHLC
        
        # Branch pour patterns courts
        short_conv = layers.Conv1D(64, 3, activation='relu')(inputs)
        short_conv = layers.Conv1D(32, 3, activation='relu')(short_conv)
        short_pool = layers.GlobalMaxPooling1D()(short_conv)
        
        # Branch pour patterns longs  
        long_conv = layers.Conv1D(32, 7, activation='relu')(inputs)
        long_conv = layers.Conv1D(16, 7, activation='relu')(long_conv)
        long_pool = layers.GlobalMaxPooling1D()(long_conv)
        
        # Combinaison
        combined = layers.concatenate([short_pool, long_pool])
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        
        # Sortie multi-label pour différents patterns
        outputs = layers.Dense(len(PriceActionPattern), activation='sigmoid')(combined)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_regime_detection_model(self):
        """🌍 Modèle de détection de régime de marché"""
        model = keras.Sequential([
            layers.Input(shape=(20,)),  # Features de régime
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(MarketRegime), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def analyze_price_action(self, market_data: Dict, symbol: str) -> PriceActionSignal:
        """📊 Analyse complète du Price Action"""
        
        # 1. Préparation des données
        ohlc_data = self._prepare_ohlc_data(market_data)
        if ohlc_data is None:
            return None
        
        # 2. Détection du régime de marché
        market_regime = self._detect_market_regime(ohlc_data)
        
        # 3. Identification des niveaux Support/Résistance
        sr_levels = self._identify_support_resistance(ohlc_data)
        
        # 4. Reconnaissance de patterns
        patterns = self._recognize_patterns(ohlc_data)
        
        # 5. Analyse de la confluence multi-timeframe
        confluence = self._analyze_timeframe_confluence(symbol, ohlc_data)
        
        # 6. Validation par le volume
        volume_confirmation = self._validate_with_volume(ohlc_data, patterns)
        
        # 7. Génération du signal final
        signal = self._generate_price_action_signal(
            patterns, sr_levels, market_regime, confluence, volume_confirmation, ohlc_data
        )
        
        return signal
    
    def _prepare_ohlc_data(self, market_data: Dict) -> Optional[pd.DataFrame]:
        """📈 Prépare les données OHLC pour l'analyse"""
        try:
            # Simule des données OHLC (dans la vraie implémentation, utilise les vraies données)
            df = pd.DataFrame({
                'open': market_data.get('open_prices', []),
                'high': market_data.get('high_prices', []),  
                'low': market_data.get('low_prices', []),
                'close': market_data.get('close_prices', []),
                'volume': market_data.get('volumes', [])
            })
            
            if len(df) < 50:
                return None
                
            return df
            
        except Exception as e:
            logging.error(f"Erreur préparation données OHLC: {e}")
            return None
    
    def _detect_market_regime(self, ohlc_data: pd.DataFrame) -> MarketRegime:
        """🌍 Détecte le régime de marché actuel"""
        try:
            close = ohlc_data['close'].values
            high = ohlc_data['high'].values
            low = ohlc_data['low'].values
            
            # Calcul de volatilité
            returns = np.diff(np.log(close))
            volatility = np.std(returns) * np.sqrt(252)
            
            # Calcul de tendance
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
            
            # Calcul de range
            atr = np.mean(high[-14:] - low[-14:])
            current_range = (max(high[-20:]) - min(low[-20:])) / close[-1]
            
            # Classification du régime
            if volatility > 0.25:
                regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.10:
                regime = MarketRegime.LOW_VOLATILITY
            elif sma_20 > sma_50 * 1.005:  # Tendance haussière forte
                regime = MarketRegime.TRENDING_BULL
            elif sma_20 < sma_50 * 0.995:  # Tendance baissière forte
                regime = MarketRegime.TRENDING_BEAR
            elif current_range < 0.02:  # Faible range
                regime = MarketRegime.RANGING
            else:
                # Détection de breakout potentiel
                recent_high = max(high[-5:])
                recent_low = min(low[-5:])
                prev_high = max(high[-20:-5])
                prev_low = min(low[-20:-5])
                
                if recent_high > prev_high * 1.002:
                    regime = MarketRegime.BREAKOUT
                else:
                    regime = MarketRegime.RANGING
            
            self.current_market_regime = regime
            return regime
            
        except Exception as e:
            logging.error(f"Erreur détection régime: {e}")
            return MarketRegime.RANGING
    
    def _identify_support_resistance(self, ohlc_data: pd.DataFrame) -> Dict[str, List[float]]:
        """🎯 Identifie les niveaux de support et résistance par Price Action pur"""
        try:
            high = ohlc_data['high'].values
            low = ohlc_data['low'].values
            close = ohlc_data['close'].values
            
            # Détection des pivots hauts (résistances)
            resistance_indices = find_peaks(high, distance=5, prominence=np.std(high)*0.5)[0]
            resistance_levels = high[resistance_indices]
            
            # Détection des pivots bas (supports) 
            support_indices = find_peaks(-low, distance=5, prominence=np.std(low)*0.5)[0]
            support_levels = low[support_indices]
            
            # Filtrage par force (nombre de touches)
            strong_resistances = []
            strong_supports = []
            
            current_price = close[-1]
            
            # Analyse de la force des résistances
            for level in resistance_levels:
                touches = sum(1 for h in high if abs(h - level) / level < 0.001)
                if touches >= 2 and level > current_price:  # Au moins 2 touches, au-dessus du prix
                    strong_resistances.append(level)
            
            # Analyse de la force des supports
            for level in support_levels:
                touches = sum(1 for l in low if abs(l - level) / level < 0.001)
                if touches >= 2 and level < current_price:  # Au moins 2 touches, en-dessous du prix
                    strong_supports.append(level)
            
            # Tri par proximité du prix actuel
            strong_resistances = sorted(strong_resistances)[:5]  # Top 5 résistances
            strong_supports = sorted(strong_supports, reverse=True)[:5]  # Top 5 supports
            
            levels = {
                'resistances': strong_resistances,
                'supports': strong_supports,
                'current_price': current_price
            }
            
            # Mise à jour de la cache
            self.support_resistance_levels = levels
            
            return levels
            
        except Exception as e:
            logging.error(f"Erreur identification S/R: {e}")
            return {'resistances': [], 'supports': [], 'current_price': 0}
    
    def _recognize_patterns(self, ohlc_data: pd.DataFrame) -> List[Dict]:
        """🔍 Reconnaissance de patterns Price Action"""
        patterns = []
        
        try:
            # Pattern Double Top/Bottom
            double_patterns = self._detect_double_patterns(ohlc_data)
            patterns.extend(double_patterns)
            
            # Pattern Head & Shoulders
            hs_patterns = self._detect_head_shoulders(ohlc_data)
            patterns.extend(hs_patterns)
            
            # Pattern Triangles  
            triangle_patterns = self._detect_triangles(ohlc_data)
            patterns.extend(triangle_patterns)
            
            # Pattern Flags & Pennants
            flag_patterns = self._detect_flags_pennants(ohlc_data)
            patterns.extend(flag_patterns)
            
            # Pattern Breakouts
            breakout_patterns = self._detect_breakouts(ohlc_data)
            patterns.extend(breakout_patterns)
            
            return patterns
            
        except Exception as e:
            logging.error(f"Erreur reconnaissance patterns: {e}")
            return []
    
    def _detect_double_patterns(self, ohlc_data: pd.DataFrame) -> List[Dict]:
        """🔄 Détecte les patterns Double Top/Bottom"""
        patterns = []
        
        try:
            high = ohlc_data['high'].values
            low = ohlc_data['low'].values
            
            # Double Top
            peaks = find_peaks(high, distance=10, prominence=np.std(high)*0.8)[0]
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                peak1_level = high[last_two_peaks[0]]
                peak2_level = high[last_two_peaks[1]]
                
                # Vérifier si les niveaux sont similaires (tolérance 0.2%)
                if abs(peak1_level - peak2_level) / peak1_level < 0.002:
                    valley_between = min(low[last_two_peaks[0]:last_two_peaks[1]])
                    
                    if (peak1_level - valley_between) / peak1_level > 0.005:  # Decline significatif
                        patterns.append({
                            'pattern': PriceActionPattern.DOUBLE_TOP,
                            'confidence': 0.8,
                            'resistance_level': (peak1_level + peak2_level) / 2,
                            'target': valley_between - (peak1_level - valley_between) * 0.618,  # Extension Fibonacci
                            'direction': 'bearish'
                        })
            
            # Double Bottom
            valleys = find_peaks(-low, distance=10, prominence=np.std(low)*0.8)[0]
            if len(valleys) >= 2:
                last_two_valleys = valleys[-2:]
                valley1_level = low[last_two_valleys[0]]
                valley2_level = low[last_two_valleys[1]]
                
                if abs(valley1_level - valley2_level) / valley1_level < 0.002:
                    peak_between = max(high[last_two_valleys[0]:last_two_valleys[1]])
                    
                    if (peak_between - valley1_level) / valley1_level > 0.005:
                        patterns.append({
                            'pattern': PriceActionPattern.DOUBLE_BOTTOM,
                            'confidence': 0.8,
                            'support_level': (valley1_level + valley2_level) / 2,
                            'target': peak_between + (peak_between - valley1_level) * 0.618,
                            'direction': 'bullish'
                        })
            
        except Exception as e:
            logging.error(f"Erreur détection double patterns: {e}")
        
        return patterns
    
    def _detect_breakouts(self, ohlc_data: pd.DataFrame) -> List[Dict]:
        """💥 Détecte les breakouts avec confirmation de volume"""
        patterns = []
        
        try:
            high = ohlc_data['high'].values
            low = ohlc_data['low'].values
            close = ohlc_data['close'].values
            volume = ohlc_data.get('volume', np.ones(len(close))).values
            
            # Calcul des niveaux de consolidation récents
            lookback = min(20, len(close) - 1)
            recent_high = max(high[-lookback:-1])  # Exclut la barre actuelle
            recent_low = min(low[-lookback:-1])
            avg_volume = np.mean(volume[-lookback:-1])
            
            current_high = high[-1]
            current_low = low[-1]
            current_volume = volume[-1]
            current_close = close[-1]
            
            # Breakout haussier
            if current_high > recent_high * 1.001:  # 0.1% au-dessus
                volume_confirmation = current_volume > avg_volume * 1.3
                
                # Mesure la force du breakout
                breakout_strength = (current_high - recent_high) / recent_high
                
                patterns.append({
                    'pattern': PriceActionPattern.BREAKOUT_BULLISH,
                    'confidence': 0.9 if volume_confirmation else 0.6,
                    'breakout_level': recent_high,
                    'strength': breakout_strength,
                    'volume_confirmed': volume_confirmation,
                    'target': recent_high + (recent_high - recent_low) * 1.0,  # Hauteur de la consolidation
                    'direction': 'bullish'
                })
            
            # Breakout baissier
            elif current_low < recent_low * 0.999:  # 0.1% en-dessous
                volume_confirmation = current_volume > avg_volume * 1.3
                
                breakout_strength = (recent_low - current_low) / recent_low
                
                patterns.append({
                    'pattern': PriceActionPattern.BREAKOUT_BEARISH,
                    'confidence': 0.9 if volume_confirmation else 0.6,
                    'breakout_level': recent_low,
                    'strength': breakout_strength,
                    'volume_confirmed': volume_confirmation,
                    'target': recent_low - (recent_high - recent_low) * 1.0,
                    'direction': 'bearish'
                })
            
            # Détection de faux breakout
            elif len(close) > 3:
                # Vérifier si breakout précédent a échoué
                prev_high = max(high[-4:-1])
                prev_close = close[-2]
                
                if prev_high > recent_high and current_close < recent_high * 0.998:
                    patterns.append({
                        'pattern': PriceActionPattern.FALSE_BREAKOUT,
                        'confidence': 0.7,
                        'failed_level': recent_high,
                        'direction': 'bearish'  # Retour dans la range = signal baissier
                    })
            
        except Exception as e:
            logging.error(f"Erreur détection breakouts: {e}")
        
        return patterns
    
    def _detect_triangles(self, ohlc_data: pd.DataFrame) -> List[Dict]:
        """📐 Détecte les patterns triangulaires"""
        patterns = []
        
        if len(ohlc_data) < 30:
            return patterns
        
        try:
            high = ohlc_data['high'].values
            low = ohlc_data['low'].values
            
            # Trouve les pics et vallées significatifs
            peaks = find_peaks(high, distance=5, prominence=np.std(high)*0.3)[0]
            valleys = find_peaks(-low, distance=5, prominence=np.std(low)*0.3)[0]
            
            if len(peaks) >= 3 and len(valleys) >= 3:
                # Analyse des 3 derniers pics et vallées
                recent_peaks = peaks[-3:]
                recent_valleys = valleys[-3:]
                
                # Calcul des pentes
                peak_slope = (high[recent_peaks[-1]] - high[recent_peaks[0]]) / (recent_peaks[-1] - recent_peaks[0])
                valley_slope = (low[recent_valleys[-1]] - low[recent_valleys[0]]) / (recent_valleys[-1] - recent_valleys[0])
                
                # Triangle ascendant (résistance horizontale, support montant)
                if abs(peak_slope) < 0.0001 and valley_slope > 0.0001:
                    patterns.append({
                        'pattern': PriceActionPattern.TRIANGLE_ASCENDING,
                        'confidence': 0.75,
                        'resistance_level': np.mean(high[recent_peaks]),
                        'direction': 'bullish',
                        'apex_estimate': len(ohlc_data) + 10  # Estimation du sommet
                    })
                
                # Triangle descendant (support horizontal, résistance descendante)
                elif abs(valley_slope) < 0.0001 and peak_slope < -0.0001:
                    patterns.append({
                        'pattern': PriceActionPattern.TRIANGLE_DESCENDING,
                        'confidence': 0.75,
                        'support_level': np.mean(low[recent_valleys]),
                        'direction': 'bearish',
                        'apex_estimate': len(ohlc_data) + 10
                    })
                
                # Triangle symétrique (les deux pentes convergent)
                elif peak_slope < -0.0001 and valley_slope > 0.0001:
                    patterns.append({
                        'pattern': PriceActionPattern.PENNANT,
                        'confidence': 0.65,
                        'upper_line': high[recent_peaks[-1]],
                        'lower_line': low[recent_valleys[-1]],
                        'direction': 'neutral'  # Direction selon breakout
                    })
        
        except Exception as e:
            logging.error(f"Erreur détection triangles: {e}")
        
        return patterns
    
    def _detect_head_shoulders(self, ohlc_data: pd.DataFrame) -> List[Dict]:
        """👤 Détecte les patterns Head & Shoulders"""
        patterns = []
        
        try:
            high = ohlc_data['high'].values
            low = ohlc_data['low'].values
            
            # Trouve les pics significatifs pour H&S
            peaks = find_peaks(high, distance=8, prominence=np.std(high)*0.5)[0]
            
            if len(peaks) >= 3:
                # Analyse des 3 derniers pics pour H&S classique
                last_three = peaks[-3:]
                left_shoulder = high[last_three[0]]
                head = high[last_three[1]]
                right_shoulder = high[last_three[2]]
                
                # Conditions pour H&S valide:
                # 1. La tête est plus haute que les épaules
                # 2. Les épaules sont approximativement au même niveau
                if (head > left_shoulder and head > right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.03):
                    
                    # Trouve la neckline (ligne de cou)
                    valley1_idx = last_three[0] + np.argmin(low[last_three[0]:last_three[1]])
                    valley2_idx = last_three[1] + np.argmin(low[last_three[1]:last_three[2]])
                    neckline = (low[valley1_idx] + low[valley2_idx]) / 2
                    
                    # Target = neckline - (head - neckline)
                    target = neckline - (head - neckline)
                    
                    patterns.append({
                        'pattern': PriceActionPattern.HEAD_SHOULDERS,
                        'confidence': 0.85,
                        'neckline': neckline,
                        'head_level': head,
                        'left_shoulder': left_shoulder,
                        'right_shoulder': right_shoulder,
                        'target': target,
                        'direction': 'bearish'
                    })
            
            # H&S inversé (sur les lows)
            valleys = find_peaks(-low, distance=8, prominence=np.std(low)*0.5)[0]
            
            if len(valleys) >= 3:
                last_three = valleys[-3:]
                left_shoulder = low[last_three[0]]
                head = low[last_three[1]]
                right_shoulder = low[last_three[2]]
                
                if (head < left_shoulder and head < right_shoulder and
                    abs(left_shoulder - right_shoulder) / left_shoulder < 0.03):
                    
                    peak1_idx = last_three[0] + np.argmax(high[last_three[0]:last_three[1]])
                    peak2_idx = last_three[1] + np.argmax(high[last_three[1]:last_three[2]])
                    neckline = (high[peak1_idx] + high[peak2_idx]) / 2
                    
                    target = neckline + (neckline - head)
                    
                    patterns.append({
                        'pattern': PriceActionPattern.HEAD_SHOULDERS,
                        'confidence': 0.85,
                        'neckline': neckline,
                        'head_level': head,
                        'left_shoulder': left_shoulder,
                        'right_shoulder': right_shoulder,
                        'target': target,
                        'direction': 'bullish',
                        'inverted': True
                    })
        
        except Exception as e:
            logging.error(f"Erreur détection H&S: {e}")
        
        return patterns
    
    def _detect_flags_pennants(self, ohlc_data: pd.DataFrame) -> List[Dict]:
        """🏁 Détecte les patterns Flags et Pennants"""
        patterns = []
        
        try:
            high = ohlc_data['high'].values
            low = ohlc_data['low'].values
            close = ohlc_data['close'].values
            
            # Détecte un mouvement fort récent (flagpole)
            if len(close) >= 20:
                # Cherche un mouvement de plus de 1% en moins de 10 barres
                for i in range(10, len(close)-5):
                    move_start = close[i-10]
                    move_end = close[i]
                    move_pct = abs(move_end - move_start) / move_start
                    
                    if move_pct > 0.01:  # Mouvement > 1%
                        # Analyse la consolidation qui suit (flag)
                        consolidation = close[i:i+5] if i+5 < len(close) else close[i:]
                        
                        if len(consolidation) >= 3:
                            consolidation_range = (max(consolidation) - min(consolidation)) / np.mean(consolidation)
                            
                            # Flag = consolidation serrée après mouvement fort
                            if consolidation_range < 0.005:  # Range < 0.5%
                                direction = 'bullish' if move_end > move_start else 'bearish'
                                
                                patterns.append({
                                    'pattern': PriceActionPattern.FLAG_BULLISH if direction == 'bullish' else PriceActionPattern.FLAG_BEARISH,
                                    'confidence': 0.7,
                                    'flagpole_size': move_pct,
                                    'consolidation_level': np.mean(consolidation),
                                    'direction': direction,
                                    'target': move_end + (move_end - move_start) if direction == 'bullish' else move_end - (move_start - move_end)
                                })
                                break  # Une seule détection par analyse
        
        except Exception as e:
            logging.error(f"Erreur détection flags: {e}")
        
        return patterns
    
    def _analyze_timeframe_confluence(self, symbol: str, ohlc_data: pd.DataFrame) -> Dict[str, bool]:
        """⏰ Analyse la confluence multi-timeframe"""
        confluence = {
            'M5_bullish': False,
            'M15_bullish': False, 
            'H1_bullish': False,
            'H4_bullish': False,
            'overall_bullish': False
        }
        
        try:
            # Simule l'analyse multi-timeframe
            # Dans la vraie implémentation, analyserait chaque timeframe
            close = ohlc_data['close'].values
            
            # Trends courts et longs termes
            short_trend = close[-5:].mean() > close[-10:-5].mean()
            medium_trend = close[-10:].mean() > close[-20:-10].mean()
            long_trend = close[-20:].mean() > close[-40:-20].mean() if len(close) >= 40 else True
            
            confluence['M5_bullish'] = short_trend
            confluence['M15_bullish'] = medium_trend
            confluence['H1_bullish'] = long_trend
            confluence['H4_bullish'] = long_trend
            
            # Confluence globale = majorité des timeframes alignés
            bullish_count = sum(confluence.values())
            confluence['overall_bullish'] = bullish_count >= 3
            
        except Exception as e:
            logging.error(f"Erreur analyse confluence: {e}")
        
        return confluence
    
    def _validate_with_volume(self, ohlc_data: pd.DataFrame, patterns: List[Dict]) -> bool:
        """📊 Validation des patterns par le volume"""
        try:
            volume = ohlc_data.get('volume', np.ones(len(ohlc_data))).values
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            
            # Volume élevé confirme les breakouts
            high_volume = current_volume > avg_volume * 1.5
            
            # Pour chaque pattern, ajuste la confiance selon le volume
            for pattern in patterns:
                if pattern['pattern'] in [PriceActionPattern.BREAKOUT_BULLISH, 
                                        PriceActionPattern.BREAKOUT_BEARISH]:
                    if high_volume:
                        pattern['confidence'] *= 1.2  # Boost confiance
                    else:
                        pattern['confidence'] *= 0.8  # Réduit confiance
            
            return high_volume
            
        except Exception as e:
            logging.error(f"Erreur validation volume: {e}")
            return False
    
    def _generate_price_action_signal(self, patterns: List[Dict], sr_levels: Dict, 
                                    market_regime: MarketRegime, confluence: Dict, 
                                    volume_confirmation: bool, ohlc_data: pd.DataFrame) -> Optional[PriceActionSignal]:
        """🎯 Génère le signal final de Price Action"""
        
        if not patterns:
            return None
        
        try:
            # Trie les patterns par confiance
            patterns = sorted(patterns, key=lambda x: x.get('confidence', 0), reverse=True)
            best_pattern = patterns[0]
            
            # Calcule les niveaux d'entrée, SL et TP
            current_price = ohlc_data['close'].iloc[-1]
            
            # Détermine la direction du signal
            direction = best_pattern.get('direction', 'neutral')
            if direction == 'neutral':
                return None
            
            # Calcule les niveaux en fonction du pattern
            if direction == 'bullish':
                entry_price = current_price
                # SL sous le support le plus proche
                nearest_support = max([s for s in sr_levels['supports'] if s < current_price], default=current_price * 0.998)
                stop_loss = nearest_support - (current_price - nearest_support) * 0.2
                
                # TP vers la résistance ou target du pattern
                target = best_pattern.get('target', current_price * 1.01)
                take_profit = min(target, current_price * 1.02)  # Limite à 2%
                
            else:  # bearish
                entry_price = current_price
                # SL au-dessus de la résistance la plus proche
                nearest_resistance = min([r for r in sr_levels['resistances'] if r > current_price], default=current_price * 1.002)
                stop_loss = nearest_resistance + (nearest_resistance - current_price) * 0.2
                
                # TP vers le support ou target du pattern
                target = best_pattern.get('target', current_price * 0.99)
                take_profit = max(target, current_price * 0.98)  # Limite à 2%
            
            # Calcule le Risk/Reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Filtre les signaux avec mauvais R/R
            if rr_ratio < 1.5:
                return None
            
            # Calcule la confiance finale
            base_confidence = best_pattern.get('confidence', 0.5)
            
            # Ajustements de confiance
            if volume_confirmation:
                base_confidence *= 1.1
            if confluence.get('overall_bullish') == (direction == 'bullish'):
                base_confidence *= 1.15
            if market_regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
                base_confidence *= 1.05
            
            # Filtre confiance minimale
            if base_confidence < self.config['min_pattern_confidence']:
                return None
            
            # Calcule le mouvement attendu
            expected_move = abs(take_profit - entry_price) / entry_price * 10000  # En pips
            
            # Crée le signal final
            signal = PriceActionSignal(
                pattern=best_pattern['pattern'],
                confidence=min(base_confidence, 0.95),  # Max 95%
                strength=best_pattern.get('strength', 0.5),
                direction=direction.upper(),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr_ratio,
                market_regime=market_regime,
                timeframe_confluence=confluence,
                volume_confirmation=volume_confirmation,
                momentum_alignment=True,  # Simplifié
                support_resistance_level=nearest_support if direction == 'bullish' else nearest_resistance,
                pattern_completion_bar=len(ohlc_data) - 1,
                expected_move_pips=expected_move
            )
            
            return signal
            
        except Exception as e:
            logging.error(f"Erreur génération signal Price Action: {e}")
            return None
    
    def learn_from_trade_outcome(self, signal: PriceActionSignal, trade_context: TradeContext):
        """🎓 Apprend du résultat du trade pour améliorer les futures prédictions"""
        try:
            # Analyse de la performance du pattern
            was_successful = trade_context.profit_pips > 0
            
            # Met à jour les métriques de performance
            if signal.pattern == PriceActionPattern.BREAKOUT_BULLISH or signal.pattern == PriceActionPattern.BREAKOUT_BEARISH:
                if was_successful:
                    self.performance_metrics['breakout_success_rate'] = (
                        self.performance_metrics.get('breakout_success_rate', 0.5) * 0.9 + 0.1
                    )
                else:
                    self.performance_metrics['breakout_success_rate'] = (
                        self.performance_metrics.get('breakout_success_rate', 0.5) * 0.9
                    )
            
            # Analyse des niveaux Support/Résistance
            if abs(trade_context.entry_price - signal.support_resistance_level) / trade_context.entry_price < 0.005:
                # Trade proche d'un niveau S/R
                if was_successful:
                    self.performance_metrics['support_resistance_accuracy'] = (
                        self.performance_metrics.get('support_resistance_accuracy', 0.5) * 0.9 + 0.1
                    )
            
            # Sauvegarde du contexte pour future analyse
            self.price_action_memory.append({
                'signal': signal,
                'outcome': trade_context,
                'success': was_successful,
                'timestamp': datetime.utcnow()
            })
            
            # Calcule la récompense avec le nouveau système
            market_data = {'current_price': trade_context.entry_price}  # Simplifié
            ai_state = {'account_balance': 10000}  # Simplifié
            
            reward_result = ultra_reward_system.calculate_ultra_reward(
                trade_context, market_data, ai_state
            )
            
            logging.info(f"🎯 Trade appris - Pattern: {signal.pattern.value}, "
                        f"Succès: {was_successful}, Récompense: {reward_result['total_reward']:.3f}")
            
        except Exception as e:
            logging.error(f"Erreur apprentissage trade: {e}")
    
    def get_performance_summary(self) -> Dict:
        """📊 Résumé des performances de l'IA Price Action"""
        return {
            'total_patterns_detected': len(self.pattern_memory),
            'success_rates': {
                'breakouts': self.performance_metrics.get('breakout_success_rate', 0.5),
                'support_resistance': self.performance_metrics.get('support_resistance_accuracy', 0.5),
                'overall_pattern': self.performance_metrics.get('pattern_accuracy', 0.5)
            },
            'current_market_regime': self.current_market_regime.value,
            'volatility_regime': self.volatility_regime,
            'active_sr_levels': len(self.support_resistance_levels.get('supports', [])) + len(self.support_resistance_levels.get('resistances', []))
        }

# Instance globale de l'IA Price Action
ultra_price_action_ai = UltraPriceActionAI()