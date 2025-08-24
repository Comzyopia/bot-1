"""
⚖️ ULTRA RISK MANAGEMENT SYSTEM
═══════════════════════════════════

Système de gestion du risque ultra-avancé pour maximiser
les gains et minimiser les pertes avec des stratégies
sophistiquées de position sizing et de protection.

Version: 3.0 - Ultra Performance Edition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import math

class RiskLevel(Enum):
    """Niveaux de risque dynamiques"""
    ULTRA_CONSERVATIVE = "ultra_conservative"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"

class MarketCondition(Enum):
    """Conditions de marché pour ajustement du risque"""
    VERY_LOW_VOLATILITY = "very_low_vol"
    LOW_VOLATILITY = "low_vol"
    NORMAL_VOLATILITY = "normal_vol"
    HIGH_VOLATILITY = "high_vol"
    EXTREME_VOLATILITY = "extreme_vol"
    NEWS_EVENT = "news_event"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"

@dataclass
class RiskParameters:
    """Paramètres de risque calculés dynamiquement"""
    max_risk_per_trade: float
    position_size: float
    stop_loss_pips: float
    take_profit_pips: float
    max_daily_risk: float
    max_weekly_risk: float
    correlation_limit: int
    leverage_factor: float
    kelly_fraction: float
    var_limit: float
    confidence_threshold: float

class UltraRiskManager:
    """🛡️ Gestionnaire de risque ultra-performant"""
    
    def __init__(self):
        self.account_balance = 10000.0
        self.current_equity = 10000.0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        
        # 📊 TRACKING DES MÉTRIQUES DE RISQUE
        self.risk_metrics = {
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "peak_equity": 10000.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "kelly_criterion": 0.0,
            "win_rate": 0.0,
            "profit_factor": 1.0,
            "expectancy": 0.0,
            "consecutive_losses": 0,
            "max_consecutive_losses": 0,
            "risk_adjusted_return": 0.0
        }
        
        # 🎛️ CONFIGURATION DYNAMIQUE DU RISQUE
        self.risk_config = {
            RiskLevel.ULTRA_CONSERVATIVE: {
                "max_risk_per_trade": 0.005,  # 0.5%
                "max_daily_risk": 0.02,       # 2%
                "max_positions": 2,
                "confidence_threshold": 0.85,
                "leverage_factor": 1.0
            },
            RiskLevel.CONSERVATIVE: {
                "max_risk_per_trade": 0.01,   # 1%
                "max_daily_risk": 0.03,       # 3%
                "max_positions": 3,
                "confidence_threshold": 0.75,
                "leverage_factor": 2.0
            },
            RiskLevel.MODERATE: {
                "max_risk_per_trade": 0.02,   # 2%
                "max_daily_risk": 0.05,       # 5%
                "max_positions": 5,
                "confidence_threshold": 0.65,
                "leverage_factor": 5.0
            },
            RiskLevel.AGGRESSIVE: {
                "max_risk_per_trade": 0.03,   # 3%
                "max_daily_risk": 0.08,       # 8%
                "max_positions": 7,
                "confidence_threshold": 0.55,
                "leverage_factor": 10.0
            },
            RiskLevel.ULTRA_AGGRESSIVE: {
                "max_risk_per_trade": 0.05,   # 5%
                "max_daily_risk": 0.12,       # 12%
                "max_positions": 10,
                "confidence_threshold": 0.45,
                "leverage_factor": 20.0
            }
        }
        
        # 📈 HISTORIQUE POUR CALCULS AVANCÉS
        self.trade_history = []
        self.daily_returns = []
        self.equity_curve = [10000.0]
        
        # 🧠 INTELLIGENCE ADAPTATIVE
        self.market_condition = MarketCondition.NORMAL_VOLATILITY
        self.current_risk_level = RiskLevel.MODERATE
        self.adaptation_enabled = True
        
        logging.info("🛡️ Ultra Risk Manager initialisé avec succès!")
    
    def calculate_optimal_position_size(self, signal_confidence: float, 
                                      entry_price: float, stop_loss: float,
                                      symbol: str, market_condition: MarketCondition = None) -> Dict:
        """🎯 Calcule la taille de position optimale avec Kelly Criterion amélioré"""
        
        try:
            # 1. Détermination du niveau de risque adaptatif
            risk_level = self._determine_adaptive_risk_level(signal_confidence, market_condition)
            risk_params = self.risk_config[risk_level]
            
            # 2. Calcul du risque en pips
            risk_pips = abs(entry_price - stop_loss) * 10000  # Conversion en pips
            
            # 3. Kelly Criterion avec améliorations
            kelly_fraction = self._calculate_advanced_kelly(symbol, signal_confidence)
            
            # 4. Position sizing multi-facteurs
            base_risk = risk_params["max_risk_per_trade"]
            
            # Ajustement basé sur la confiance du signal
            confidence_multiplier = self._calculate_confidence_multiplier(signal_confidence)
            
            # Ajustement basé sur les conditions de marché
            market_multiplier = self._calculate_market_multiplier(market_condition or self.market_condition)
            
            # Ajustement basé sur la performance récente
            performance_multiplier = self._calculate_performance_multiplier()
            
            # Ajustement basé sur la corrélation
            correlation_multiplier = self._calculate_correlation_multiplier(symbol)
            
            # 5. Calcul de la taille finale
            adjusted_risk = base_risk * confidence_multiplier * market_multiplier * performance_multiplier * correlation_multiplier
            
            # Application du Kelly Criterion modifié
            if kelly_fraction > 0:
                kelly_adjusted_risk = min(adjusted_risk, kelly_fraction * 0.5)  # Kelly fractionné pour sécurité
            else:
                kelly_adjusted_risk = adjusted_risk * 0.5  # Réduction si Kelly négatif
            
            # Limite absolue pour éviter le sur-risque
            final_risk = min(kelly_adjusted_risk, 0.05)  # Max 5% par trade
            
            # 6. Calcul de la taille en lots
            risk_amount = self.current_equity * final_risk
            pip_value = self._calculate_pip_value(symbol, entry_price)
            
            if pip_value > 0 and risk_pips > 0:
                lot_size = risk_amount / (risk_pips * pip_value)
                lot_size = round(lot_size, 2)  # Arrondi à 2 décimales
            else:
                lot_size = 0.01  # Lot size minimal de sécurité
            
            # 7. Validation finale
            max_lot_size = self._calculate_max_lot_size(symbol, risk_level)
            lot_size = min(lot_size, max_lot_size)
            
            return {
                "lot_size": lot_size,
                "risk_amount": risk_amount,
                "risk_percentage": final_risk * 100,
                "risk_pips": risk_pips,
                "kelly_fraction": kelly_fraction,
                "confidence_adjustment": confidence_multiplier,
                "market_adjustment": market_multiplier,
                "performance_adjustment": performance_multiplier,
                "correlation_adjustment": correlation_multiplier,
                "risk_level": risk_level.value,
                "recommended": True if lot_size > 0.01 else False
            }
            
        except Exception as e:
            logging.error(f"Erreur calcul position size: {e}")
            return {
                "lot_size": 0.01,
                "risk_amount": self.current_equity * 0.01,
                "risk_percentage": 1.0,
                "recommended": False,
                "error": str(e)
            }
    
    def _determine_adaptive_risk_level(self, confidence: float, market_condition: MarketCondition) -> RiskLevel:
        """🔄 Détermine le niveau de risque de manière adaptative"""
        
        # Base sur la confiance du signal
        if confidence >= 0.9:
            base_level = RiskLevel.AGGRESSIVE
        elif confidence >= 0.8:
            base_level = RiskLevel.MODERATE
        elif confidence >= 0.7:
            base_level = RiskLevel.CONSERVATIVE
        else:
            base_level = RiskLevel.ULTRA_CONSERVATIVE
        
        # Ajustement selon les conditions de marché
        if market_condition in [MarketCondition.EXTREME_VOLATILITY, MarketCondition.NEWS_EVENT]:
            # Réduction du risque en volatilité extrême
            risk_levels = list(RiskLevel)
            current_index = risk_levels.index(base_level)
            adjusted_index = max(0, current_index - 1)  # Plus conservateur
            base_level = risk_levels[adjusted_index]
        
        # Ajustement selon la performance récente
        if self.risk_metrics["consecutive_losses"] >= 3:
            # Réduction après pertes consécutives
            risk_levels = list(RiskLevel)
            current_index = risk_levels.index(base_level)
            adjusted_index = max(0, current_index - 1)
            base_level = risk_levels[adjusted_index]
        
        # Ajustement selon le drawdown
        if self.risk_metrics["current_drawdown"] > 0.1:  # 10%
            risk_levels = list(RiskLevel)
            current_index = risk_levels.index(base_level)
            adjusted_index = max(0, current_index - 2)  # Très conservateur
            base_level = risk_levels[adjusted_index]
        
        return base_level
    
    def _calculate_advanced_kelly(self, symbol: str, confidence: float) -> float:
        """🧮 Calcule le Kelly Criterion avancé avec historical data"""
        
        try:
            # Récupère l'historique des trades pour ce symbole
            symbol_trades = [t for t in self.trade_history[-50:] 
                           if t.get('symbol') == symbol]
            
            if len(symbol_trades) < 10:
                # Pas assez d'historique, utilise Kelly conservateur
                return confidence * 0.1  # Kelly fractionné
            
            # Calcule win rate et average win/loss
            wins = [t for t in symbol_trades if t.get('profit', 0) > 0]
            losses = [t for t in symbol_trades if t.get('profit', 0) < 0]
            
            if not wins or not losses:
                return confidence * 0.05  # Très conservateur
            
            win_rate = len(wins) / len(symbol_trades)
            avg_win = np.mean([t['profit'] for t in wins])
            avg_loss = abs(np.mean([t['profit'] for t in losses]))
            
            if avg_loss == 0:
                return 0.05
            
            # Kelly formula: f = (bp - q) / b
            # où b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly = (b * p - q) / b
            
            # Limitations de sécurité
            kelly = max(0, min(kelly, 0.25))  # Entre 0 et 25%
            
            # Ajustement par la confiance
            adjusted_kelly = kelly * confidence
            
            return adjusted_kelly
            
        except Exception as e:
            logging.error(f"Erreur calcul Kelly: {e}")
            return 0.02  # Fallback conservateur
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """📊 Calcule le multiplicateur basé sur la confiance"""
        # Fonction non-linéaire pour récompenser la haute confiance
        if confidence >= 0.9:
            return 1.5  # Boost pour très haute confiance
        elif confidence >= 0.8:
            return 1.2
        elif confidence >= 0.7:
            return 1.0
        elif confidence >= 0.6:
            return 0.8
        else:
            return 0.5  # Réduction drastique pour faible confiance
    
    def _calculate_market_multiplier(self, market_condition: MarketCondition) -> float:
        """🌍 Calcule le multiplicateur basé sur les conditions de marché"""
        multipliers = {
            MarketCondition.VERY_LOW_VOLATILITY: 0.7,    # Moins d'opportunités
            MarketCondition.LOW_VOLATILITY: 0.9,
            MarketCondition.NORMAL_VOLATILITY: 1.0,
            MarketCondition.HIGH_VOLATILITY: 1.1,        # Plus d'opportunités
            MarketCondition.EXTREME_VOLATILITY: 0.6,     # Trop risqué
            MarketCondition.NEWS_EVENT: 0.5,             # Très risqué
            MarketCondition.MARKET_OPEN: 0.8,            # Spread élargi
            MarketCondition.MARKET_CLOSE: 0.7             # Faible liquidité
        }
        
        return multipliers.get(market_condition, 1.0)
    
    def _calculate_performance_multiplier(self) -> float:
        """🏆 Calcule le multiplicateur basé sur la performance récente"""
        
        # Basé sur les profits récents
        if len(self.daily_returns) >= 5:
            recent_performance = np.mean(self.daily_returns[-5:])
            
            if recent_performance > 0.02:  # Excellente performance
                return 1.3
            elif recent_performance > 0.01:  # Bonne performance
                return 1.1
            elif recent_performance > -0.01:  # Performance neutre
                return 1.0
            elif recent_performance > -0.02:  # Mauvaise performance
                return 0.8
            else:  # Très mauvaise performance
                return 0.6
        
        # Basé sur le drawdown actuel
        current_dd = self.risk_metrics["current_drawdown"]
        if current_dd > 0.15:  # Drawdown > 15%
            return 0.5
        elif current_dd > 0.10:  # Drawdown > 10%
            return 0.7
        elif current_dd > 0.05:  # Drawdown > 5%
            return 0.9
        else:
            return 1.0
    
    def _calculate_correlation_multiplier(self, symbol: str) -> float:
        """🔗 Calcule le multiplicateur basé sur la corrélation des positions"""
        
        # Simule l'analyse de corrélation (dans la vraie implémentation, utiliserait les positions actuelles)
        
        # Groupes de corrélation
        major_pairs = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
        yen_pairs = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"]
        usd_pairs = ["USDCHF", "USDCAD", "USDSEK", "USDNOK"]
        
        current_positions = getattr(self, 'current_positions', [])
        
        # Compte les positions dans le même groupe
        same_group_count = 0
        
        if symbol in major_pairs:
            same_group_count = sum(1 for pos in current_positions 
                                 if pos.get('symbol') in major_pairs)
        elif symbol in yen_pairs:
            same_group_count = sum(1 for pos in current_positions 
                                 if pos.get('symbol') in yen_pairs)
        elif symbol in usd_pairs:
            same_group_count = sum(1 for pos in current_positions 
                                 if pos.get('symbol') in usd_pairs)
        
        # Réduit le risque si trop de corrélation
        if same_group_count >= 3:
            return 0.5  # Forte réduction
        elif same_group_count >= 2:
            return 0.7  # Réduction modérée
        else:
            return 1.0  # Pas de réduction
    
    def _calculate_pip_value(self, symbol: str, price: float) -> float:
        """💰 Calcule la valeur d'un pip pour le symbole"""
        # Valeurs approximatives pour 1 lot standard
        pip_values = {
            "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0, "NZDUSD": 10.0,
            "USDJPY": 10.0 / price * 100, "USDCHF": 10.0 / price,
            "USDCAD": 10.0 / price, "EURJPY": 10.0 / price * 100,
            "GBPJPY": 10.0 / price * 100, "EURGBP": 10.0,
        }
        
        return pip_values.get(symbol, 10.0)  # Défaut à 10 USD/pip
    
    def _calculate_max_lot_size(self, symbol: str, risk_level: RiskLevel) -> float:
        """📏 Calcule la taille maximale de lot autorisée"""
        
        # Limites basées sur le capital et le risk level
        max_risk_amount = self.current_equity * self.risk_config[risk_level]["max_risk_per_trade"]
        
        # Limites absolues de sécurité
        max_lots_by_equity = {
            1000: 0.1,    # 1K = max 0.1 lot
            5000: 0.5,    # 5K = max 0.5 lot  
            10000: 1.0,   # 10K = max 1 lot
            50000: 5.0,   # 50K = max 5 lots
            100000: 10.0  # 100K = max 10 lots
        }
        
        for equity_threshold, max_lots in sorted(max_lots_by_equity.items()):
            if self.current_equity <= equity_threshold:
                return max_lots
        
        return 10.0  # Max absolu
    
    def calculate_dynamic_stop_loss(self, entry_price: float, direction: str,
                                  symbol: str, atr: float = None, 
                                  support_resistance: float = None) -> Dict:
        """🛑 Calcule un stop loss dynamique optimisé"""
        
        try:
            # 1. ATR-based stop loss
            if atr is None:
                atr = self._estimate_atr(symbol, entry_price)
            
            atr_multiplier = 2.0  # Multiplicateur ATR standard
            
            if direction.upper() == "BUY":
                atr_sl = entry_price - (atr * atr_multiplier)
            else:
                atr_sl = entry_price + (atr * atr_multiplier)
            
            # 2. Support/Resistance based stop loss
            sr_sl = support_resistance
            
            # 3. Percentage based stop loss (fallback)
            pct_sl_distance = 0.015  # 1.5% par défaut
            if direction.upper() == "BUY":
                pct_sl = entry_price * (1 - pct_sl_distance)
            else:
                pct_sl = entry_price * (1 + pct_sl_distance)
            
            # 4. Volatility adjusted stop loss
            market_volatility = self._estimate_market_volatility(symbol)
            vol_multiplier = 1.0 + (market_volatility - 0.02) / 0.02  # Ajustement selon volatilité
            vol_multiplier = max(0.5, min(vol_multiplier, 2.0))  # Limites
            
            # 5. Sélection du stop loss optimal
            candidates = [atr_sl, pct_sl]
            if sr_sl is not None:
                candidates.append(sr_sl)
            
            if direction.upper() == "BUY":
                # Pour un achat, prendre le SL le plus haut (moins risqué)
                optimal_sl = max(candidates)
            else:
                # Pour une vente, prendre le SL le plus bas (moins risqué)  
                optimal_sl = min(candidates)
            
            # Application du multiplicateur de volatilité
            if direction.upper() == "BUY":
                final_sl = entry_price - abs(entry_price - optimal_sl) * vol_multiplier
            else:
                final_sl = entry_price + abs(optimal_sl - entry_price) * vol_multiplier
            
            # 6. Validation finale
            risk_pips = abs(entry_price - final_sl) * 10000
            max_risk_pips = 100  # Maximum 100 pips de risque
            
            if risk_pips > max_risk_pips:
                if direction.upper() == "BUY":
                    final_sl = entry_price - (max_risk_pips / 10000)
                else:
                    final_sl = entry_price + (max_risk_pips / 10000)
            
            return {
                "stop_loss": final_sl,
                "risk_pips": abs(entry_price - final_sl) * 10000,
                "atr_based": atr_sl,
                "percentage_based": pct_sl,
                "sr_based": sr_sl,
                "volatility_multiplier": vol_multiplier,
                "method_used": "hybrid_optimized"
            }
            
        except Exception as e:
            logging.error(f"Erreur calcul stop loss: {e}")
            # Stop loss de sécurité
            safety_distance = 0.02  # 2%
            if direction.upper() == "BUY":
                safety_sl = entry_price * (1 - safety_distance)
            else:
                safety_sl = entry_price * (1 + safety_distance)
            
            return {
                "stop_loss": safety_sl,
                "risk_pips": abs(entry_price - safety_sl) * 10000,
                "method_used": "safety_fallback"
            }
    
    def calculate_dynamic_take_profit(self, entry_price: float, stop_loss: float,
                                    direction: str, symbol: str, 
                                    signal_strength: float = 0.5) -> Dict:
        """🎯 Calcule un take profit dynamique optimisé"""
        
        try:
            risk_distance = abs(entry_price - stop_loss)
            
            # 1. Risk/Reward basique selon la force du signal
            base_rr_ratio = 2.0  # Défaut 1:2
            
            # Ajustement selon la force du signal
            if signal_strength >= 0.9:
                rr_ratio = 3.5  # Très fort signal = TP plus élevé
            elif signal_strength >= 0.8:
                rr_ratio = 3.0
            elif signal_strength >= 0.7:
                rr_ratio = 2.5
            elif signal_strength >= 0.6:
                rr_ratio = 2.0
            else:
                rr_ratio = 1.5  # Signal faible = TP conservateur
            
            # 2. TP basé sur R/R
            if direction.upper() == "BUY":
                rr_tp = entry_price + (risk_distance * rr_ratio)
            else:
                rr_tp = entry_price - (risk_distance * rr_ratio)
            
            # 3. TP basé sur ATR (mouvement attendu)
            atr = self._estimate_atr(symbol, entry_price)
            atr_target_multiplier = 3.0  # 3x ATR comme target
            
            if direction.upper() == "BUY":
                atr_tp = entry_price + (atr * atr_target_multiplier)
            else:
                atr_tp = entry_price - (atr * atr_target_multiplier)
            
            # 4. TP basé sur les niveaux techniques (si disponibles)
            # Simule la détection de résistance/support comme target
            technical_levels = self._get_technical_targets(symbol, entry_price, direction)
            
            # 5. Sélection du TP optimal
            candidates = [rr_tp, atr_tp]
            if technical_levels:
                candidates.extend(technical_levels)
            
            if direction.upper() == "BUY":
                # Pour un achat, prendre le TP le plus conservateur (le plus bas)
                optimal_tp = min(candidates)
            else:
                # Pour une vente, prendre le TP le plus conservateur (le plus haut)
                optimal_tp = max(candidates)
            
            # 6. Validation et ajustements finaux
            final_rr = abs(optimal_tp - entry_price) / risk_distance
            
            # Assure un R/R minimum de 1.5
            if final_rr < 1.5:
                if direction.upper() == "BUY":
                    optimal_tp = entry_price + (risk_distance * 1.5)
                else:
                    optimal_tp = entry_price - (risk_distance * 1.5)
                final_rr = 1.5
            
            # Limite le R/R maximum à 5.0 pour éviter les TP irréalistes
            elif final_rr > 5.0:
                if direction.upper() == "BUY":
                    optimal_tp = entry_price + (risk_distance * 5.0)
                else:
                    optimal_tp = entry_price - (risk_distance * 5.0)
                final_rr = 5.0
            
            return {
                "take_profit": optimal_tp,
                "risk_reward_ratio": final_rr,
                "profit_pips": abs(optimal_tp - entry_price) * 10000,
                "rr_based": rr_tp,
                "atr_based": atr_tp,
                "technical_levels": technical_levels,
                "signal_strength_used": signal_strength,
                "method_used": "hybrid_optimized"
            }
            
        except Exception as e:
            logging.error(f"Erreur calcul take profit: {e}")
            # TP de sécurité (R/R 2:1)
            risk_distance = abs(entry_price - stop_loss)
            if direction.upper() == "BUY":
                safety_tp = entry_price + (risk_distance * 2.0)
            else:
                safety_tp = entry_price - (risk_distance * 2.0)
            
            return {
                "take_profit": safety_tp,
                "risk_reward_ratio": 2.0,
                "method_used": "safety_fallback"
            }
    
    def _estimate_atr(self, symbol: str, current_price: float) -> float:
        """📊 Estime l'ATR pour le symbole (approximation)"""
        # ATR approximatifs basés sur la volatilité typique des paires
        atr_estimates = {
            "EURUSD": current_price * 0.008,  # ~80 pips
            "GBPUSD": current_price * 0.012,  # ~120 pips
            "USDJPY": current_price * 0.010,  # ~100 pips
            "USDCHF": current_price * 0.009,  # ~90 pips
            "AUDUSD": current_price * 0.011,  # ~110 pips
            "USDCAD": current_price * 0.009,  # ~90 pips
            "NZDUSD": current_price * 0.013,  # ~130 pips
        }
        
        return atr_estimates.get(symbol, current_price * 0.010)  # Défaut 1%
    
    def _estimate_market_volatility(self, symbol: str) -> float:
        """🌊 Estime la volatilité du marché (approximation)"""
        # Simule l'estimation de volatilité
        # Dans la vraie implémentation, calculerait à partir des données historiques
        return 0.02  # 2% de volatilité par défaut
    
    def _get_technical_targets(self, symbol: str, entry_price: float, direction: str) -> List[float]:
        """🎯 Obtient les targets techniques (simulation)"""
        # Simule la détection de niveaux techniques
        # Dans la vraie implémentation, utiliserait les vrais niveaux S/R
        targets = []
        
        if direction.upper() == "BUY":
            # Résistances au-dessus
            targets = [
                entry_price * 1.01,   # +1%
                entry_price * 1.02,   # +2%
                entry_price * 1.035   # +3.5%
            ]
        else:
            # Supports en-dessous
            targets = [
                entry_price * 0.99,   # -1%
                entry_price * 0.98,   # -2%
                entry_price * 0.965   # -3.5%
            ]
        
        return targets
    
    def update_performance_metrics(self, trade_result: Dict):
        """📊 Met à jour les métriques de performance"""
        try:
            profit = trade_result.get('profit_usd', 0)
            
            # Met à jour l'équité
            self.current_equity += profit
            self.daily_pnl += profit
            
            # Met à jour l'historique
            self.trade_history.append(trade_result)
            
            # Calcule le drawdown
            self.risk_metrics["peak_equity"] = max(self.risk_metrics["peak_equity"], self.current_equity)
            current_dd = (self.risk_metrics["peak_equity"] - self.current_equity) / self.risk_metrics["peak_equity"]
            self.risk_metrics["current_drawdown"] = current_dd
            self.risk_metrics["max_drawdown"] = max(self.risk_metrics["max_drawdown"], current_dd)
            
            # Perte consécutive
            if profit < 0:
                self.risk_metrics["consecutive_losses"] += 1
                self.risk_metrics["max_consecutive_losses"] = max(
                    self.risk_metrics["max_consecutive_losses"],
                    self.risk_metrics["consecutive_losses"]
                )
            else:
                self.risk_metrics["consecutive_losses"] = 0
            
            # Calcule les métriques avancées
            self._calculate_advanced_metrics()
            
            logging.info(f"📊 Métriques mises à jour - Equity: ${self.current_equity:.2f}, "
                        f"DD: {current_dd*100:.1f}%")
            
        except Exception as e:
            logging.error(f"Erreur mise à jour métriques: {e}")
    
    def _calculate_advanced_metrics(self):
        """📈 Calcule les métriques avancées (Sharpe, Sortino, etc.)"""
        if len(self.trade_history) < 10:
            return
        
        try:
            # Récupère les returns
            returns = [t.get('profit_usd', 0) / self.account_balance for t in self.trade_history[-50:]]
            
            if not returns:
                return
            
            # Sharpe Ratio
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                self.risk_metrics["sharpe_ratio"] = mean_return / std_return * np.sqrt(252)
            
            # Sortino Ratio (utilise seulement la volatilité négative)
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    self.risk_metrics["sortino_ratio"] = mean_return / downside_std * np.sqrt(252)
            
            # Win Rate
            winning_trades = len([t for t in self.trade_history[-50:] if t.get('profit_usd', 0) > 0])
            self.risk_metrics["win_rate"] = winning_trades / min(len(self.trade_history), 50)
            
            # Profit Factor
            gross_profit = sum([t.get('profit_usd', 0) for t in self.trade_history[-50:] if t.get('profit_usd', 0) > 0])
            gross_loss = abs(sum([t.get('profit_usd', 0) for t in self.trade_history[-50:] if t.get('profit_usd', 0) < 0]))
            
            if gross_loss > 0:
                self.risk_metrics["profit_factor"] = gross_profit / gross_loss
            
            # Expectancy
            avg_win = gross_profit / max(1, winning_trades)
            avg_loss = gross_loss / max(1, len(self.trade_history[-50:]) - winning_trades)
            win_rate = self.risk_metrics["win_rate"]
            
            self.risk_metrics["expectancy"] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
        except Exception as e:
            logging.error(f"Erreur calcul métriques avancées: {e}")
    
    def should_stop_trading(self) -> Dict:
        """🚨 Détermine si le trading doit être arrêté (protection)"""
        
        stop_reasons = []
        should_stop = False
        
        # 1. Drawdown excessif
        if self.risk_metrics["current_drawdown"] > 0.2:  # 20%
            stop_reasons.append("Drawdown excessif (>20%)")
            should_stop = True
        
        # 2. Pertes consécutives
        if self.risk_metrics["consecutive_losses"] >= 5:
            stop_reasons.append("5 pertes consécutives")
            should_stop = True
        
        # 3. Perte journalière excessive
        daily_loss_limit = self.account_balance * 0.05  # 5% par jour
        if self.daily_pnl < -daily_loss_limit:
            stop_reasons.append("Limite de perte journalière atteinte")
            should_stop = True
        
        # 4. Performance très dégradée
        if (self.risk_metrics["win_rate"] < 0.3 and 
            len(self.trade_history) > 20):
            stop_reasons.append("Win rate très faible (<30%)")
            should_stop = True
        
        return {
            "should_stop": should_stop,
            "reasons": stop_reasons,
            "current_drawdown": self.risk_metrics["current_drawdown"],
            "consecutive_losses": self.risk_metrics["consecutive_losses"],
            "daily_pnl": self.daily_pnl,
            "win_rate": self.risk_metrics["win_rate"]
        }
    
    def get_risk_summary(self) -> Dict:
        """📋 Résumé complet des métriques de risque"""
        return {
            "account_info": {
                "balance": self.account_balance,
                "equity": self.current_equity,
                "daily_pnl": self.daily_pnl,
                "pnl_percentage": (self.daily_pnl / self.account_balance) * 100
            },
            "risk_metrics": self.risk_metrics.copy(),
            "current_settings": {
                "risk_level": self.current_risk_level.value,
                "market_condition": self.market_condition.value,
                "adaptation_enabled": self.adaptation_enabled
            },
            "performance_summary": {
                "total_trades": len(self.trade_history),
                "recent_performance": "positive" if self.daily_pnl > 0 else "negative",
                "risk_score": self._calculate_risk_score()
            }
        }
    
    def _calculate_risk_score(self) -> str:
        """🎯 Calcule un score de risque global"""
        score = 0
        
        # Drawdown
        if self.risk_metrics["current_drawdown"] < 0.05:
            score += 3
        elif self.risk_metrics["current_drawdown"] < 0.10:
            score += 2
        elif self.risk_metrics["current_drawdown"] < 0.15:
            score += 1
        
        # Win rate
        if self.risk_metrics["win_rate"] > 0.6:
            score += 3
        elif self.risk_metrics["win_rate"] > 0.5:
            score += 2
        elif self.risk_metrics["win_rate"] > 0.4:
            score += 1
        
        # Sharpe ratio
        if self.risk_metrics["sharpe_ratio"] > 2.0:
            score += 3
        elif self.risk_metrics["sharpe_ratio"] > 1.0:
            score += 2
        elif self.risk_metrics["sharpe_ratio"] > 0.5:
            score += 1
        
        # Classification
        if score >= 8:
            return "EXCELLENT"
        elif score >= 6:
            return "GOOD"
        elif score >= 4:
            return "MODERATE"
        elif score >= 2:
            return "POOR"
        else:
            return "CRITICAL"

# Instance globale du gestionnaire de risque
ultra_risk_manager = UltraRiskManager()