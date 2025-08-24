"""
🚀 SYSTÈME DE RÉCOMPENSES ULTRA-AVANCÉ POUR BOT DE TRADING
═══════════════════════════════════════════════════════════

Ce système révolutionnaire calcule des récompenses multi-dimensionnelles
basées sur le Price Action, la gestion du risque, et la performance globale.

Version: 3.0 - Ultra Performance Edition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import math

class MarketRegime(Enum):
    """Régimes de marché détectés"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

@dataclass
class TradeContext:
    """Contexte complet d'un trade pour l'analyse des récompenses"""
    symbol: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    position_size: float
    direction: str  # "BUY" or "SELL"
    profit_pips: float
    profit_usd: float
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE
    market_regime: Optional[MarketRegime] = None
    confidence_at_entry: float = 0.5
    volatility_at_entry: float = 0.0
    support_resistance_distance: float = 0.0
    pattern_strength: float = 0.0

class UltraRewardSystem:
    """🎯 Système de récompenses ultra-performant basé sur Price Action"""
    
    def __init__(self):
        self.reward_history = []
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "kelly_criterion": 0.0,
            "var_95": 0.0,
            "expected_value": 0.0,
            "consistency_score": 0.0,
            "risk_adjusted_returns": 0.0
        }
        
        # 🎛️ POIDS DES DIFFÉRENTES COMPOSANTES DE RÉCOMPENSE
        self.reward_weights = {
            "profit_reward": 0.25,           # 25% - Profit de base
            "price_action_reward": 0.30,     # 30% - Price Action Analysis
            "risk_management_reward": 0.25,  # 25% - Gestion du risque
            "timing_reward": 0.10,           # 10% - Timing d'entrée/sortie
            "consistency_reward": 0.10,      # 10% - Cohérence stratégique
        }
        
        # 📊 SEUILS POUR LES MÉTRIQUES DE PERFORMANCE
        self.performance_thresholds = {
            "excellent_winrate": 0.65,
            "good_winrate": 0.55,
            "excellent_sharpe": 2.0,
            "good_sharpe": 1.5,
            "max_acceptable_drawdown": 0.15,
            "excellent_profit_factor": 2.0,
            "good_profit_factor": 1.5
        }
        
    def calculate_ultra_reward(self, trade_context: TradeContext, 
                             market_data: Dict, ai_state: Dict) -> Dict:
        """🚀 Calcule la récompense ultra-avancée pour un trade"""
        
        total_reward = 0.0
        reward_breakdown = {}
        
        # 1. 💰 RÉCOMPENSE DE PROFIT (Price Action orientée)
        profit_reward = self._calculate_profit_reward(trade_context, market_data)
        total_reward += profit_reward * self.reward_weights["profit_reward"]
        reward_breakdown["profit"] = profit_reward
        
        # 2. 📈 RÉCOMPENSE PRICE ACTION (Nouveauté majeure !)
        price_action_reward = self._calculate_price_action_reward(trade_context, market_data)
        total_reward += price_action_reward * self.reward_weights["price_action_reward"]
        reward_breakdown["price_action"] = price_action_reward
        
        # 3. ⚖️ RÉCOMPENSE GESTION DU RISQUE
        risk_reward = self._calculate_risk_management_reward(trade_context, ai_state)
        total_reward += risk_reward * self.reward_weights["risk_management_reward"]
        reward_breakdown["risk_management"] = risk_reward
        
        # 4. ⏰ RÉCOMPENSE TIMING
        timing_reward = self._calculate_timing_reward(trade_context, market_data)
        total_reward += timing_reward * self.reward_weights["timing_reward"]
        reward_breakdown["timing"] = timing_reward
        
        # 5. 🎯 RÉCOMPENSE COHÉRENCE
        consistency_reward = self._calculate_consistency_reward(trade_context, ai_state)
        total_reward += consistency_reward * self.reward_weights["consistency_reward"]
        reward_breakdown["consistency"] = consistency_reward
        
        # 6. 🌟 BONUS SPÉCIAUX
        special_bonuses = self._calculate_special_bonuses(trade_context, market_data)
        total_reward += special_bonuses
        reward_breakdown["special_bonuses"] = special_bonuses
        
        # 7. 📊 MISE À JOUR DES MÉTRIQUES
        self._update_performance_metrics(trade_context)
        
        # 8. 🔄 RÉCOMPENSE ADAPTIVE SELON PERFORMANCE GLOBALE
        adaptive_multiplier = self._calculate_adaptive_multiplier()
        final_reward = total_reward * adaptive_multiplier
        
        return {
            "total_reward": final_reward,
            "base_reward": total_reward,
            "adaptive_multiplier": adaptive_multiplier,
            "breakdown": reward_breakdown,
            "performance_metrics": self.performance_metrics.copy(),
            "improvement_suggestions": self._generate_improvement_suggestions(trade_context, reward_breakdown)
        }
    
    def _calculate_profit_reward(self, trade_context: TradeContext, market_data: Dict) -> float:
        """💰 Calcule la récompense basée sur le profit (améliorée)"""
        profit_pips = trade_context.profit_pips
        profit_usd = trade_context.profit_usd
        
        # Base reward normalisée
        base_reward = np.tanh(profit_pips / 10.0)  # Normalise entre -1 et 1
        
        # Bonus pour gros profits
        if profit_pips > 20:
            base_reward += min(0.5, (profit_pips - 20) / 100)
        
        # Bonus pour Risk/Reward ratio excellent
        if trade_context.sl_price and trade_context.tp_price:
            risk = abs(trade_context.entry_price - trade_context.sl_price)
            reward_potential = abs(trade_context.tp_price - trade_context.entry_price)
            if risk > 0:
                rr_ratio = reward_potential / risk
                if rr_ratio >= 3.0:  # Excellent R:R
                    base_reward += 0.3
                elif rr_ratio >= 2.0:  # Bon R:R
                    base_reward += 0.15
        
        # Pénalité pour petits profits avec gros risque
        if 0 < profit_pips < 5 and trade_context.max_adverse_excursion > 15:
            base_reward -= 0.2
        
        return np.clip(base_reward, -2.0, 2.0)
    
    def _calculate_price_action_reward(self, trade_context: TradeContext, market_data: Dict) -> float:
        """📈 NOUVELLE FONCTIONNALITÉ: Récompense Price Action Analysis"""
        price_action_score = 0.0
        
        # 1. 🎯 RÉCOMPENSE SUPPORT/RÉSISTANCE
        sr_reward = self._analyze_support_resistance_quality(trade_context, market_data)
        price_action_score += sr_reward * 0.3
        
        # 2. 📊 RÉCOMPENSE PATTERN RECOGNITION
        pattern_reward = self._analyze_pattern_quality(trade_context, market_data)
        price_action_score += pattern_reward * 0.25
        
        # 3. 🌊 RÉCOMPENSE MOMENTUM ANALYSIS
        momentum_reward = self._analyze_momentum_quality(trade_context, market_data)
        price_action_score += momentum_reward * 0.25
        
        # 4. 💥 RÉCOMPENSE BREAKOUT QUALITY
        breakout_reward = self._analyze_breakout_quality(trade_context, market_data)
        price_action_score += breakout_reward * 0.2
        
        return np.clip(price_action_score, -1.5, 1.5)
    
    def _analyze_support_resistance_quality(self, trade_context: TradeContext, market_data: Dict) -> float:
        """🎯 Analyse la qualité des niveaux de support/résistance"""
        # Simule l'analyse S/R (dans la vraie implémentation, utiliserait les données de marché réelles)
        distance_to_sr = trade_context.support_resistance_distance
        
        # Récompense pour trades près des niveaux S/R importants
        if distance_to_sr < 5:  # Très proche d'un niveau S/R
            if trade_context.profit_pips > 0:  # Et profitable
                return 1.2  # Excellente lecture du marché
            else:
                return -0.3  # Mauvaise lecture malgré bon niveau
        elif distance_to_sr < 10:  # Assez proche
            return 0.6 if trade_context.profit_pips > 0 else -0.1
        else:
            return 0.0  # Pas de bonus/malus particulier
    
    def _analyze_pattern_quality(self, trade_context: TradeContext, market_data: Dict) -> float:
        """📊 Analyse la qualité des patterns de price action"""
        pattern_strength = trade_context.pattern_strength
        
        # Récompense basée sur la force du pattern détecté
        if pattern_strength > 0.8:  # Pattern très fort
            return 1.0 if trade_context.profit_pips > 0 else -0.5
        elif pattern_strength > 0.6:  # Pattern modéré
            return 0.5 if trade_context.profit_pips > 0 else -0.2
        else:
            return 0.0
    
    def _analyze_momentum_quality(self, trade_context: TradeContext, market_data: Dict) -> float:
        """🌊 Analyse la qualité du momentum au moment du trade"""
        # Analyse MFE vs MAE pour comprendre la qualité du timing
        mfe = trade_context.max_favorable_excursion
        mae = trade_context.max_adverse_excursion
        
        if mfe > mae * 2:  # Le trade a rapidement évolué en notre faveur
            return 0.8
        elif mfe > mae:  # Momentum favorable
            return 0.4
        elif mae > mfe * 2:  # Momentum très défavorable
            return -0.6
        else:
            return -0.2
    
    def _analyze_breakout_quality(self, trade_context: TradeContext, market_data: Dict) -> float:
        """💥 Analyse la qualité des breakouts"""
        # Détecte si c'était un vrai breakout ou un faux signal
        volatility = trade_context.volatility_at_entry
        
        if trade_context.market_regime == MarketRegime.BREAKOUT:
            if trade_context.profit_pips > 15:  # Breakout réussi
                return 1.0
            elif trade_context.profit_pips < -5:  # Faux breakout
                return -0.8
        
        return 0.0
    
    def _calculate_risk_management_reward(self, trade_context: TradeContext, ai_state: Dict) -> float:
        """⚖️ Récompense pour la gestion du risque"""
        risk_score = 0.0
        
        # 1. Position sizing approprié
        account_balance = ai_state.get('account_balance', 10000)
        position_risk = abs(trade_context.profit_usd) / account_balance
        
        if position_risk <= 0.01:  # Risque très conservateur (1%)
            risk_score += 0.5
        elif position_risk <= 0.02:  # Risque modéré (2%)
            risk_score += 0.3
        elif position_risk > 0.05:  # Risque trop élevé (>5%)
            risk_score -= 0.5
        
        # 2. Stop Loss utilisé
        if trade_context.sl_price is not None:
            risk_score += 0.3
            # Bonus si SL pas touché mais trade profitable
            if trade_context.profit_pips > 0:
                risk_score += 0.2
        else:
            risk_score -= 0.4  # Pénalité pour absence de SL
        
        # 3. Gestion du drawdown personnel
        current_dd = self.performance_metrics.get('current_drawdown', 0)
        if current_dd > 0.10:  # Si en drawdown >10%
            if trade_context.profit_pips > 0:  # Mais trade profitable
                risk_score += 0.4  # Bonus pour sortie de drawdown
        
        # 4. Éviter l'over-leveraging durant haute volatilité
        if trade_context.volatility_at_entry > 0.02:  # Haute volatilité
            if position_risk <= 0.01:  # Mais position conservative
                risk_score += 0.3
        
        return np.clip(risk_score, -1.0, 1.5)
    
    def _calculate_timing_reward(self, trade_context: TradeContext, market_data: Dict) -> float:
        """⏰ Récompense pour la qualité du timing"""
        timing_score = 0.0
        
        # 1. Durée du trade appropriée selon le timeframe
        trade_duration = (trade_context.exit_time - trade_context.entry_time).total_seconds() / 3600
        
        # Pour M5/M15: trades courts favorisés
        if trade_duration < 2 and trade_context.profit_pips > 0:
            timing_score += 0.5  # Excellent timing
        elif trade_duration > 24 and trade_context.profit_pips < 5:
            timing_score -= 0.3  # Trop long pour peu de gain
        
        # 2. Efficacité du trade (MFE analysis)
        if trade_context.max_favorable_excursion > 0:
            efficiency = trade_context.profit_pips / trade_context.max_favorable_excursion
            if efficiency > 0.8:  # Sorti proche du top
                timing_score += 0.4
            elif efficiency < 0.3:  # Sorti tôt
                timing_score -= 0.2
        
        # 3. Éviter les trades en fin de session (si données disponibles)
        entry_hour = trade_context.entry_time.hour
        if 22 <= entry_hour <= 23 or 0 <= entry_hour <= 1:  # Sessions creuses
            if trade_context.profit_pips < 0:
                timing_score -= 0.3
        
        return np.clip(timing_score, -0.8, 0.8)
    
    def _calculate_consistency_reward(self, trade_context: TradeContext, ai_state: Dict) -> float:
        """🎯 Récompense pour la cohérence stratégique"""
        consistency_score = 0.0
        
        # 1. Éviter l'over-trading
        recent_trades = len([r for r in self.reward_history[-20:] if r])  # 20 derniers trades
        if recent_trades > 15:  # Trop de trades récents
            consistency_score -= 0.4
        elif recent_trades < 5:  # Pas assez actif
            consistency_score -= 0.1
        else:
            consistency_score += 0.2
        
        # 2. Cohérence dans le même symbole
        same_symbol_trades = [r for r in self.reward_history[-10:] 
                             if r and r.get('symbol') == trade_context.symbol]
        if len(same_symbol_trades) >= 3:
            avg_performance = np.mean([t.get('profit_pips', 0) for t in same_symbol_trades])
            if avg_performance > 5:  # Bon sur ce symbole
                consistency_score += 0.3
            elif avg_performance < -5:  # Mauvais sur ce symbole
                consistency_score -= 0.3
        
        # 3. Respect des règles de gestion de risque
        if hasattr(self, 'consecutive_losses'):
            if self.consecutive_losses >= 3 and trade_context.profit_pips > 0:
                consistency_score += 0.5  # Bonus pour casser série de pertes
        
        return np.clip(consistency_score, -0.6, 0.6)
    
    def _calculate_special_bonuses(self, trade_context: TradeContext, market_data: Dict) -> float:
        """🌟 Bonus spéciaux pour performances exceptionnelles"""
        bonus = 0.0
        
        # 1. 🏆 Bonus "Home Run" pour très gros gains
        if trade_context.profit_pips > 50:
            bonus += min(0.5, trade_context.profit_pips / 100)
        
        # 2. 🛡️ Bonus "Risk Master" pour excellent R:R avec profit
        if (trade_context.sl_price and trade_context.tp_price and 
            trade_context.profit_pips > 0):
            risk = abs(trade_context.entry_price - trade_context.sl_price)
            if risk > 0:
                rr_achieved = trade_context.profit_pips / (risk * 10000)  # Convert to pips
                if rr_achieved >= 4.0:
                    bonus += 0.3
        
        # 3. 🎯 Bonus "Sniper" pour trades ultra-précis
        if (trade_context.max_adverse_excursion < 3 and 
            trade_context.profit_pips > 10):
            bonus += 0.2
        
        # 4. 🌪️ Bonus "Storm Rider" pour trader bien durant haute volatilité
        if (trade_context.volatility_at_entry > 0.025 and 
            trade_context.profit_pips > 0):
            bonus += 0.15
        
        # 5. 💎 Bonus "Diamond Hands" pour ne pas céder à la panique
        if (trade_context.max_adverse_excursion > 15 and 
            trade_context.profit_pips > 20):
            bonus += 0.25
        
        return bonus
    
    def _calculate_adaptive_multiplier(self) -> float:
        """🔄 Multiplicateur adaptatif basé sur la performance globale"""
        base_multiplier = 1.0
        
        # Performance récente
        win_rate = self.performance_metrics.get('win_rate', 0.5)
        sharpe_ratio = self.performance_metrics.get('sharpe_ratio', 0.0)
        profit_factor = self.performance_metrics.get('profit_factor', 1.0)
        
        # Bonus pour excellente performance
        if win_rate > self.performance_thresholds['excellent_winrate']:
            base_multiplier += 0.3
        elif win_rate > self.performance_thresholds['good_winrate']:
            base_multiplier += 0.15
        
        if sharpe_ratio > self.performance_thresholds['excellent_sharpe']:
            base_multiplier += 0.2
        elif sharpe_ratio > self.performance_thresholds['good_sharpe']:
            base_multiplier += 0.1
        
        # Malus pour mauvaise performance
        if win_rate < 0.4:
            base_multiplier -= 0.2
        if self.performance_metrics.get('current_drawdown', 0) > 0.1:
            base_multiplier -= 0.15
        
        return np.clip(base_multiplier, 0.5, 1.8)
    
    def _update_performance_metrics(self, trade_context: TradeContext):
        """📊 Met à jour les métriques de performance"""
        self.performance_metrics['total_trades'] += 1
        
        if trade_context.profit_pips > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        # Win rate
        total = self.performance_metrics['total_trades']
        winning = self.performance_metrics['winning_trades']
        self.performance_metrics['win_rate'] = winning / total if total > 0 else 0
        
        # Garde l'historique pour calculs plus complexes
        self.reward_history.append({
            'symbol': trade_context.symbol,
            'profit_pips': trade_context.profit_pips,
            'profit_usd': trade_context.profit_usd,
            'timestamp': trade_context.exit_time
        })
        
        # Limite l'historique
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-500:]
    
    def _generate_improvement_suggestions(self, trade_context: TradeContext, 
                                        reward_breakdown: Dict) -> List[str]:
        """💡 Génère des suggestions d'amélioration basées sur l'analyse"""
        suggestions = []
        
        # Analyse des points faibles
        if reward_breakdown['price_action'] < 0:
            suggestions.append("🎯 Améliorer l'analyse des niveaux support/résistance")
        
        if reward_breakdown['risk_management'] < 0:
            suggestions.append("⚖️ Revoir la gestion du risque et position sizing")
        
        if reward_breakdown['timing'] < -0.3:
            suggestions.append("⏰ Optimiser les points d'entrée et de sortie")
        
        if reward_breakdown['consistency'] < 0:
            suggestions.append("🎯 Réduire le nombre de trades et être plus sélectif")
        
        # Suggestions positives
        if reward_breakdown['price_action'] > 0.5:
            suggestions.append("🌟 Excellente lecture du price action - continuer!")
        
        if reward_breakdown['risk_management'] > 0.5:
            suggestions.append("🛡️ Gestion du risque exemplaire")
        
        return suggestions

# Instance globale du système de récompenses
ultra_reward_system = UltraRewardSystem()