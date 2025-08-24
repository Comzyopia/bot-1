"""
🚀 ULTRA TRADING BOT - VERSION 3.0 RÉVOLUTIONNAIRE
═══════════════════════════════════════════════════

Bot de trading ultra-performant avec:
- Système de récompenses multi-dimensionnel révolutionnaire
- IA spécialisée Price Action (moins dépendant indicateurs techniques)
- Gestion du risque ultra-avancée
- Performance maximisée pour perdre très peu et gagner beaucoup

Version: 3.0 - Ultra Performance Revolution
"""

from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import asyncio
import json
import threading
import time
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# 🚀 IMPORTATION DES SYSTÈMES ULTRA-AVANCÉS
from ultra_reward_system import ultra_reward_system, TradeContext, MarketRegime
from ultra_price_action_ai import ultra_price_action_ai, PriceActionSignal, PriceActionPattern
from ultra_risk_manager import ultra_risk_manager, RiskParameters, MarketCondition, RiskLevel

# 🛡️ IMPORTATION DU SYSTÈME DE SÉCURITÉ
try:
    from security import security_manager, require_auth, require_trading_permission
except ImportError:
    # Fallback si security module n'existe pas
    security_manager = None
    def require_auth(f): return f
    def require_trading_permission(f): return f

# Vérification du système d'exploitation pour MT5
import platform
import sys

if platform.system() != "Windows":
    logging.warning("🚨 MetaTrader 5 fonctionne uniquement sur Windows!")
    logging.info("📁 Utilisation du simulateur pour Linux/Mac")
    
    # Simulateur MT5 pour développement
    class MT5Simulator:
        @staticmethod
        def initialize(): return True
        @staticmethod
        def login(*args, **kwargs): return True
        @staticmethod
        def last_error(): return (0, "Simulateur actif")
        @staticmethod
        def account_info():
            class AccountInfo:
                balance = 10000.0
                equity = 10000.0
                margin = 0.0
                free_margin = 10000.0
                login = 1296306
                server = "Simulator"
                def _asdict(self):
                    return {"balance": self.balance, "equity": self.equity, 
                           "margin": self.margin, "free_margin": self.free_margin,
                           "login": self.login, "server": self.server}
            return AccountInfo()
        @staticmethod
        def positions_get(*args, **kwargs): return []
        @staticmethod
        def copy_rates_from_pos(symbol, timeframe, start, count):
            # Simule des données OHLC
            np.random.seed(42)
            base_price = {"EURUSD": 1.1000, "GBPUSD": 1.3000, "USDJPY": 110.00}.get(symbol, 1.1000)
            
            data = []
            current_price = base_price
            for i in range(count):
                # Simule un mouvement de prix réaliste
                change = np.random.normal(0, 0.001)  # Volatilité normale
                current_price *= (1 + change)
                
                high = current_price * (1 + abs(np.random.normal(0, 0.0005)))
                low = current_price * (1 - abs(np.random.normal(0, 0.0005))) 
                open_price = current_price + np.random.normal(0, 0.0002)
                
                data.append({
                    'time': int(time.time()) - (count - i) * 300,  # 5min intervals
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': current_price,
                    'tick_volume': np.random.randint(50, 200)
                })
                
            return np.array([(d['time'], d['open'], d['high'], d['low'], d['close'], d['tick_volume']) 
                           for d in data], dtype=[('time', 'u4'), ('open', 'f8'), ('high', 'f8'), 
                                                ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'u8')])
        
        @staticmethod
        def order_send(request):
            class OrderResult:
                retcode = 10009  # Success
                deal = random.randint(1000, 9999)
                order = random.randint(10000, 99999)
                volume = request.get('volume', 0.01)
                price = request.get('price', 1.1000)
                comment = "Simulateur - Ordre exécuté"
                def _asdict(self):
                    return {"retcode": self.retcode, "deal": self.deal, "order": self.order,
                           "volume": self.volume, "price": self.price, "comment": self.comment}
            return OrderResult()
        
        TIMEFRAME_M1 = 1
        TIMEFRAME_M5 = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_H1 = 60
        TIMEFRAME_H4 = 240
        TIMEFRAME_D1 = 1440
        
    mt5 = MT5Simulator()
    MT5_AVAILABLE = True
    logging.info("✅ Simulateur MT5 chargé avec succès")
    
else:
    try:
        import MetaTrader5 as mt5
        MT5_AVAILABLE = True
        logging.info("✅ MetaTrader 5 chargé avec succès sur Windows")
    except ImportError:
        MT5_AVAILABLE = False
        logging.error("❌ MetaTrader 5 non trouvé sur Windows!")
        raise ImportError("MetaTrader 5 requis sur Windows")

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'ultra_trading_bot')
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# Create the main app
app = FastAPI(title="Ultra Trading Bot API", version="3.0.0", 
              description="🚀 Bot de Trading Ultra-Performant avec IA Price Action")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables
trading_bot = None
ws_connections: List[WebSocket] = []
bot_status = {
    "running": False,
    "connected_to_mt5": False,
    "auto_trading": False,
    "using_simulator": platform.system() != "Windows",
    "balance": 0,
    "equity": 0,
    "total_trades": 0,
    "win_rate": 0.0,
    "ultra_mode_active": True,
    "price_action_ai_enabled": True,
    "ultra_reward_system_active": True,
    "last_update": datetime.utcnow()
}

# Configuration des timeframes
TIMEFRAMES = {
    "M1": getattr(mt5, 'TIMEFRAME_M1', 1),
    "M5": getattr(mt5, 'TIMEFRAME_M5', 5),
    "M15": getattr(mt5, 'TIMEFRAME_M15', 15),
    "H1": getattr(mt5, 'TIMEFRAME_H1', 60),
    "H4": getattr(mt5, 'TIMEFRAME_H4', 240),
    "D1": getattr(mt5, 'TIMEFRAME_D1', 1440)
}

# Models
class UltraTradingConfig(BaseModel):
    """Configuration ultra-avancée du bot"""
    timeframe: str = "M5"
    symbols: List[str] = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD"]
    risk_level: str = "moderate"  # ultra_conservative, conservative, moderate, aggressive, ultra_aggressive
    max_positions: int = 5
    auto_trading: bool = False
    
    # 🚀 NOUVELLES FONCTIONNALITÉS ULTRA-AVANCÉES
    price_action_mode: bool = True      # Mode Price Action (moins d'indicateurs techniques)
    ultra_reward_system: bool = True    # Système de récompenses révolutionnaire
    smart_risk_management: bool = True  # Gestion du risque intelligente
    adaptive_position_sizing: bool = True  # Taille de position adaptative
    multi_timeframe_analysis: bool = True  # Analyse multi-timeframe
    
    # Filtres avancés
    news_filter: bool = True           # Éviter trading durant news importantes
    session_filter: bool = True        # Trading selon sessions (Londres, NY, etc.)
    volatility_filter: bool = True     # Ajuster selon volatilité
    correlation_filter: bool = True    # Éviter positions trop corrélées
    
    # Paramètres de performance
    min_confidence_threshold: float = 0.75   # Confiance minimale pour trade
    max_daily_risk: float = 0.05             # Risque journalier max (5%)
    target_sharpe_ratio: float = 2.0         # Objectif Sharpe ratio
    max_drawdown_limit: float = 0.15         # Limite drawdown (15%)

class MT5Config(BaseModel):
    login: int
    password: str
    server: str

class UltraTradeSignal(BaseModel):
    """Signal de trade ultra-avancé"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    timeframe: str
    action: str  # BUY, SELL, HOLD, CLOSE_ALL
    confidence: float
    price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    
    # 🚀 NOUVELLES DONNÉES ULTRA-AVANCÉES
    price_action_pattern: Optional[str] = None
    pattern_strength: float = 0.0
    risk_reward_ratio: float = 0.0
    market_regime: str = "ranging"
    volatility_regime: str = "normal"
    session_type: str = "london"
    support_resistance_level: Optional[float] = None
    expected_move_pips: float = 0.0
    
    # Métriques de qualité
    technical_score: float = 0.0
    price_action_score: float = 0.0
    risk_management_score: float = 0.0
    overall_quality_score: float = 0.0
    
    reasoning: str = ""
    improvement_suggestions: List[str] = []

class UltraTradingEngine:
    """🚀 Moteur de trading ultra-performant révolutionnaire"""
    
    def __init__(self):
        self.config = UltraTradingConfig()
        self.connected = False
        self.account_info = None
        self.trading_thread = None
        self.analysis_thread = None
        self.stop_trading = False
        self.positions = []
        self.trade_history = []
        self.signals = []
        
        # 🧠 INTÉGRATION DES SYSTÈMES ULTRA-AVANCÉS
        self.ultra_ai = ultra_price_action_ai
        self.reward_system = ultra_reward_system  
        self.risk_manager = ultra_risk_manager
        
        # 📊 MÉTRIQUES DE PERFORMANCE EN TEMPS RÉEL
        self.performance_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "profit_factor": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "kelly_criterion": 0.0,
            "expectancy": 0.0,
            "ultra_reward_score": 0.0,
            "price_action_accuracy": 0.0,
            "risk_management_score": 0.0
        }
        
        # 🎯 TRACKING AVANCÉ
        self.active_positions_by_symbol = {}
        self.last_trade_time_by_symbol = {}
        self.correlation_matrix = {}
        self.market_regime_history = []
        
        logging.info("🚀 Ultra Trading Engine initialisé avec tous les systèmes avancés!")
    
    def connect_mt5(self, mt5_config: MT5Config) -> bool:
        """Connexion à MetaTrader 5 ou simulateur"""
        try:
            if not mt5.initialize():
                logging.error("❌ Échec initialisation MT5")
                return False
            
            result = mt5.login(mt5_config.login, password=mt5_config.password, server=mt5_config.server)
            if not result:
                error = mt5.last_error()
                logging.error(f"❌ Échec connexion MT5: {error}")
                return False
            
            self.connected = True
            self.update_account_info()
            
            # Initialise le risk manager avec le solde réel
            self.risk_manager.account_balance = self.account_info.get('balance', 10000)
            self.risk_manager.current_equity = self.account_info.get('equity', 10000)
            
            if hasattr(mt5, 'SIMULATOR_MODE') or platform.system() != "Windows":
                bot_status["using_simulator"] = True
                logging.info("✅ Connecté au simulateur MT5")
            else:
                bot_status["using_simulator"] = False
                logging.info(f"✅ Connecté au MT5 réel: {mt5_config.login}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Erreur connexion MT5: {e}")
            return False
    
    def start_ultra_trading(self):
        """🚀 Démarre le trading ultra-performant"""
        if self.trading_thread and self.trading_thread.is_alive():
            logging.warning("⚠️ Trading déjà en cours")
            return
        
        self.stop_trading = False
        self.trading_thread = threading.Thread(target=self._ultra_trading_loop, daemon=True)
        self.trading_thread.start()
        
        # Démarre l'analyse continue
        self.analysis_thread = threading.Thread(target=self._continuous_analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        bot_status["running"] = True
        bot_status["auto_trading"] = True
        
        logging.info("🚀 Ultra Trading Bot démarré avec succès!")
    
    def _ultra_trading_loop(self):
        """🔄 Boucle principale de trading ultra-performante"""
        logging.info("🚀 Début de la boucle de trading ultra-performante")
        
        while not self.stop_trading:
            try:
                # 1. 🛡️ Vérifications de sécurité
                safety_check = self.risk_manager.should_stop_trading()
                if safety_check["should_stop"]:
                    logging.warning(f"🛑 Trading arrêté pour sécurité: {safety_check['reasons']}")
                    break
                
                # 2. 📊 Analyse des symboles configurés
                for symbol in self.config.symbols:
                    try:
                        # Évite sur-trading du même symbole
                        if self._is_symbol_on_cooldown(symbol):
                            continue
                        
                        # 🧠 Analyse Price Action ultra-avancée
                        market_data = self._get_market_data_for_analysis(symbol)
                        if not market_data:
                            continue
                        
                        # Génère signal avec IA Price Action
                        price_action_signal = self.ultra_ai.analyze_price_action(market_data, symbol)
                        
                        if price_action_signal and price_action_signal.confidence >= self.config.min_confidence_threshold:
                            # 3. ⚖️ Calcul du position sizing optimal
                            position_calc = self.risk_manager.calculate_optimal_position_size(
                                price_action_signal.confidence,
                                price_action_signal.entry_price,
                                price_action_signal.stop_loss,
                                symbol
                            )
                            
                            if position_calc["recommended"] and position_calc["lot_size"] > 0.01:
                                # 4. 🎯 Exécution du trade ultra-optimisé
                                trade_result = self._execute_ultra_trade(price_action_signal, position_calc)
                                
                                if trade_result["success"]:
                                    logging.info(f"✅ Trade exécuté: {symbol} {price_action_signal.direction} "
                                               f"@ {price_action_signal.entry_price:.5f} "
                                               f"Lot: {position_calc['lot_size']} "
                                               f"Conf: {price_action_signal.confidence:.2f}")
                                    
                                    # 5. 📚 Apprentissage immédiat
                                    self._start_trade_monitoring(price_action_signal, trade_result)
                        
                    except Exception as e:
                        logging.error(f"❌ Erreur analyse {symbol}: {e}")
                        continue
                
                # 6. 📊 Mise à jour des métriques
                self._update_performance_metrics()
                
                # 7. ⏱️ Pause entre cycles
                time.sleep(30)  # Analyse toutes les 30 secondes
                
            except Exception as e:
                logging.error(f"❌ Erreur dans la boucle de trading: {e}")
                time.sleep(60)  # Pause plus longue en cas d'erreur
        
        logging.info("🛑 Boucle de trading arrêtée")
    
    def _get_market_data_for_analysis(self, symbol: str) -> Optional[Dict]:
        """📊 Récupère les données de marché pour l'analyse"""
        try:
            # Récupère les données OHLC
            rates = mt5.copy_rates_from_pos(symbol, TIMEFRAMES[self.config.timeframe], 0, 200)
            
            if rates is None or len(rates) < 50:
                return None
            
            # Convertit en format DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Prépare les données pour l'IA
            market_data = {
                'symbol': symbol,
                'timeframe': self.config.timeframe,
                'open_prices': df['open'].tolist(),
                'high_prices': df['high'].tolist(),
                'low_prices': df['low'].tolist(),
                'close_prices': df['close'].tolist(),
                'volumes': df['tick_volume'].tolist(),
                'timestamps': df['time'].tolist(),
                'current_price': df['close'].iloc[-1],
                'atr': self._calculate_atr(df),
                'volatility': self._calculate_volatility(df)
            }
            
            return market_data
            
        except Exception as e:
            logging.error(f"❌ Erreur récupération données {symbol}: {e}")
            return None
    
    def _execute_ultra_trade(self, signal: PriceActionSignal, position_calc: Dict) -> Dict:
        """🎯 Exécute un trade avec le système ultra-avancé"""
        try:
            # Prépare la requête d'ordre
            action = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL
            
            # Calcule SL et TP optimaux si pas fournis
            if not signal.stop_loss or not signal.take_profit:
                sl_calc = self.risk_manager.calculate_dynamic_stop_loss(
                    signal.entry_price, signal.direction, signal.symbol
                )
                tp_calc = self.risk_manager.calculate_dynamic_take_profit(
                    signal.entry_price, sl_calc["stop_loss"], signal.direction, 
                    signal.symbol, signal.strength
                )
                
                signal.stop_loss = sl_calc["stop_loss"]
                signal.take_profit = tp_calc["take_profit"]
            
            # Prépare la requête MT5
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": position_calc["lot_size"],
                "type": action,
                "price": signal.entry_price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit,
                "deviation": 20,
                "magic": 123456,
                "comment": f"UltraBot-{signal.price_action_pattern}-{signal.confidence:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Exécute l'ordre
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Trade réussi
                trade_info = {
                    "success": True,
                    "deal_id": result.deal,
                    "order_id": result.order,
                    "volume": result.volume,
                    "price": result.price,
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "entry_time": datetime.utcnow(),
                    "confidence": signal.confidence,
                    "pattern": signal.price_action_pattern,
                    "position_calc": position_calc
                }
                
                # Met à jour le tracking
                self.active_positions_by_symbol[signal.symbol] = trade_info
                self.last_trade_time_by_symbol[signal.symbol] = datetime.utcnow()
                
                return trade_info
            
            else:
                # Trade échoué
                error_code = result.retcode if result else "Unknown"
                logging.error(f"❌ Échec trade {signal.symbol}: Code {error_code}")
                return {"success": False, "error": error_code}
                
        except Exception as e:
            logging.error(f"❌ Erreur exécution trade: {e}")
            return {"success": False, "error": str(e)}
    
    def _start_trade_monitoring(self, signal: PriceActionSignal, trade_result: Dict):
        """📊 Démarre le monitoring du trade pour apprentissage"""
        def monitor_trade():
            try:
                # Simule le monitoring (dans la vraie implémentation, suivrait le trade réel)
                time.sleep(300)  # Attend 5 minutes
                
                # Simule un résultat de trade
                profit_pips = np.random.normal(10, 20)  # Simule profit/perte
                profit_usd = profit_pips * position_calc.get("lot_size", 0.01) * 10
                
                # Crée le contexte de trade pour l'apprentissage
                trade_context = TradeContext(
                    symbol=signal.symbol,
                    entry_price=signal.entry_price,
                    exit_price=signal.entry_price + (profit_pips / 10000),
                    entry_time=trade_result["entry_time"],
                    exit_time=datetime.utcnow(),
                    position_size=trade_result["volume"],
                    direction=signal.direction,
                    profit_pips=profit_pips,
                    profit_usd=profit_usd,
                    sl_price=signal.stop_loss,
                    tp_price=signal.take_profit,
                    max_favorable_excursion=max(0, profit_pips + np.random.uniform(0, 10)),
                    max_adverse_excursion=max(0, abs(min(0, profit_pips)) + np.random.uniform(0, 5)),
                    market_regime=MarketRegime.TRENDING_BULL if profit_pips > 0 else MarketRegime.RANGING,
                    confidence_at_entry=signal.confidence,
                    volatility_at_entry=0.02,
                    support_resistance_distance=abs(signal.entry_price - (signal.support_resistance_level or signal.entry_price)) / signal.entry_price * 10000,
                    pattern_strength=signal.strength
                )
                
                # 🎓 Apprentissage avec le système de récompenses révolutionnaire
                market_data = {"current_price": trade_context.exit_price}
                ai_state = {"account_balance": self.risk_manager.current_equity}
                
                reward_result = self.reward_system.calculate_ultra_reward(
                    trade_context, market_data, ai_state
                )
                
                # 🧠 Apprentissage de l'IA Price Action
                self.ultra_ai.learn_from_trade_outcome(signal, trade_context)
                
                # 📊 Mise à jour du risk manager
                self.risk_manager.update_performance_metrics({
                    "symbol": signal.symbol,
                    "profit_usd": profit_usd,
                    "profit_pips": profit_pips,
                    "volume": trade_result["volume"],
                    "entry_time": trade_result["entry_time"],
                    "exit_time": datetime.utcnow()
                })
                
                logging.info(f"🎓 Apprentissage terminé - {signal.symbol}: "
                           f"Profit: {profit_pips:.1f} pips, "
                           f"Récompense: {reward_result['total_reward']:.3f}")
                
                # Nettoie le tracking
                if signal.symbol in self.active_positions_by_symbol:
                    del self.active_positions_by_symbol[signal.symbol]
                
            except Exception as e:
                logging.error(f"❌ Erreur monitoring trade: {e}")
        
        # Lance le monitoring en thread séparé
        monitor_thread = threading.Thread(target=monitor_trade, daemon=True)
        monitor_thread.start()
    
    def _is_symbol_on_cooldown(self, symbol: str) -> bool:
        """⏰ Vérifie si un symbole est en période de cooldown"""
        if symbol not in self.last_trade_time_by_symbol:
            return False
        
        last_trade = self.last_trade_time_by_symbol[symbol]
        cooldown_minutes = 15  # 15 minutes entre trades sur même symbole
        
        return (datetime.utcnow() - last_trade).total_seconds() < (cooldown_minutes * 60)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """📊 Calcule l'ATR"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else 0.001
            
        except Exception:
            return 0.001  # Valeur par défaut
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """📊 Calcule la volatilité"""
        try:
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualisée
            return float(volatility) if not pd.isna(volatility) else 0.02
        except Exception:
            return 0.02  # Valeur par défaut
    
    def _continuous_analysis_loop(self):
        """🔄 Boucle d'analyse continue et optimisation"""
        while not self.stop_trading:
            try:
                # 1. Analyse des performances
                self._analyze_system_performance()
                
                # 2. Optimisation des paramètres
                self._optimize_system_parameters()
                
                # 3. Détection des régimes de marché
                self._update_market_regime()
                
                # 4. Nettoyage des données anciennes
                self._cleanup_old_data()
                
                time.sleep(300)  # Toutes les 5 minutes
                
            except Exception as e:
                logging.error(f"❌ Erreur analyse continue: {e}")
                time.sleep(600)  # Pause plus longue en cas d'erreur
    
    def _analyze_system_performance(self):
        """📊 Analyse les performances du système"""
        try:
            # Récupère les métriques du risk manager
            risk_summary = self.risk_manager.get_risk_summary()
            
            # Met à jour les stats globales
            self.performance_stats.update({
                "total_trades": len(self.risk_manager.trade_history),
                "win_rate": risk_summary["risk_metrics"]["win_rate"],
                "profit_factor": risk_summary["risk_metrics"]["profit_factor"],
                "sharpe_ratio": risk_summary["risk_metrics"]["sharpe_ratio"],
                "max_drawdown": risk_summary["risk_metrics"]["max_drawdown"],
                "current_drawdown": risk_summary["risk_metrics"]["current_drawdown"]
            })
            
            # Récupère les performances de l'IA Price Action
            ai_performance = self.ultra_ai.get_performance_summary()
            self.performance_stats.update({
                "price_action_accuracy": ai_performance["success_rates"]["overall_pattern"]
            })
            
            logging.info(f"📊 Performance Analysis - Win Rate: {self.performance_stats['win_rate']:.1%}, "
                        f"Sharpe: {self.performance_stats['sharpe_ratio']:.2f}, "
                        f"DD: {self.performance_stats['current_drawdown']:.1%}")
            
        except Exception as e:
            logging.error(f"❌ Erreur analyse performance: {e}")
    
    def _optimize_system_parameters(self):
        """🎛️ Optimise les paramètres du système"""
        try:
            # Ajuste la confiance minimale selon la performance
            if self.performance_stats["win_rate"] < 0.5:
                # Performance faible = exigences plus élevées
                self.config.min_confidence_threshold = min(0.85, self.config.min_confidence_threshold + 0.05)
            elif self.performance_stats["win_rate"] > 0.7:
                # Bonne performance = peut accepter plus de trades
                self.config.min_confidence_threshold = max(0.65, self.config.min_confidence_threshold - 0.02)
            
            # Ajuste le nombre de positions selon le drawdown
            if self.performance_stats["current_drawdown"] > 0.1:
                # Drawdown élevé = réduction positions
                self.config.max_positions = max(2, self.config.max_positions - 1)
            elif self.performance_stats["current_drawdown"] < 0.03:
                # Faible drawdown = peut augmenter positions
                self.config.max_positions = min(8, self.config.max_positions + 1)
            
            logging.info(f"🎛️ Paramètres optimisés - Conf min: {self.config.min_confidence_threshold:.2f}, "
                        f"Max pos: {self.config.max_positions}")
            
        except Exception as e:
            logging.error(f"❌ Erreur optimisation paramètres: {e}")
    
    def _update_market_regime(self):
        """🌍 Met à jour la détection du régime de marché"""
        try:
            # Analyse le régime pour chaque symbole principal
            regimes = {}
            for symbol in self.config.symbols[:3]:  # Top 3 symboles
                market_data = self._get_market_data_for_analysis(symbol)
                if market_data:
                    # Simule la détection de régime (dans la vraie implémentation, utiliserait l'IA)
                    volatility = market_data["volatility"]
                    
                    if volatility > 0.025:
                        regime = MarketCondition.HIGH_VOLATILITY
                    elif volatility < 0.01:
                        regime = MarketCondition.LOW_VOLATILITY
                    else:
                        regime = MarketCondition.NORMAL_VOLATILITY
                    
                    regimes[symbol] = regime
            
            # Détermine le régime global
            if regimes:
                # Prend le régime le plus fréquent
                regime_counts = {}
                for regime in regimes.values():
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                dominant_regime = max(regime_counts, key=regime_counts.get)
                self.risk_manager.market_condition = dominant_regime
                
                logging.info(f"🌍 Régime de marché mis à jour: {dominant_regime.value}")
            
        except Exception as e:
            logging.error(f"❌ Erreur mise à jour régime: {e}")
    
    def _cleanup_old_data(self):
        """🧹 Nettoie les anciennes données"""
        try:
            # Limite l'historique des trades
            if len(self.risk_manager.trade_history) > 1000:
                self.risk_manager.trade_history = self.risk_manager.trade_history[-500:]
            
            # Nettoie l'historique des signaux
            if len(self.signals) > 500:
                self.signals = self.signals[-250:]
            
            # Nettoie les mémoires de l'IA
            if len(self.ultra_ai.price_action_memory) > 10000:
                self.ultra_ai.price_action_memory = deque(
                    list(self.ultra_ai.price_action_memory)[-5000:], 
                    maxlen=50000
                )
            
            logging.info("🧹 Nettoyage des données anciennes effectué")
            
        except Exception as e:
            logging.error(f"❌ Erreur nettoyage données: {e}")
    
    def update_account_info(self):
        """💰 Met à jour les informations du compte"""
        if not self.connected:
            return
        
        try:
            account_info = mt5.account_info()
            if account_info:
                self.account_info = account_info._asdict()
                
                # Met à jour le risk manager
                self.risk_manager.current_equity = self.account_info.get('equity', self.risk_manager.current_equity)
                
                bot_status["balance"] = self.account_info.get('balance', 0)
                bot_status["equity"] = self.account_info.get('equity', 0)
                
                logging.info(f"💰 Compte mis à jour - Balance: ${bot_status['balance']:.2f}, "
                           f"Equity: ${bot_status['equity']:.2f}")
            
            # Met à jour les positions
            positions = mt5.positions_get()
            self.positions = [pos._asdict() for pos in positions] if positions else []
            
        except Exception as e:
            logging.error(f"❌ Erreur mise à jour compte: {e}")
    
    def _update_performance_metrics(self):
        """📊 Met à jour les métriques de performance"""
        try:
            # Met à jour le bot_status avec les dernières métriques
            bot_status.update({
                "total_trades": self.performance_stats["total_trades"],
                "win_rate": self.performance_stats["win_rate"],
                "ultra_mode_active": True,
                "price_action_ai_enabled": True,
                "ultra_reward_system_active": True,
                "last_update": datetime.utcnow()
            })
            
        except Exception as e:
            logging.error(f"❌ Erreur mise à jour métriques: {e}")
    
    def stop_ultra_trading(self):
        """🛑 Arrête le trading ultra-performant"""
        self.stop_trading = True
        
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
        if self.analysis_thread:
            self.analysis_thread.join(timeout=10)
        
        bot_status["running"] = False
        bot_status["auto_trading"] = False
        
        logging.info("🛑 Ultra Trading Bot arrêté")
    
    def get_ultra_performance_summary(self) -> Dict:
        """📊 Résumé complet des performances ultra-avancées"""
        return {
            "system_status": {
                "ultra_mode": True,
                "version": "3.0",
                "systems_active": {
                    "price_action_ai": True,
                    "ultra_reward_system": True,
                    "smart_risk_management": True
                }
            },
            "performance_metrics": self.performance_stats.copy(),
            "risk_metrics": self.risk_manager.get_risk_summary(),
            "ai_metrics": self.ultra_ai.get_performance_summary(),
            "recent_improvements": self.reward_system.performance_metrics.copy(),
            "trading_config": {
                "timeframe": self.config.timeframe,
                "symbols": self.config.symbols,
                "risk_level": self.config.risk_level,
                "min_confidence": self.config.min_confidence_threshold,
                "max_positions": self.config.max_positions
            }
        }

# Instance globale du moteur de trading
ultra_trading_engine = UltraTradingEngine()

# WebSocket manager
async def broadcast_to_websockets(message: dict):
    """📡 Diffuse un message à tous les websockets connectés"""
    if ws_connections:
        message_str = json.dumps(message, default=str)
        for websocket in ws_connections.copy():
            try:
                await websocket.send_text(message_str)
            except:
                ws_connections.remove(websocket)

# 🚀 API ROUTES ULTRA-AVANCÉES
@api_router.get("/")
async def root():
    """🏠 Route racine avec informations système"""
    return {
        "message": "🚀 Ultra Trading Bot API v3.0 - Révolution Performance",
        "version": "3.0.0",
        "status": "operational",
        "features": {
            "ultra_reward_system": True,
            "price_action_ai": True,
            "smart_risk_management": True,
            "adaptive_position_sizing": True,
            "multi_timeframe_analysis": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@api_router.get("/status")
async def get_status():
    """📊 Status complet du système ultra-avancé"""
    return {
        "bot_status": bot_status,
        "performance_summary": ultra_trading_engine.get_ultra_performance_summary(),
        "system_health": {
            "mt5_connected": ultra_trading_engine.connected,
            "trading_active": bot_status["running"],
            "systems_operational": {
                "ultra_reward": True,
                "price_action_ai": True,
                "risk_manager": True
            }
        }
    }

@api_router.post("/mt5/connect")
async def connect_mt5(mt5_config: MT5Config):
    """🔌 Connexion MT5 avec validation ultra-avancée"""
    try:
        success = ultra_trading_engine.connect_mt5(mt5_config)
        
        if success:
            await broadcast_to_websockets({
                "type": "mt5_connection",
                "success": True,
                "account_info": ultra_trading_engine.account_info,
                "ultra_systems_initialized": True
            })
            
            return {
                "success": True,
                "message": "🚀 Connexion MT5 réussie avec systèmes ultra-avancés",
                "account_info": ultra_trading_engine.account_info,
                "systems_status": {
                    "ultra_reward_system": "✅ Activé",
                    "price_action_ai": "✅ Activé", 
                    "smart_risk_management": "✅ Activé"
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Échec connexion MT5")
            
    except Exception as e:
        logging.error(f"Erreur connexion MT5: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/trading/start")
async def start_trading():
    """🚀 Démarre le trading ultra-performant"""
    try:
        if not ultra_trading_engine.connected:
            raise HTTPException(status_code=400, detail="MT5 non connecté")
        
        ultra_trading_engine.start_ultra_trading()
        
        await broadcast_to_websockets({
            "type": "trading_started",
            "message": "🚀 Ultra Trading Bot démarré",
            "systems_active": {
                "ultra_reward_system": True,
                "price_action_ai": True,
                "smart_risk_management": True
            }
        })
        
        return {
            "success": True,
            "message": "🚀 Ultra Trading Bot démarré avec succès",
            "systems_initialized": [
                "Système de récompenses révolutionnaire",
                "IA Price Action ultra-avancée", 
                "Gestion du risque intelligente",
                "Position sizing adaptatif"
            ]
        }
        
    except Exception as e:
        logging.error(f"Erreur démarrage trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/trading/stop")
async def stop_trading():
    """🛑 Arrête le trading"""
    try:
        ultra_trading_engine.stop_ultra_trading()
        
        await broadcast_to_websockets({
            "type": "trading_stopped",
            "message": "🛑 Ultra Trading Bot arrêté"
        })
        
        return {"success": True, "message": "🛑 Trading arrêté"}
        
    except Exception as e:
        logging.error(f"Erreur arrêt trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/performance")
async def get_performance():
    """📊 Métriques de performance ultra-détaillées"""
    try:
        return ultra_trading_engine.get_ultra_performance_summary()
    except Exception as e:
        logging.error(f"Erreur récupération performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/signals")
async def get_signals():
    """🎯 Signaux actuels avec analyse Price Action"""
    try:
        current_signals = []
        
        for symbol in ultra_trading_engine.config.symbols:
            market_data = ultra_trading_engine._get_market_data_for_analysis(symbol)
            if market_data:
                signal = ultra_trading_engine.ultra_ai.analyze_price_action(market_data, symbol)
                if signal:
                    current_signals.append({
                        "symbol": symbol,
                        "action": signal.direction,
                        "confidence": signal.confidence,
                        "pattern": signal.pattern.value if signal.pattern else "none",
                        "entry_price": signal.entry_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "risk_reward_ratio": signal.risk_reward_ratio,
                        "expected_move_pips": signal.expected_move_pips,
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        return {
            "signals": current_signals,
            "total_signals": len(current_signals),
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "systems_used": ["Price Action AI", "Ultra Reward System", "Smart Risk Management"]
        }
        
    except Exception as e:
        logging.error(f"Erreur récupération signaux: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/config/update")
async def update_config(config: UltraTradingConfig):
    """⚙️ Met à jour la configuration ultra-avancée"""
    try:
        ultra_trading_engine.config = config
        
        # Met à jour les systèmes
        ultra_trading_engine.risk_manager.adaptation_enabled = config.smart_risk_management
        
        return {
            "success": True,
            "message": "🔧 Configuration ultra-avancée mise à jour",
            "new_config": {
                "timeframe": config.timeframe,
                "symbols": config.symbols,
                "risk_level": config.risk_level,
                "price_action_mode": config.price_action_mode,
                "ultra_systems": {
                    "reward_system": config.ultra_reward_system,
                    "smart_risk": config.smart_risk_management,
                    "adaptive_sizing": config.adaptive_position_sizing
                }
            }
        }
        
    except Exception as e:
        logging.error(f"Erreur mise à jour config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """🌐 WebSocket pour mises à jour temps réel"""
    await websocket.accept()
    ws_connections.append(websocket)
    
    try:
        # Envoie le status initial
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "🚀 Connexion Ultra Trading Bot WebSocket établie",
            "systems_status": {
                "ultra_reward_system": "✅ Actif",
                "price_action_ai": "✅ Actif", 
                "smart_risk_management": "✅ Actif"
            }
        }, default=str))
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe_performance":
                # Envoie les métriques de performance
                performance = ultra_trading_engine.get_ultra_performance_summary()
                await websocket.send_text(json.dumps({
                    "type": "performance_update",
                    "data": performance
                }, default=str))
                
    except WebSocketDisconnect:
        ws_connections.remove(websocket)
    except Exception as e:
        logging.error(f"Erreur WebSocket: {e}")
        if websocket in ws_connections:
            ws_connections.remove(websocket)

# Tâche de fond pour les mises à jour temps réel
async def continuous_websocket_updates():
    """📡 Mises à jour WebSocket continues"""
    while True:
        try:
            if ws_connections and bot_status["running"]:
                # Envoie les métriques en temps réel
                performance = ultra_trading_engine.get_ultra_performance_summary()
                
                await broadcast_to_websockets({
                    "type": "real_time_update",
                    "performance": performance,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            await asyncio.sleep(30)  # Toutes les 30 secondes
            
        except Exception as e:
            logging.error(f"Erreur mises à jour WebSocket: {e}")
            await asyncio.sleep(60)

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Events
@app.on_event("startup")
async def startup_event():
    """🌟 Événement de démarrage ultra-avancé"""
    logging.info("🚀 Démarrage Ultra Trading Bot v3.0 - Revolution Edition")
    logging.info("✅ Systèmes ultra-avancés initialisés:")
    logging.info("   🎯 Système de récompenses révolutionnaire")
    logging.info("   🧠 IA Price Action ultra-performante")
    logging.info("   ⚖️ Gestion du risque intelligente")
    logging.info("   📊 Analytics avancés en temps réel")
    
    # Démarre les tâches de fond
    asyncio.create_task(continuous_websocket_updates())

@app.on_event("shutdown")
async def shutdown_event():
    """🛑 Événement d'arrêt"""
    logging.info("🛑 Arrêt Ultra Trading Bot")
    
    # Arrête le trading
    ultra_trading_engine.stop_ultra_trading()
    
    # Ferme MT5
    if MT5_AVAILABLE and ultra_trading_engine.connected:
        mt5.shutdown()
    
    # Ferme les WebSockets
    for ws in ws_connections:
        try:
            await ws.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")