"""
Bot de Trading Ultra-Performant - Version Corrigée
Avec sélection de timeframes et trading automatique
"""

from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import asyncio
import json
import logging
import time
import threading
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil

# Imports locaux
from config import config_manager, Timeframe, TradingMode, RiskLevel
from indicators import TechnicalIndicators

# Import MetaTrader 5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logging.info("MetaTrader 5 library loaded successfully")
except ImportError:
    MT5_AVAILABLE = False
    logging.error("MetaTrader 5 library not found")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create the main app
app = FastAPI(
    title="Ultra Trading Bot API - Professional Edition",
    description="Bot de trading algorithmique ultra-performant avec IA",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Create router
api_router = APIRouter(prefix="/api/v2")

# Global variables
trading_bot = None
ws_connections: List[WebSocket] = []
system_stats = {
    "cpu_usage": 0,
    "memory_usage": 0,
    "uptime": 0,
    "last_update": datetime.utcnow()
}

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=4)

# Models
class AdvancedTradingConfig(BaseModel):
    """Configuration avancée de trading"""
    timeframe: str = Field(default="M5", description="Timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)")
    symbols: List[str] = Field(default=["EURUSD", "GBPUSD", "USDJPY"])
    risk_level: str = Field(default="moderate", description="conservative, moderate, aggressive")
    max_positions: int = Field(default=5, ge=1, le=20)
    use_ml: bool = Field(default=True)
    use_indicators: bool = Field(default=True)
    use_price_action: bool = Field(default=True)
    multi_timeframe: bool = Field(default=True)
    trading_hours: Dict[str, str] = Field(default={"start": "00:00", "end": "23:59"})

class SignalRequest(BaseModel):
    """Requête pour génération de signal"""
    symbol: str
    timeframe: str = "M5"
    use_ml: bool = True
    force_analysis: bool = False

class BacktestRequest(BaseModel):
    """Requête pour backtesting"""
    symbol: str
    timeframe: str = "M5"
    start_date: str
    end_date: str
    initial_balance: float = 10000
    strategy_params: Dict = Field(default={})

class NotificationSettings(BaseModel):
    """Paramètres de notification"""
    email_enabled: bool = False
    email_address: str = ""
    telegram_enabled: bool = False
    telegram_chat_id: str = ""
    webhook_url: str = ""

class PerformanceMetrics(BaseModel):
    """Métriques de performance"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0

class AdvancedMarketAnalysis:
    """Analyse de marché avancée"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.cache = {}
        self.cache_ttl = 60  # Cache TTL en secondes
    
    def get_market_data(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """Récupère les données de marché avec cache"""
        cache_key = f"{symbol}_{timeframe}_{count}"
        
        # Vérifie le cache
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return data
        
        # Récupère les nouvelles données
        if not MT5_AVAILABLE:
            return pd.DataFrame()
        
        try:
            mt5_timeframe = config_manager.get_timeframe_mt5(timeframe)
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None:
                logger.error(f"Failed to get rates for {symbol}: {mt5.last_error()}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Met en cache
            self.cache[cache_key] = (df, datetime.now())
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_symbol(self, symbol: str, timeframe: str = "M5", 
                      use_ml: bool = True, multi_timeframe: bool = True) -> Dict:
        """Analyse complète d'un symbole"""
        try:
            # Données principales
            df = self.get_market_data(symbol, timeframe)
            if df.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Calcul des indicateurs
            indicators = self.indicators.calculate_all_indicators(df)
            
            # Analyse multi-timeframe
            mtf_analysis = {}
            if multi_timeframe:
                mtf_analysis = self._multi_timeframe_analysis(symbol, timeframe)
            
            # Analyse de force du signal
            current_price = df['close'].iloc[-1]
            signal_strength = self.indicators.get_signal_strength(indicators, current_price)
            
            # Détection de divergences
            divergences = self.indicators.detect_divergences(df, indicators)
            
            # Analyse des volumes
            volume_analysis = self._analyze_volume(df)
            
            # Détection de patterns
            patterns = self._detect_advanced_patterns(df, indicators)
            
            # Calcul du score de trading
            trading_score = self._calculate_trading_score(
                signal_strength, divergences, volume_analysis, patterns, mtf_analysis
            )
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat(),
                "current_price": float(current_price),
                "signal_strength": signal_strength,
                "divergences": divergences,
                "volume_analysis": volume_analysis,
                "patterns": patterns,
                "multi_timeframe": mtf_analysis,
                "trading_score": trading_score,
                "indicators": self._format_indicators(indicators),
                "recommendation": self._generate_recommendation(trading_score, signal_strength)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"error": f"Analysis failed for {symbol}: {str(e)}"}
    
    def _multi_timeframe_analysis(self, symbol: str, base_timeframe: str) -> Dict:
        """Analyse multi-timeframe"""
        timeframes = ["M5", "M15", "H1", "H4", "D1"]
        if base_timeframe not in timeframes:
            timeframes.append(base_timeframe)
        
        mtf_data = {}
        
        for tf in timeframes:
            try:
                df = self.get_market_data(symbol, tf, 200)
                if not df.empty:
                    indicators = self.indicators.calculate_all_indicators(df)
                    current_price = df['close'].iloc[-1]
                    signal = self.indicators.get_signal_strength(indicators, current_price)
                    
                    mtf_data[tf] = {
                        "direction": signal["direction"],
                        "strength": signal["strength"],
                        "trend": self._determine_trend(indicators)
                    }
            except Exception as e:
                logger.error(f"Error in MTF analysis for {tf}: {e}")
        
        return mtf_data
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyse avancée des volumes"""
        if 'tick_volume' not in df.columns:
            return {"error": "No volume data available"}
        
        volume = df['tick_volume']
        price = df['close']
        
        # Volume moyen
        avg_volume = volume.rolling(20).mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
        
        # Volume-Price Trend
        vpt = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i-1] != 0:
                vpt += volume.iloc[i] * (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
        
        # Classification du volume
        volume_classification = "normal"
        if volume_ratio > 2:
            volume_classification = "very_high"
        elif volume_ratio > 1.5:
            volume_classification = "high"
        elif volume_ratio < 0.5:
            volume_classification = "low"
        
        return {
            "current_volume": float(current_volume),
            "average_volume": float(avg_volume.iloc[-1]),
            "volume_ratio": float(volume_ratio),
            "volume_classification": volume_classification,
            "vpt": float(vpt),
            "volume_trend": "increasing" if volume.iloc[-3:].mean() > volume.iloc[-10:-3].mean() else "decreasing"
        }
    
    def _detect_advanced_patterns(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Détection de patterns avancés"""
        patterns = {}
        
        try:
            # Patterns de chandeliers
            candlestick_patterns = []
            for pattern_name, pattern_data in indicators.items():
                if pattern_name.startswith(('doji', 'hammer', 'engulfing', 'star')):
                    if isinstance(pattern_data, np.ndarray) and len(pattern_data) > 0:
                        if pattern_data[-1] != 0:
                            candlestick_patterns.append({
                                "name": pattern_name,
                                "strength": abs(pattern_data[-1]) / 100,
                                "direction": "bullish" if pattern_data[-1] > 0 else "bearish"
                            })
            
            patterns["candlestick"] = candlestick_patterns
            
            # Patterns de prix
            price_patterns = self._detect_price_patterns(df)
            patterns["price"] = price_patterns
            
            # Patterns de structure de marché
            market_structure = self._analyze_market_structure(df)
            patterns["market_structure"] = market_structure
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
        
        return patterns
    
    def _detect_price_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Détecte les patterns de prix"""
        patterns = []
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Double top/bottom
            if len(close) >= 50:
                peaks = []
                valleys = []
                
                for i in range(5, len(close) - 5):
                    if all(high[i] > high[i-j] for j in range(1, 6)) and all(high[i] > high[i+j] for j in range(1, 6)):
                        peaks.append((i, high[i]))
                    if all(low[i] < low[i-j] for j in range(1, 6)) and all(low[i] < low[i+j] for j in range(1, 6)):
                        valleys.append((i, low[i]))
                
                # Double top
                if len(peaks) >= 2:
                    last_peaks = peaks[-2:]
                    if abs(last_peaks[0][1] - last_peaks[1][1]) / last_peaks[0][1] < 0.02:
                        patterns.append({
                            "name": "double_top",
                            "strength": 0.8,
                            "direction": "bearish"
                        })
                
                # Double bottom
                if len(valleys) >= 2:
                    last_valleys = valleys[-2:]
                    if abs(last_valleys[0][1] - last_valleys[1][1]) / last_valleys[0][1] < 0.02:
                        patterns.append({
                            "name": "double_bottom",
                            "strength": 0.8,
                            "direction": "bullish"
                        })
            
            # Triangle patterns
            triangle_pattern = self._detect_triangle_pattern(df)
            if triangle_pattern:
                patterns.append(triangle_pattern)
            
        except Exception as e:
            logger.error(f"Error detecting price patterns: {e}")
        
        return patterns
    
    def _detect_triangle_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """Détecte les patterns triangulaires"""
        if len(df) < 30:
            return None
        
        try:
            high = df['high'].values
            low = df['low'].values
            
            # Trouve les points hauts et bas
            recent_highs = []
            recent_lows = []
            
            for i in range(10, len(high) - 10):
                if all(high[i] >= high[i-j] for j in range(1, 11)) and all(high[i] >= high[i+j] for j in range(1, 11)):
                    recent_highs.append((i, high[i]))
                if all(low[i] <= low[i-j] for j in range(1, 11)) and all(low[i] <= low[i+j] for j in range(1, 11)):
                    recent_lows.append((i, low[i]))
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                # Calcule les pentes
                highs_slope = (recent_highs[-1][1] - recent_highs[0][1]) / (recent_highs[-1][0] - recent_highs[0][0])
                lows_slope = (recent_lows[-1][1] - recent_lows[0][1]) / (recent_lows[-1][0] - recent_lows[0][0])
                
                # Triangle ascendant
                if abs(highs_slope) < 0.0001 and lows_slope > 0.0001:
                    return {
                        "name": "ascending_triangle",
                        "strength": 0.7,
                        "direction": "bullish"
                    }
                
                # Triangle descendant
                if abs(lows_slope) < 0.0001 and highs_slope < -0.0001:
                    return {
                        "name": "descending_triangle",
                        "strength": 0.7,
                        "direction": "bearish"
                    }
                
                # Triangle symétrique
                if highs_slope < -0.0001 and lows_slope > 0.0001:
                    return {
                        "name": "symmetric_triangle",
                        "strength": 0.6,
                        "direction": "neutral"
                    }
        
        except Exception as e:
            logger.error(f"Error detecting triangle pattern: {e}")
        
        return None
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyse la structure de marché"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Trend analysis
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
            sma_200 = np.mean(close[-200:]) if len(close) >= 200 else sma_50
            
            # Higher highs, higher lows
            recent_highs = [high[i] for i in range(-10, 0)]
            recent_lows = [low[i] for i in range(-10, 0)]
            
            higher_highs = all(recent_highs[i] >= recent_highs[i-1] for i in range(1, len(recent_highs)))
            higher_lows = all(recent_lows[i] >= recent_lows[i-1] for i in range(1, len(recent_lows)))
            lower_highs = all(recent_highs[i] <= recent_highs[i-1] for i in range(1, len(recent_highs)))
            lower_lows = all(recent_lows[i] <= recent_lows[i-1] for i in range(1, len(recent_lows)))
            
            # Détermine la structure
            if higher_highs and higher_lows:
                structure = "uptrend"
            elif lower_highs and lower_lows:
                structure = "downtrend"
            else:
                structure = "sideways"
            
            # Force de la tendance
            trend_strength = abs(close[-1] - close[-20]) / close[-20] * 100
            
            return {
                "structure": structure,
                "trend_strength": float(trend_strength),
                "sma_alignment": sma_20 > sma_50 > sma_200,
                "higher_highs": higher_highs,
                "higher_lows": higher_lows,
                "lower_highs": lower_highs,
                "lower_lows": lower_lows
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return {"structure": "unknown", "trend_strength": 0}
    
    def _calculate_trading_score(self, signal_strength: Dict, divergences: List,
                               volume_analysis: Dict, patterns: Dict, mtf_analysis: Dict) -> Dict:
        """Calcule un score de trading global"""
        try:
            total_score = 0
            max_score = 0
            details = []
            
            # Score de force du signal (40%)
            signal_score = signal_strength["strength"] * 0.4
            if signal_strength["direction"] != "NEUTRAL":
                total_score += signal_score
            max_score += 0.4
            details.append(f"Signal strength: {signal_score:.2f}")
            
            # Score des divergences (20%)
            if divergences:
                div_score = len(divergences) * 0.1
                total_score += min(div_score, 0.2)
                details.append(f"Divergences: {div_score:.2f}")
            max_score += 0.2
            
            # Score des volumes (15%)
            if "volume_ratio" in volume_analysis:
                vol_ratio = volume_analysis["volume_ratio"]
                if vol_ratio > 1.5:
                    vol_score = 0.15
                elif vol_ratio > 1.2:
                    vol_score = 0.1
                else:
                    vol_score = 0.05
                total_score += vol_score
                details.append(f"Volume: {vol_score:.2f}")
            max_score += 0.15
            
            # Score des patterns (15%)
            pattern_score = 0
            for pattern_type, pattern_list in patterns.items():
                if isinstance(pattern_list, list):
                    for pattern in pattern_list:
                        if "strength" in pattern:
                            pattern_score += pattern["strength"] * 0.05
            total_score += min(pattern_score, 0.15)
            max_score += 0.15
            details.append(f"Patterns: {pattern_score:.2f}")
            
            # Score multi-timeframe (10%)
            if mtf_analysis:
                mtf_score = 0
                for tf, data in mtf_analysis.items():
                    if data["direction"] == signal_strength["direction"]:
                        mtf_score += data["strength"] * 0.02
                total_score += min(mtf_score, 0.1)
                details.append(f"Multi-timeframe: {mtf_score:.2f}")
            max_score += 0.1
            
            # Normalisation
            final_score = total_score / max_score if max_score > 0 else 0
            
            return {
                "score": float(final_score),
                "max_score": float(max_score),
                "details": details,
                "grade": self._get_grade(final_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading score: {e}")
            return {"score": 0, "max_score": 1, "details": [], "grade": "F"}
    
    def _get_grade(self, score: float) -> str:
        """Attribue une note au score"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C"
        elif score >= 0.4:
            return "D"
        else:
            return "F"
    
    def _format_indicators(self, indicators: Dict) -> Dict:
        """Formate les indicateurs pour l'API"""
        formatted = {}
        
        for key, value in indicators.items():
            if isinstance(value, np.ndarray):
                if not np.isnan(value[-1]):
                    formatted[key] = float(value[-1])
            elif isinstance(value, (int, float)) and not np.isnan(value):
                formatted[key] = float(value)
        
        return formatted
    
    def _determine_trend(self, indicators: Dict) -> str:
        """Détermine la tendance basée sur les indicateurs"""
        try:
            sma_20 = indicators.get('sma_20')
            sma_50 = indicators.get('sma_50')
            
            if sma_20 is not None and sma_50 is not None:
                if not np.isnan(sma_20[-1]) and not np.isnan(sma_50[-1]):
                    if sma_20[-1] > sma_50[-1]:
                        return "bullish"
                    else:
                        return "bearish"
            
            return "neutral"
            
        except Exception:
            return "neutral"
    
    def _generate_recommendation(self, trading_score: Dict, signal_strength: Dict) -> Dict:
        """Génère une recommandation de trading"""
        score = trading_score["score"]
        direction = signal_strength["direction"]
        strength = signal_strength["strength"]
        
        if score >= 0.8 and strength >= 0.7:
            if direction == "BULLISH":
                action = "STRONG_BUY"
                confidence = 0.9
            elif direction == "BEARISH":
                action = "STRONG_SELL"
                confidence = 0.9
            else:
                action = "HOLD"
                confidence = 0.5
        elif score >= 0.6 and strength >= 0.5:
            if direction == "BULLISH":
                action = "BUY"
                confidence = 0.7
            elif direction == "BEARISH":
                action = "SELL"
                confidence = 0.7
            else:
                action = "HOLD"
                confidence = 0.5
        else:
            action = "HOLD"
            confidence = 0.3
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": f"Score: {score:.2f}, Direction: {direction}, Strength: {strength:.2f}"
        }

# Instance globale
market_analyzer = AdvancedMarketAnalysis()

# System monitoring
def update_system_stats():
    """Met à jour les statistiques système"""
    global system_stats
    system_stats.update({
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "uptime": time.time() - psutil.boot_time(),
        "last_update": datetime.utcnow()
    })

# WebSocket manager
async def broadcast_to_websockets(message: dict):
    """Diffuse un message à tous les websockets connectés"""
    if ws_connections:
        message_str = json.dumps(message, default=str)
        for websocket in ws_connections.copy():
            try:
                await websocket.send_text(message_str)
            except:
                ws_connections.remove(websocket)

# API Routes
@api_router.get("/health")
async def health_check():
    """Vérification de l'état de santé"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "mt5_available": MT5_AVAILABLE,
        "system_stats": system_stats
    }

@api_router.get("/config")
async def get_config():
    """Récupère la configuration actuelle"""
    return {
        "config": config_manager.config,
        "timeframes": [tf.value for tf in Timeframe],
        "risk_levels": [rl.value for rl in RiskLevel],
        "trading_modes": [tm.value for tm in TradingMode]
    }

@api_router.post("/config")
async def update_config(config: AdvancedTradingConfig):
    """Met à jour la configuration"""
    try:
        config_manager.set("trading.timeframe", config.timeframe)
        config_manager.set("trading.symbols", config.symbols)
        config_manager.set("trading.risk_level", config.risk_level)
        config_manager.set("trading.max_positions", config.max_positions)
        config_manager.set("analysis.use_ml", config.use_ml)
        config_manager.set("analysis.use_indicators", config.use_indicators)
        config_manager.set("analysis.use_price_action", config.use_price_action)
        config_manager.set("analysis.combine_timeframes", config.multi_timeframe)
        config_manager.set("trading.trading_hours", config.trading_hours)
        
        return {"success": True, "message": "Configuration mise à jour"}
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analysis/symbol")
async def analyze_symbol(request: SignalRequest):
    """Analyse un symbole spécifique"""
    try:
        analysis = market_analyzer.analyze_symbol(
            request.symbol,
            request.timeframe,
            request.use_ml,
            config_manager.get("analysis.combine_timeframes", True)
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing symbol {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/analysis/all")
async def analyze_all_symbols():
    """Analyse tous les symboles configurés"""
    try:
        symbols = config_manager.get("trading.symbols", ["EURUSD", "GBPUSD", "USDJPY"])
        timeframe = config_manager.get("trading.timeframe", "M5")
        
        analyses = []
        for symbol in symbols:
            analysis = market_analyzer.analyze_symbol(symbol, timeframe)
            if "error" not in analysis:
                analyses.append(analysis)
        
        return {"analyses": analyses, "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Error analyzing all symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/symbols/categories")
async def get_symbol_categories():
    """Récupère les catégories de symboles"""
    from config import SYMBOL_CATEGORIES
    return {"categories": SYMBOL_CATEGORIES}

@api_router.get("/indicators/config")
async def get_indicators_config():
    """Récupère la configuration des indicateurs"""
    from config import TECHNICAL_INDICATORS
    return {"indicators": TECHNICAL_INDICATORS}

@api_router.post("/mt5/connect")
async def connect_mt5(credentials: Dict[str, Any]):
    """Connexion à MetaTrader 5"""
    if not MT5_AVAILABLE:
        raise HTTPException(status_code=400, detail="MetaTrader 5 not available")
    
    try:
        # Initialize MT5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 initialization failed")
        
        # Login
        login_result = mt5.login(
            credentials["login"],
            password=credentials["password"],
            server=credentials["server"]
        )
        
        if not login_result:
            error_code = mt5.last_error()
            raise HTTPException(status_code=401, detail=f"Login failed: {error_code}")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            account_data = account_info._asdict()
        else:
            account_data = {}
        
        await broadcast_to_websockets({
            "type": "mt5_connection",
            "success": True,
            "account_info": account_data
        })
        
        return {
            "success": True,
            "message": "Connected to MT5",
            "account_info": account_data
        }
        
    except Exception as e:
        logger.error(f"MT5 connection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/mt5/disconnect")
async def disconnect_mt5():
    """Déconnexion de MetaTrader 5"""
    if MT5_AVAILABLE:
        mt5.shutdown()
    
    await broadcast_to_websockets({
        "type": "mt5_disconnection",
        "message": "Disconnected from MT5"
    })
    
    return {"success": True, "message": "Disconnected from MT5"}

@api_router.get("/market/data/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "M5", count: int = 100):
    """Récupère les données de marché"""
    try:
        df = market_analyzer.get_market_data(symbol, timeframe, count)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Convertit en format JSON
        data = df.reset_index().to_dict('records')
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": len(data),
            "data": data
        }
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint pour les mises à jour en temps réel"""
    await websocket.accept()
    ws_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Traite les messages entrants si nécessaire
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                # Abonnement à des mises à jour spécifiques
                symbols = message.get("symbols", [])
                # Logique d'abonnement
                
            elif message.get("type") == "unsubscribe":
                # Désabonnement
                pass
                
    except WebSocketDisconnect:
        ws_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in ws_connections:
            ws_connections.remove(websocket)

# Background tasks
async def continuous_market_analysis():
    """Analyse continue du marché"""
    while True:
        try:
            if config_manager.get("trading.mode") != "backtest":
                symbols = config_manager.get("trading.symbols", ["EURUSD"])
                timeframe = config_manager.get("trading.timeframe", "M5")
                
                analyses = []
                for symbol in symbols:
                    analysis = market_analyzer.analyze_symbol(symbol, timeframe)
                    if "error" not in analysis:
                        analyses.append(analysis)
                
                # Diffuse les analyses
                await broadcast_to_websockets({
                    "type": "market_analysis",
                    "analyses": analyses,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Met à jour les stats système
                update_system_stats()
                
                await broadcast_to_websockets({
                    "type": "system_stats",
                    "stats": system_stats
                })
            
            await asyncio.sleep(30)  # Analyse toutes les 30 secondes
            
        except Exception as e:
            logger.error(f"Error in continuous analysis: {e}")
            await asyncio.sleep(60)  # Attendre plus longtemps en cas d'erreur

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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Événement de démarrage"""
    logger.info("Starting Ultra Trading Bot - Professional Edition")
    
    # Vérifie la configuration
    if not config_manager.validate_config():
        logger.warning("Configuration validation failed, using defaults")
    
    # Démarre les tâches de fond
    asyncio.create_task(continuous_market_analysis())
    
    # Initialise les stats système
    update_system_stats()

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt"""
    logger.info("Shutting down Ultra Trading Bot")
    
    # Ferme les connexions MT5
    if MT5_AVAILABLE:
        mt5.shutdown()
    
    # Ferme les connexions WebSocket
    for ws in ws_connections:
        try:
            await ws.close()
        except:
            pass

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Main HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Page principale"""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")