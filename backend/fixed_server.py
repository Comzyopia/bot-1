"""
Bot de Trading Ultra-Performant - Version Corrigée
Avec sélection de timeframes et trading automatique
"""

from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

# Import MetaTrader 5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logging.info("MetaTrader 5 library loaded successfully")
except ImportError:
    MT5_AVAILABLE = False
    logging.error("MetaTrader 5 library not found")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the main app
app = FastAPI(title="Ultra Trading Bot", version="2.0.0")
api_router = APIRouter(prefix="/api")

# Global variables
trading_bot = None
ws_connections: List[WebSocket] = []
bot_status = {
    "running": False,
    "connected_to_mt5": False,
    "auto_trading": False,
    "balance": 0,
    "equity": 0,
    "total_trades": 0,
    "win_rate": 0.0
}

# Timeframes configuration
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
    "M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
    "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    "M30": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
    "H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
    "H4": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
    "D1": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
    "W1": mt5.TIMEFRAME_W1 if MT5_AVAILABLE else 10080,
    "MN1": mt5.TIMEFRAME_MN1 if MT5_AVAILABLE else 43200
}

# Models
class TradingConfig(BaseModel):
    timeframe: str = "M5"
    symbols: List[str] = ["EURUSD", "GBPUSD", "USDJPY"]
    risk_per_trade: float = 0.02
    max_positions: int = 5
    auto_trading: bool = False
    take_trades: bool = True

class MT5Config(BaseModel):
    login: int
    password: str
    server: str

class TradeSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str
    timeframe: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    sl: Optional[float] = None
    tp: Optional[float] = None
    reasoning: str

# Enhanced Trading Bot Class
class UltraTradingBot:
    def __init__(self):
        self.config = TradingConfig()
        self.connected = False
        self.account_info = None
        self.trading_thread = None
        self.stop_trading = False
        self.positions = []
        self.trade_history = []
        
    def connect_mt5(self, mt5_config: MT5Config) -> bool:
        """Connect to MetaTrader 5"""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader 5 not available")
            return False
        
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Login to account
            result = mt5.login(mt5_config.login, password=mt5_config.password, server=mt5_config.server)
            if not result:
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                return False
            
            self.connected = True
            self.update_account_info()
            logger.info(f"Connected to MT5 account: {mt5_config.login}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect_mt5(self):
        """Disconnect from MetaTrader 5"""
        self.stop_trading = True
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
        
        self.connected = False
        logger.info("Disconnected from MT5")
    
    def update_account_info(self):
        """Update account information"""
        if not self.connected or not MT5_AVAILABLE:
            return
        
        try:
            account_info = mt5.account_info()
            if account_info:
                self.account_info = account_info._asdict()
            
            # Update positions
            positions = mt5.positions_get()
            self.positions = [pos._asdict() for pos in positions] if positions else []
            
        except Exception as e:
            logger.error(f"Error updating account info: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get market data for analysis"""
        if not self.connected or not MT5_AVAILABLE:
            return pd.DataFrame()
        
        try:
            mt5_timeframe = TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_M5)
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None:
                logger.error(f"Failed to get rates for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def analyze_symbol(self, symbol: str, timeframe: str) -> Dict:
        """Analyze symbol for trading signals"""
        df = self.get_market_data(symbol, timeframe, 200)
        if df.empty:
            return {"error": f"No data for {symbol}"}
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            
            # Current values
            current_sma20 = df['sma_20'].iloc[-1]
            current_sma50 = df['sma_50'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_macd = macd.iloc[-1]
            current_macd_signal = macd_signal.iloc[-1]
            
            # Generate signal
            signal_strength = 0
            reasoning = []
            
            # Trend analysis
            if current_price > current_sma20 > current_sma50:
                signal_strength += 2
                reasoning.append("Bullish trend (price > SMA20 > SMA50)")
            elif current_price < current_sma20 < current_sma50:
                signal_strength -= 2
                reasoning.append("Bearish trend (price < SMA20 < SMA50)")
            
            # RSI analysis
            if current_rsi < 30:
                signal_strength += 1
                reasoning.append(f"RSI oversold ({current_rsi:.1f})")
            elif current_rsi > 70:
                signal_strength -= 1
                reasoning.append(f"RSI overbought ({current_rsi:.1f})")
            
            # MACD analysis
            if current_macd > current_macd_signal:
                signal_strength += 1
                reasoning.append("MACD bullish")
            else:
                signal_strength -= 1
                reasoning.append("MACD bearish")
            
            # Determine action
            if signal_strength >= 3:
                action = "STRONG_BUY"
                confidence = 0.9
            elif signal_strength >= 2:
                action = "BUY"
                confidence = 0.75
            elif signal_strength <= -3:
                action = "STRONG_SELL"
                confidence = 0.9
            elif signal_strength <= -2:
                action = "SELL"
                confidence = 0.75
            else:
                action = "HOLD"
                confidence = 0.5
            
            # Calculate SL/TP
            volatility = df['close'].pct_change().std() * 2
            if action in ["BUY", "STRONG_BUY"]:
                sl = current_price * (1 - volatility)
                tp = current_price * (1 + volatility * 2)
            elif action in ["SELL", "STRONG_SELL"]:
                sl = current_price * (1 + volatility)
                tp = current_price * (1 - volatility * 2)
            else:
                sl = None
                tp = None
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": current_price,
                "action": action,
                "confidence": confidence,
                "sl": sl,
                "tp": tp,
                "reasoning": " | ".join(reasoning),
                "indicators": {
                    "sma_20": current_sma20,
                    "sma_50": current_sma50,
                    "rsi": current_rsi,
                    "macd": current_macd,
                    "signal_strength": signal_strength
                },
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def execute_trade(self, signal: Dict) -> Dict:
        """Execute a trade based on signal"""
        if not self.connected or not MT5_AVAILABLE or not self.config.take_trades:
            return {"success": False, "message": "Trading disabled or not connected"}
        
        if signal["action"] in ["HOLD"]:
            return {"success": False, "message": "No trade action required"}
        
        try:
            symbol = signal["symbol"]
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {"success": False, "message": f"Symbol {symbol} not found"}
            
            # Calculate lot size based on risk
            balance = self.account_info.get("balance", 10000)
            risk_amount = balance * self.config.risk_per_trade
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return {"success": False, "message": "Failed to get current price"}
            
            # Determine order type and price
            if signal["action"] in ["BUY", "STRONG_BUY"]:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:  # SELL, STRONG_SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            
            # Calculate lot size
            if signal["sl"]:
                sl_distance = abs(price - signal["sl"])
                pip_value = symbol_info.trade_tick_value
                lot_size = min(0.1, risk_amount / (sl_distance * pip_value * 100000))
            else:
                lot_size = 0.01
            
            # Round lot size to valid increment
            lot_size = round(lot_size, 2)
            lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": f"UltraBot_{signal['timeframe']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add SL/TP if provided
            if signal["sl"]:
                request["sl"] = round(signal["sl"], symbol_info.digits)
            if signal["tp"]:
                request["tp"] = round(signal["tp"], symbol_info.digits)
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                trade_info = {
                    "success": True,
                    "message": f"Trade executed: {signal['action']} {lot_size} {symbol}",
                    "order_id": result.order,
                    "price": result.price,
                    "volume": result.volume,
                    "symbol": symbol,
                    "action": signal["action"],
                    "timestamp": datetime.utcnow()
                }
                
                self.trade_history.append(trade_info)
                logger.info(f"Trade executed: {trade_info['message']}")
                return trade_info
            else:
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                logger.error(error_msg)
                return {"success": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"Error executing trade: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def start_auto_trading(self):
        """Start automatic trading"""
        if self.trading_thread and self.trading_thread.is_alive():
            return
        
        self.stop_trading = False
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.start()
        logger.info("Auto trading started")
    
    def stop_auto_trading(self):
        """Stop automatic trading"""
        self.stop_trading = True
        if self.trading_thread:
            self.trading_thread.join(timeout=5)
        logger.info("Auto trading stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        while not self.stop_trading and self.connected:
            try:
                for symbol in self.config.symbols:
                    if self.stop_trading:
                        break
                    
                    # Check if we have too many positions for this symbol
                    symbol_positions = [p for p in self.positions if p["symbol"] == symbol]
                    if len(symbol_positions) >= 2:  # Max 2 positions per symbol
                        continue
                    
                    # Analyze symbol
                    analysis = self.analyze_symbol(symbol, self.config.timeframe)
                    if "error" in analysis:
                        continue
                    
                    # Only trade strong signals
                    if analysis["confidence"] >= 0.75 and analysis["action"] in ["STRONG_BUY", "STRONG_SELL", "BUY", "SELL"]:
                        # Execute trade
                        trade_result = self.execute_trade(analysis)
                        if trade_result["success"]:
                            # Broadcast trade notification
                            asyncio.create_task(broadcast_to_websockets({
                                "type": "trade_executed",
                                "trade": trade_result,
                                "analysis": analysis
                            }))
                        
                        # Wait a bit between trades
                        time.sleep(5)
                
                # Update account info
                self.update_account_info()
                
                # Wait before next cycle
                time.sleep(30)  # Analyze every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait longer on error

# Global trading bot instance
trading_bot = UltraTradingBot()

# WebSocket manager
async def broadcast_to_websockets(message: dict):
    """Broadcast message to all connected websockets"""
    if ws_connections:
        message_str = json.dumps(message, default=str)
        for websocket in ws_connections.copy():
            try:
                await websocket.send_text(message_str)
            except:
                if websocket in ws_connections:
                    ws_connections.remove(websocket)

# API Routes
@api_router.get("/")
async def root():
    return {
        "message": "Ultra Trading Bot API",
        "version": "2.0.0",
        "mt5_available": MT5_AVAILABLE,
        "timeframes": list(TIMEFRAMES.keys())
    }

@api_router.get("/timeframes")
async def get_timeframes():
    """Get available timeframes"""
    return {
        "timeframes": [
            {"value": "M1", "label": "1 Minute", "description": "Scalping ultra-rapide"},
            {"value": "M5", "label": "5 Minutes", "description": "Scalping et day trading"},
            {"value": "M15", "label": "15 Minutes", "description": "Day trading"},
            {"value": "M30", "label": "30 Minutes", "description": "Day trading et swing"},
            {"value": "H1", "label": "1 Heure", "description": "Swing trading"},
            {"value": "H4", "label": "4 Heures", "description": "Swing et position trading"},
            {"value": "D1", "label": "1 Jour", "description": "Position trading"},
            {"value": "W1", "label": "1 Semaine", "description": "Trading long terme"},
            {"value": "MN1", "label": "1 Mois", "description": "Investissement long terme"}
        ]
    }

@api_router.post("/mt5/connect")
async def connect_mt5(config: MT5Config):
    """Connect to MetaTrader 5"""
    global bot_status
    
    success = trading_bot.connect_mt5(config)
    bot_status["connected_to_mt5"] = success
    
    if success:
        bot_status["balance"] = trading_bot.account_info.get("balance", 0)
        bot_status["equity"] = trading_bot.account_info.get("equity", 0)
        
        await broadcast_to_websockets({
            "type": "mt5_connection",
            "success": True,
            "account_info": trading_bot.account_info
        })
        
        return {"success": True, "message": "Connected to MT5", "account_info": trading_bot.account_info}
    else:
        await broadcast_to_websockets({
            "type": "mt5_connection",
            "success": False,
            "message": "Failed to connect to MT5"
        })
        return {"success": False, "message": "Failed to connect to MT5"}

@api_router.post("/mt5/disconnect")
async def disconnect_mt5():
    """Disconnect from MetaTrader 5"""
    global bot_status
    
    trading_bot.disconnect_mt5()
    bot_status["connected_to_mt5"] = False
    bot_status["running"] = False
    bot_status["auto_trading"] = False
    
    await broadcast_to_websockets({
        "type": "mt5_disconnection",
        "message": "Disconnected from MT5"
    })
    
    return {"success": True, "message": "Disconnected from MT5"}

@api_router.post("/config/update")
async def update_config(config: TradingConfig):
    """Update trading configuration"""
    trading_bot.config = config
    
    return {"success": True, "message": "Configuration updated", "config": config.dict()}

@api_router.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "config": trading_bot.config.dict(),
        "available_timeframes": list(TIMEFRAMES.keys())
    }

@api_router.post("/analysis/symbol")
async def analyze_symbol(symbol: str, timeframe: str = "M5"):
    """Analyze a specific symbol"""
    if not trading_bot.connected:
        raise HTTPException(status_code=400, detail="Not connected to MT5")
    
    analysis = trading_bot.analyze_symbol(symbol, timeframe)
    return analysis

@api_router.get("/analysis/all")
async def analyze_all_symbols():
    """Analyze all configured symbols"""
    if not trading_bot.connected:
        raise HTTPException(status_code=400, detail="Not connected to MT5")
    
    analyses = []
    for symbol in trading_bot.config.symbols:
        analysis = trading_bot.analyze_symbol(symbol, trading_bot.config.timeframe)
        if "error" not in analysis:
            analyses.append(analysis)
    
    return {"analyses": analyses, "timestamp": datetime.utcnow()}

@api_router.post("/trading/start")
async def start_trading():
    """Start automatic trading"""
    global bot_status
    
    if not trading_bot.connected:
        raise HTTPException(status_code=400, detail="Not connected to MT5")
    
    trading_bot.start_auto_trading()
    bot_status["running"] = True
    bot_status["auto_trading"] = True
    
    await broadcast_to_websockets({
        "type": "trading_status",
        "running": True,
        "message": "Automatic trading started"
    })
    
    return {"success": True, "message": "Automatic trading started"}

@api_router.post("/trading/stop")
async def stop_trading():
    """Stop automatic trading"""
    global bot_status
    
    trading_bot.stop_auto_trading()
    bot_status["running"] = False
    bot_status["auto_trading"] = False
    
    await broadcast_to_websockets({
        "type": "trading_status",
        "running": False,
        "message": "Automatic trading stopped"
    })
    
    return {"success": True, "message": "Automatic trading stopped"}

@api_router.get("/status")
async def get_status():
    """Get bot status"""
    if trading_bot.connected:
        trading_bot.update_account_info()
        bot_status["balance"] = trading_bot.account_info.get("balance", 0)
        bot_status["equity"] = trading_bot.account_info.get("equity", 0)
        bot_status["total_trades"] = len(trading_bot.trade_history)
    
    return bot_status

@api_router.get("/trades/history")
async def get_trade_history():
    """Get trade history"""
    return {"trades": trading_bot.trade_history}

@api_router.get("/positions")
async def get_positions():
    """Get current positions"""
    if trading_bot.connected:
        trading_bot.update_account_info()
    return {"positions": trading_bot.positions}

@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    ws_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        if websocket in ws_connections:
            ws_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in ws_connections:
            ws_connections.remove(websocket)

# Background tasks
async def continuous_updates():
    """Send continuous updates to connected clients"""
    while True:
        try:
            if trading_bot.connected:
                # Send status update
                trading_bot.update_account_info()
                bot_status["balance"] = trading_bot.account_info.get("balance", 0)
                bot_status["equity"] = trading_bot.account_info.get("equity", 0)
                bot_status["total_trades"] = len(trading_bot.trade_history)
                
                await broadcast_to_websockets({
                    "type": "status_update",
                    "status": bot_status,
                    "positions": trading_bot.positions,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            await asyncio.sleep(10)  # Update every 10 seconds
            
        except Exception as e:
            logger.error(f"Error in continuous updates: {e}")
            await asyncio.sleep(30)

# Include router and middleware
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main HTML page"""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    else:
        return HTMLResponse(content="<h1>Ultra Trading Bot</h1><p>Interface file not found</p>")

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("Ultra Trading Bot starting...")
    asyncio.create_task(continuous_updates())

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("Ultra Trading Bot shutting down...")
    if trading_bot:
        trading_bot.disconnect_mt5()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")