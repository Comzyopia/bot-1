"""
Configuration centralisée pour le Bot de Trading Ultra-Performant
Version: 2.0 - Professional Edition
"""

import os
from typing import Dict, List
from enum import Enum
import json

class Timeframe(Enum):
    """Timeframes disponibles pour l'analyse"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"
    MN1 = "MN1"

class TradingMode(Enum):
    """Modes de trading disponibles"""
    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"

class RiskLevel(Enum):
    """Niveaux de risque prédéfinis"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

# Configuration MT5 Timeframes
MT5_TIMEFRAMES = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
    Timeframe.W1: 10080,
    Timeframe.MN1: 43200
}

# Configuration des risques par niveau
RISK_CONFIGS = {
    RiskLevel.CONSERVATIVE: {
        "max_risk_per_trade": 0.01,  # 1%
        "max_drawdown": 0.10,        # 10%
        "max_positions": 3,
        "lot_size": 0.01,
        "confidence_threshold": 0.80
    },
    RiskLevel.MODERATE: {
        "max_risk_per_trade": 0.02,  # 2%
        "max_drawdown": 0.15,        # 15%
        "max_positions": 5,
        "lot_size": 0.02,
        "confidence_threshold": 0.70
    },
    RiskLevel.AGGRESSIVE: {
        "max_risk_per_trade": 0.05,  # 5%
        "max_drawdown": 0.25,        # 25%
        "max_positions": 10,
        "lot_size": 0.05,
        "confidence_threshold": 0.60
    }
}

# Symboles disponibles par catégorie
SYMBOL_CATEGORIES = {
    "forex_majors": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
    "forex_minors": ["EURJPY", "GBPJPY", "EURGBP", "EURAUD", "EURCHF", "GBPAUD", "GBPCHF"],
    "forex_exotics": ["USDZAR", "USDTRY", "USDSEK", "USDNOK", "USDMXN", "USDPLN"],
    "metals": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"],
    "indices": ["US30", "US500", "NAS100", "GER30", "UK100", "JPN225", "AUS200"],
    "crypto": ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "ADAUSD", "DOTUSD"],
    "commodities": ["USOIL", "UKOIL", "NGAS", "WHEAT", "CORN", "SOYBEAN"]
}

# Configuration des indicateurs techniques
TECHNICAL_INDICATORS = {
    "trend": {
        "sma": {"periods": [20, 50, 200], "enabled": True},
        "ema": {"periods": [12, 26, 50], "enabled": True},
        "macd": {"fast": 12, "slow": 26, "signal": 9, "enabled": True},
        "adx": {"period": 14, "enabled": True},
        "bollinger": {"period": 20, "deviation": 2, "enabled": True}
    },
    "momentum": {
        "rsi": {"period": 14, "enabled": True},
        "stoch": {"k": 14, "d": 3, "enabled": True},
        "cci": {"period": 20, "enabled": True},
        "williams": {"period": 14, "enabled": True}
    },
    "volume": {
        "volume_sma": {"period": 20, "enabled": True},
        "mfi": {"period": 14, "enabled": True},
        "obv": {"enabled": True}
    },
    "volatility": {
        "atr": {"period": 14, "enabled": True},
        "bb_width": {"period": 20, "enabled": True},
        "vix": {"enabled": False}
    }
}

# Configuration des patterns de chandeliers
CANDLESTICK_PATTERNS = {
    "reversal": {
        "doji": {"enabled": True, "weight": 0.7},
        "hammer": {"enabled": True, "weight": 0.8},
        "shooting_star": {"enabled": True, "weight": 0.8},
        "engulfing": {"enabled": True, "weight": 0.9},
        "harami": {"enabled": True, "weight": 0.6},
        "morning_star": {"enabled": True, "weight": 0.9},
        "evening_star": {"enabled": True, "weight": 0.9}
    },
    "continuation": {
        "three_white_soldiers": {"enabled": True, "weight": 0.8},
        "three_black_crows": {"enabled": True, "weight": 0.8},
        "rising_three": {"enabled": True, "weight": 0.7},
        "falling_three": {"enabled": True, "weight": 0.7}
    }
}

# Configuration de l'apprentissage par renforcement
RL_CONFIG = {
    "network": {
        "hidden_layers": [512, 256, 128, 64],
        "activation": "relu",
        "dropout": 0.3,
        "learning_rate": 0.001,
        "optimizer": "adam"
    },
    "training": {
        "episodes": 1000,
        "batch_size": 64,
        "memory_size": 50000,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "target_update_frequency": 100
    },
    "environment": {
        "lookback_window": 100,
        "feature_scaling": True,
        "reward_function": "profit_based",
        "actions": ["HOLD", "BUY", "SELL", "CLOSE_ALL"]
    }
}

# Configuration des notifications
NOTIFICATION_CONFIG = {
    "email": {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "",
        "password": "",
        "recipients": []
    },
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": ""
    },
    "webhook": {
        "enabled": False,
        "url": "",
        "headers": {}
    }
}

# Configuration du backtesting
BACKTEST_CONFIG = {
    "data_source": "mt5",
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "initial_balance": 100000,
    "commission": 0.7,  # Commission par lot
    "spread": 2,        # Spread en points
    "slippage": 1,      # Slippage en points
    "leverage": 100
}

# Configuration des alertes
ALERT_CONFIG = {
    "price_alerts": {
        "support_resistance_break": True,
        "trend_reversal": True,
        "high_volume_spike": True,
        "volatility_extreme": True
    },
    "performance_alerts": {
        "drawdown_limit": True,
        "profit_target": True,
        "consecutive_losses": True,
        "win_rate_threshold": True
    },
    "system_alerts": {
        "connection_loss": True,
        "execution_errors": True,
        "memory_usage": True,
        "cpu_usage": True
    }
}

# Configuration par défaut
DEFAULT_CONFIG = {
    "trading": {
        "mode": TradingMode.DEMO.value,
        "timeframe": Timeframe.M5.value,
        "risk_level": RiskLevel.MODERATE.value,
        "symbols": SYMBOL_CATEGORIES["forex_majors"][:3],
        "max_positions": 5,
        "trading_hours": {
            "start": "00:00",
            "end": "23:59",
            "timezone": "UTC"
        }
    },
    "analysis": {
        "lookback_periods": 200,
        "min_confidence": 0.70,
        "combine_timeframes": True,
        "use_ml": True,
        "use_price_action": True,
        "use_indicators": True
    },
    "risk_management": {
        "max_risk_per_trade": 0.02,
        "max_drawdown": 0.15,
        "position_sizing": "fixed",
        "stop_loss_method": "dynamic",
        "take_profit_method": "dynamic"
    },
    "performance": {
        "track_metrics": True,
        "save_trades": True,
        "generate_reports": True,
        "report_frequency": "daily"
    }
}

class ConfigManager:
    """Gestionnaire de configuration centralisé"""
    
    def __init__(self, config_file: str = "bot_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Charge la configuration depuis le fichier"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement de la configuration: {e}")
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()
    
    def save_config(self):
        """Sauvegarde la configuration dans le fichier"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la configuration: {e}")
    
    def get(self, key: str, default=None):
        """Récupère une valeur de configuration"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def set(self, key: str, value):
        """Définit une valeur de configuration"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        self.save_config()
    
    def get_timeframe_mt5(self, timeframe: str) -> int:
        """Convertit un timeframe string en valeur MT5"""
        try:
            tf = Timeframe(timeframe)
            return MT5_TIMEFRAMES.get(tf, 5)
        except:
            return 5
    
    def get_risk_config(self, risk_level: str) -> Dict:
        """Récupère la configuration de risque"""
        try:
            level = RiskLevel(risk_level)
            return RISK_CONFIGS.get(level, RISK_CONFIGS[RiskLevel.MODERATE])
        except:
            return RISK_CONFIGS[RiskLevel.MODERATE]
    
    def get_symbols_by_category(self, category: str) -> List[str]:
        """Récupère les symboles par catégorie"""
        return SYMBOL_CATEGORIES.get(category, [])
    
    def validate_config(self) -> bool:
        """Valide la configuration actuelle"""
        required_keys = [
            "trading.mode",
            "trading.timeframe",
            "trading.symbols",
            "risk_management.max_risk_per_trade"
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                return False
        
        return True

# Instance globale du gestionnaire de configuration
config_manager = ConfigManager()