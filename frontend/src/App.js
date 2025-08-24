import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';

// 🚀 ULTRA TRADING BOT DASHBOARD - RÉVOLUTION v3.0
const UltraTradingBotDashboard = () => {
  // États principaux
  const [botStatus, setBotStatus] = useState({
    running: false,
    connected_to_mt5: false,
    ultra_mode_active: false,
    price_action_ai_enabled: false,
    ultra_reward_system_active: false,
    balance: 0,
    equity: 0,
    total_trades: 0,
    win_rate: 0
  });

  const [performance, setPerformance] = useState({
    system_status: { ultra_mode: false },
    performance_metrics: {
      total_trades: 0,
      win_rate: 0,
      profit_factor: 1.0,
      sharpe_ratio: 0,
      max_drawdown: 0,
      current_drawdown: 0,
      ultra_reward_score: 0,
      price_action_accuracy: 0
    },
    risk_metrics: {
      account_info: { balance: 0, equity: 0 },
      risk_metrics: { consecutive_losses: 0 }
    }
  });

  const [signals, setSignals] = useState([]);
  const [mt5Config, setMt5Config] = useState({
    login: '1296306',
    password: '@0JxCmJa',
    server: 'FreshForex-MT5'
  });
  
  const [tradingConfig, setTradingConfig] = useState({
    timeframe: 'M5',
    symbols: ['EURUSD', 'GBPUSD', 'USDJPY'],
    risk_level: 'moderate',
    max_positions: 5,
    price_action_mode: true,
    ultra_reward_system: true,
    smart_risk_management: true,
    min_confidence_threshold: 0.75
  });

  const [logs, setLogs] = useState([]);
  const [isConnecting, setIsConnecting] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);

  // Backend URL depuis les variables d'environnement
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Fonction pour ajouter des logs
  const addLog = useCallback((message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-49), { timestamp, message, type }]);
  }, []);

  // Récupération du statut du bot
  const fetchBotStatus = useCallback(async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/status`);
      setBotStatus(response.data.bot_status);
      
      if (response.data.performance_summary) {
        setPerformance(response.data.performance_summary);
      }
    } catch (error) {
      console.error('Erreur récupération statut:', error);
    }
  }, [BACKEND_URL]);

  // Récupération des signaux
  const fetchSignals = useCallback(async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/signals`);
      setSignals(response.data.signals || []);
    } catch (error) {
      console.error('Erreur récupération signaux:', error);
    }
  }, [BACKEND_URL]);

  // Connexion MT5
  const connectMT5 = async () => {
    setIsConnecting(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/mt5/connect`, mt5Config);
      if (response.data.success) {
        addLog('🚀 Connexion MT5 réussie avec systèmes ultra-avancés', 'success');
        addLog('✅ Ultra Reward System activé', 'success');
        addLog('✅ Price Action AI activé', 'success');
        addLog('✅ Smart Risk Management activé', 'success');
        await fetchBotStatus();
      }
    } catch (error) {
      addLog(`❌ Erreur connexion MT5: ${error.response?.data?.detail || error.message}`, 'error');
    }
    setIsConnecting(false);
  };

  // Démarrage du trading
  const startTrading = async () => {
    try {
      const response = await axios.post(`${BACKEND_URL}/api/trading/start`);
      if (response.data.success) {
        addLog('🚀 Ultra Trading Bot démarré!', 'success');
        response.data.systems_initialized?.forEach(system => {
          addLog(`✅ ${system}`, 'success');
        });
        await fetchBotStatus();
      }
    } catch (error) {
      addLog(`❌ Erreur démarrage: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  // Arrêt du trading
  const stopTrading = async () => {
    try {
      const response = await axios.post(`${BACKEND_URL}/api/trading/stop`);
      if (response.data.success) {
        addLog('🛑 Trading arrêté', 'warning');
        await fetchBotStatus();
      }
    } catch (error) {
      addLog(`❌ Erreur arrêt: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  // Mise à jour de la configuration
  const updateConfig = async () => {
    try {
      const response = await axios.post(`${BACKEND_URL}/api/config/update`, tradingConfig);
      if (response.data.success) {
        addLog('🔧 Configuration ultra-avancée mise à jour', 'success');
      }
    } catch (error) {
      addLog(`❌ Erreur config: ${error.response?.data?.detail || error.message}`, 'error');
    }
  };

  // WebSocket pour mises à jour temps réel
  useEffect(() => {
    const connectWebSocket = () => {
      const wsUrl = BACKEND_URL.replace('http', 'ws') + '/api/ws';
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        setWsConnected(true);
        addLog('🌐 Connexion WebSocket établie', 'success');
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'real_time_update' && data.performance) {
          setPerformance(data.performance);
        } else if (data.type === 'connection_established') {
          addLog(data.message, 'success');
        }
      };
      
      ws.onclose = () => {
        setWsConnected(false);
        addLog('🔌 Connexion WebSocket fermée', 'warning');
        setTimeout(connectWebSocket, 5000); // Reconnexion automatique
      };
      
      return ws;
    };

    const ws = connectWebSocket();
    return () => ws.close();
  }, [BACKEND_URL, addLog]);

  // Polling pour les données
  useEffect(() => {
    fetchBotStatus();
    fetchSignals();
    
    const interval = setInterval(() => {
      fetchBotStatus();
      fetchSignals();
    }, 10000); // Toutes les 10 secondes

    return () => clearInterval(interval);
  }, [fetchBotStatus, fetchSignals]);

  // Fonction pour formater les pourcentages
  const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
  
  // Fonction pour formater les valeurs monétaires
  const formatCurrency = (value) => `$${value.toFixed(2)}`;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white">
      {/* Header Ultra-Premium */}
      <header className="bg-black/50 backdrop-blur-lg border-b border-purple-500/30 p-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg flex items-center justify-center">
              <span className="text-2xl font-bold">🚀</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                Ultra Trading Bot v3.0
              </h1>
              <p className="text-gray-400">Révolution Performance - Price Action AI</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${
              wsConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm font-medium">{wsConnected ? 'WebSocket Connecté' : 'Déconnecté'}</span>
            </div>
            
            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${
              botStatus.ultra_mode_active ? 'bg-purple-500/20 text-purple-400' : 'bg-gray-500/20 text-gray-400'
            }`}>
              <span className="text-sm font-medium">Ultra Mode</span>
              <div className={`w-2 h-2 rounded-full ${botStatus.ultra_mode_active ? 'bg-purple-400' : 'bg-gray-400'}`}></div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6 space-y-6">
        
        {/* Statut des Systèmes Ultra-Avancés */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className={`p-4 rounded-xl border backdrop-blur-lg ${
            botStatus.connected_to_mt5 ? 'bg-green-500/10 border-green-500/30' : 'bg-red-500/10 border-red-500/30'
          }`}>
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${botStatus.connected_to_mt5 ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="font-medium">MT5 Connection</span>
            </div>
          </div>
          
          <div className={`p-4 rounded-xl border backdrop-blur-lg ${
            botStatus.price_action_ai_enabled ? 'bg-blue-500/10 border-blue-500/30' : 'bg-gray-500/10 border-gray-500/30'
          }`}>
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${botStatus.price_action_ai_enabled ? 'bg-blue-400' : 'bg-gray-400'}`}></div>
              <span className="font-medium">Price Action AI</span>
            </div>
          </div>
          
          <div className={`p-4 rounded-xl border backdrop-blur-lg ${
            botStatus.ultra_reward_system_active ? 'bg-purple-500/10 border-purple-500/30' : 'bg-gray-500/10 border-gray-500/30'
          }`}>
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${botStatus.ultra_reward_system_active ? 'bg-purple-400' : 'bg-gray-400'}`}></div>
              <span className="font-medium">Ultra Reward System</span>
            </div>
          </div>
          
          <div className={`p-4 rounded-xl border backdrop-blur-lg ${
            botStatus.running ? 'bg-yellow-500/10 border-yellow-500/30' : 'bg-gray-500/10 border-gray-500/30'
          }`}>
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${botStatus.running ? 'bg-yellow-400' : 'bg-gray-400'}`}></div>
              <span className="font-medium">Auto Trading</span>
            </div>
          </div>
        </div>

        {/* Métriques de Performance Ultra-Avancées */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-green-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-green-400 font-medium">Balance</span>
              <span className="text-2xl">💰</span>
            </div>
            <div className="text-3xl font-bold text-green-400">
              {formatCurrency(performance.risk_metrics?.account_info?.balance || 0)}
            </div>
            <div className="text-sm text-gray-400 mt-1">
              Equity: {formatCurrency(performance.risk_metrics?.account_info?.equity || 0)}
            </div>
          </div>

          <div className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-blue-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-blue-400 font-medium">Win Rate</span>
              <span className="text-2xl">🎯</span>
            </div>
            <div className="text-3xl font-bold text-blue-400">
              {formatPercent(performance.performance_metrics?.win_rate || 0)}
            </div>
            <div className="text-sm text-gray-400 mt-1">
              Trades: {performance.performance_metrics?.total_trades || 0}
            </div>
          </div>

          <div className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-purple-400 font-medium">Sharpe Ratio</span>
              <span className="text-2xl">📊</span>
            </div>
            <div className="text-3xl font-bold text-purple-400">
              {(performance.performance_metrics?.sharpe_ratio || 0).toFixed(2)}
            </div>
            <div className="text-sm text-gray-400 mt-1">
              P.Factor: {(performance.performance_metrics?.profit_factor || 0).toFixed(2)}
            </div>
          </div>

          <div className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-red-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-red-400 font-medium">Drawdown</span>
              <span className="text-2xl">⚠️</span>
            </div>
            <div className="text-3xl font-bold text-red-400">
              {formatPercent(performance.performance_metrics?.current_drawdown || 0)}
            </div>
            <div className="text-sm text-gray-400 mt-1">
              Max: {formatPercent(performance.performance_metrics?.max_drawdown || 0)}
            </div>
          </div>
        </div>

        {/* Configuration et Contrôles */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Connexion MT5 */}
          <div className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-gray-500/30">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              🔌 Connexion MT5
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Login</label>
                <input
                  type="text"
                  value={mt5Config.login}
                  onChange={(e) => setMt5Config({...mt5Config, login: e.target.value})}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg focus:border-blue-500 focus:outline-none"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Password</label>
                <input
                  type="password"
                  value={mt5Config.password}
                  onChange={(e) => setMt5Config({...mt5Config, password: e.target.value})}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg focus:border-blue-500 focus:outline-none"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Server</label>
                <input
                  type="text"
                  value={mt5Config.server}
                  onChange={(e) => setMt5Config({...mt5Config, server: e.target.value})}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg focus:border-blue-500 focus:outline-none"
                />
              </div>
              
              <button
                onClick={connectMT5}
                disabled={isConnecting}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 px-6 py-3 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isConnecting ? '🔄 Connexion...' : '🚀 Connecter MT5'}
              </button>
            </div>
          </div>

          {/* Configuration Trading */}
          <div className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-gray-500/30">
            <h3 className="text-xl font-bold mb-4 flex items-center">
              ⚙️ Configuration Ultra-Avancée
            </h3>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Timeframe</label>
                  <select
                    value={tradingConfig.timeframe}
                    onChange={(e) => setTradingConfig({...tradingConfig, timeframe: e.target.value})}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg focus:border-blue-500 focus:outline-none"
                  >
                    <option value="M1">M1</option>
                    <option value="M5">M5</option>
                    <option value="M15">M15</option>
                    <option value="H1">H1</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Risk Level</label>
                  <select
                    value={tradingConfig.risk_level}
                    onChange={(e) => setTradingConfig({...tradingConfig, risk_level: e.target.value})}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg focus:border-blue-500 focus:outline-none"
                  >
                    <option value="ultra_conservative">Ultra Conservative</option>
                    <option value="conservative">Conservative</option>
                    <option value="moderate">Moderate</option>
                    <option value="aggressive">Aggressive</option>
                  </select>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Confiance Min: {tradingConfig.min_confidence_threshold}
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="0.95"
                  step="0.05"
                  value={tradingConfig.min_confidence_threshold}
                  onChange={(e) => setTradingConfig({...tradingConfig, min_confidence_threshold: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>
              
              <div className="flex items-center space-x-4">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={tradingConfig.price_action_mode}
                    onChange={(e) => setTradingConfig({...tradingConfig, price_action_mode: e.target.checked})}
                    className="rounded"
                  />
                  <span className="text-sm">🎯 Price Action Mode</span>
                </label>
                
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={tradingConfig.ultra_reward_system}
                    onChange={(e) => setTradingConfig({...tradingConfig, ultra_reward_system: e.target.checked})}
                    className="rounded"
                  />
                  <span className="text-sm">🚀 Ultra Rewards</span>
                </label>
              </div>
              
              <button
                onClick={updateConfig}
                className="w-full bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 px-6 py-3 rounded-lg font-medium transition-all"
              >
                🔧 Mettre à Jour Config
              </button>
            </div>
          </div>
        </div>

        {/* Contrôles de Trading */}
        <div className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-gray-500/30">
          <h3 className="text-xl font-bold mb-4">🎮 Contrôles de Trading</h3>
          
          <div className="flex space-x-4">
            <button
              onClick={startTrading}
              disabled={!botStatus.connected_to_mt5 || botStatus.running}
              className="flex-1 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 px-6 py-4 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              🚀 Démarrer Ultra Trading
            </button>
            
            <button
              onClick={stopTrading}
              disabled={!botStatus.running}
              className="flex-1 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 px-6 py-4 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              🛑 Arrêter Trading
            </button>
          </div>
        </div>

        {/* Signaux Price Action */}
        <div className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-gray-500/30">
          <h3 className="text-xl font-bold mb-4 flex items-center">
            🎯 Signaux Price Action Ultra-Avancés
          </h3>
          
          {signals.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {signals.map((signal, index) => (
                <div key={index} className={`p-4 rounded-lg border ${
                  signal.action === 'BUY' 
                    ? 'bg-green-500/10 border-green-500/30' 
                    : signal.action === 'SELL'
                    ? 'bg-red-500/10 border-red-500/30'
                    : 'bg-gray-500/10 border-gray-500/30'
                }`}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-lg">{signal.symbol}</span>
                    <span className={`px-2 py-1 rounded text-sm font-medium ${
                      signal.action === 'BUY' ? 'bg-green-500 text-white' : 
                      signal.action === 'SELL' ? 'bg-red-500 text-white' : 'bg-gray-500 text-white'
                    }`}>
                      {signal.action}
                    </span>
                  </div>
                  
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Confiance:</span>
                      <span className="text-blue-400">{(signal.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Pattern:</span>
                      <span className="text-purple-400">{signal.pattern || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">R/R:</span>
                      <span className="text-yellow-400">{signal.risk_reward_ratio.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Pips attendus:</span>
                      <span className="text-green-400">{signal.expected_move_pips.toFixed(0)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <span className="text-4xl mb-4 block">🔍</span>
              <p>Aucun signal détecté actuellement</p>
              <p className="text-sm mt-2">L'IA Price Action analyse les marchés...</p>
            </div>
          )}
        </div>

        {/* Logs Système */}
        <div className="bg-black/30 backdrop-blur-lg rounded-xl p-6 border border-gray-500/30">
          <h3 className="text-xl font-bold mb-4 flex items-center">
            📋 Logs Système Ultra-Avancé
          </h3>
          
          <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto">
            {logs.length > 0 ? (
              logs.map((log, index) => (
                <div key={index} className={`text-sm mb-1 ${
                  log.type === 'success' ? 'text-green-400' :
                  log.type === 'error' ? 'text-red-400' :
                  log.type === 'warning' ? 'text-yellow-400' :
                  'text-gray-300'
                }`}>
                  <span className="text-gray-500">[{log.timestamp}]</span> {log.message}
                </div>
              ))
            ) : (
              <div className="text-gray-500 text-center">Aucun log disponible</div>
            )}
          </div>
          
          <button
            onClick={() => setLogs([])}
            className="mt-4 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-sm transition-colors"
          >
            🗑️ Effacer Logs
          </button>
        </div>

      </div>
    </div>
  );
};

export default UltraTradingBotDashboard;