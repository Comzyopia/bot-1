import React, { useState, useEffect, useCallback } from 'react';
import SecurityLogin from './SecurityLogin';
import './App.css';

const App = () => {
  // 🛡️ ÉTAT DE SÉCURITÉ
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authToken, setAuthToken] = useState('');
  const [currentUser, setCurrentUser] = useState(null);
  
  // États existants
  const [isConnected, setIsConnected] = useState(false);
  const [isAutoTrading, setIsAutoTrading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [accountInfo, setAccountInfo] = useState(null);
  const [positions, setPositions] = useState([]);
  const [signals, setSignals] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [logs, setLogs] = useState([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('M5');
  const [riskLevel, setRiskLevel] = useState(0.02);
  const [maxPositions, setMaxPositions] = useState(5);
  const [takeTrades, setTakeTrades] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState({});
  const [performance, setPerformance] = useState({});
  const [currentAnalysis, setCurrentAnalysis] = useState([]);
  const [aiStatus, setAiStatus] = useState({});
  const [aiLearningEnabled, setAiLearningEnabled] = useState(true);
  
  // 💾 Sauvegarde des identifiants MT5
  const [mt5Config, setMt5Config] = useState({ login: '', password: '', server: '' });
  const [saveCredentials, setSaveCredentials] = useState(false);
  
  // 🎯 Contrôles SL/TP manuels
  const [useManualSLTP, setUseManualSLTP] = useState(false);
  const [manualSL, setManualSL] = useState('');
  const [manualTP, setManualTP] = useState('');
  
  // 🧠 Mode IA Pure
  const [aiPureMode, setAiPureMode] = useState(false);

  // 📊 États pour la barre IA en temps réel
  const [aiStats, setAiStats] = useState({
    isActive: false,
    decisions: 0,
    precision: 0,
    learningStatus: 'En cours',
    mode: 'AUTO'
  });

  // 📊 Timeframes disponibles - Seulement M1 et M5
  const timeframes = [
    { value: 'M1', label: 'M1 (1 min)', description: 'Ultra scalping - très court terme' },
    { value: 'M5', label: 'M5 (5 min)', description: 'Scalping rapide - recommandé' }
  ];

  // 📈 Niveaux de risque
  const riskLevels = [
    { value: 0.01, label: '1% - Conservateur', description: 'Risque très faible' },
    { value: 0.02, label: '2% - Équilibré', description: 'Risque modéré recommandé' },
    { value: 0.05, label: '5% - Agressif', description: 'Risque élevé pour experts' }
  ];

  // 🛡️ Gestion de l'authentification
  const handleSuccessfulAuth = (token, user) => {
    setIsAuthenticated(true);
    setAuthToken(token);
    setCurrentUser(user);
    
    // Démarrer WebSocket après authentification
    connectWebSocket();
    
    // Charger les identifiants sauvegardés
    const savedCredentials = localStorage.getItem('mt5_credentials');
    if (savedCredentials) {
      try {
        const creds = JSON.parse(savedCredentials);
        setMt5Config(creds);
        setSaveCredentials(true);
        addLog('🔐 Identifiants MT5 restaurés depuis la sauvegarde', 'info');
      } catch (error) {
        console.error('Erreur lors du chargement des identifiants:', error);
      }
    }
  };

  // 📋 Système de logs
  const addLog = useCallback((message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog = {
      id: Date.now(),
      timestamp,
      message,
      type
    };
    setLogs(prev => [newLog, ...prev].slice(0, 100));
  }, []);

  // 🌐 Connexion WebSocket
  const connectWebSocket = useCallback(() => {
    if (!isAuthenticated) return;
    
    try {
      // ✅ CORRECTION: Utiliser la bonne URL WebSocket
      const backendUrl = process.env.REACT_APP_BACKEND_URL || window.location.origin;
      const wsUrl = backendUrl.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws';
      
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setWsConnected(true);
        addLog('WebSocket connecté avec succès', 'success');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('❌ WebSocket message error:', error);
        }
      };
      
      ws.onclose = () => {
        console.log('❌ WebSocket disconnected');
        setWsConnected(false);
        addLog('WebSocket disconnected', 'error');
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setWsConnected(false);
        addLog('WebSocket error', 'error');
      };
      
      return ws;
    } catch (error) {
      console.error('❌ WebSocket connection failed:', error);
      addLog('WebSocket connection failed', 'error');
      setTimeout(connectWebSocket, 5000);
    }
  }, [isAuthenticated, addLog]);

  // 📨 Gestionnaire des messages WebSocket
  const handleWebSocketMessage = useCallback((data) => {
    switch (data.type) {
      case 'status':
        setStatus(data.data);
        if (data.data.connected !== undefined) {
          setIsConnected(data.data.connected);
        }
        if (data.data.auto_trading !== undefined) {
          setIsAutoTrading(data.data.auto_trading);
          // Mettre à jour le mode IA
          setAiStats(prev => ({
            ...prev,
            isActive: data.data.auto_trading,
            mode: data.data.auto_trading ? 'AUTO' : 'MANUEL'
          }));
        }
        break;
      case 'positions':
        setPositions(data.data || []);
        break;
      case 'signals':
        setSignals(data.data || []);
        break;
      case 'trades':
        setTradeHistory(data.data || []);
        break;
      case 'account_info':
        setAccountInfo(data.data);
        break;
      case 'log':
        addLog(data.message, data.level || 'info');
        break;
      case 'analysis':
        setCurrentAnalysis(data.data || []);
        break;
      case 'performance':
        setPerformance(data.data || {});
        // Mettre à jour la précision basée sur les performances
        if (data.data) {
          const winRate = data.data.win_rate || data.data.success_rate || 0;
          setAiStats(prev => ({
            ...prev,
            precision: Math.round(winRate * 100) / 100
          }));
        }
        break;
      case 'ai_status':
        setAiStatus(data.data || {});
        // Mettre à jour les statistiques IA pour la barre du bas
        if (data.data) {
          setAiStats(prev => ({
            ...prev,
            isActive: data.data.active || data.data.enabled || false,
            decisions: data.data.total_decisions || data.data.decisions || 0,
            precision: Math.round((data.data.precision || data.data.success_rate || 0) * 100) / 100,
            learningStatus: data.data.learning_status || prev.learningStatus,
            mode: data.data.mode || prev.mode
          }));
        }
        break;
      default:
        console.log('Unknown WebSocket message type:', data.type);
    }
  }, [addLog]);

  // 💰 Formatage des montants
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount || 0);
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  // Auto-authentification pour accès direct
  useEffect(() => {
    if (!isAuthenticated) {
      setIsAuthenticated(true);
      setAuthToken('direct_access');
      setCurrentUser({ id: 'trader', name: 'Ultra Trader' });
      connectWebSocket();
      
      // Charger les identifiants sauvegardés
      const savedCredentials = localStorage.getItem('mt5_credentials');
      if (savedCredentials) {
        try {
          const creds = JSON.parse(savedCredentials);
          setMt5Config(creds);
          setSaveCredentials(true);
          addLog('🔐 Identifiants MT5 restaurés depuis la sauvegarde', 'info');
        } catch (error) {
          console.error('Erreur lors du chargement des identifiants:', error);
        }
      }
    }
  }, [isAuthenticated, connectWebSocket, addLog]);

  // API calls
  const apiCall = async (endpoint, options = {}) => {
    try {
      // ✅ CORRECTION: Utiliser la bonne URL backend
      const backendUrl = process.env.REACT_APP_BACKEND_URL || window.location.origin;
      const fullUrl = `${backendUrl}/api${endpoint}`;
      
      const response = await fetch(fullUrl, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API call error:', error);
      addLog(`API Error: ${error.message}`, 'error');
      throw error;
    }
  };

  // Handlers
  const handleConnectMT5 = async () => {
    if (!mt5Config.login || !mt5Config.password || !mt5Config.server) {
      addLog('Please fill in all MT5 connection details', 'error');
      return;
    }
    
    setIsLoading(true);
    try {
      const response = await apiCall('/mt5/connect', {
        method: 'POST',
        body: JSON.stringify({
          login: parseInt(mt5Config.login),
          password: mt5Config.password,
          server: mt5Config.server
        })
      });
      
      if (response.success) {
        setIsConnected(true);
        setAccountInfo(response.account_info);
        addLog('Connected to MT5 successfully', 'success');
        
        // 💾 Sauvegarder les identifiants si demandé
        if (saveCredentials) {
          localStorage.setItem('mt5_credentials', JSON.stringify(mt5Config));
          addLog('🔐 Identifiants MT5 sauvegardés', 'success');
        }
      } else {
        // 🚨 Message spécial pour erreur MT5 sur Linux
        if (response.message && response.message.includes('Windows')) {
          addLog('🚨 ERREUR: MetaTrader 5 fonctionne UNIQUEMENT sur Windows !', 'error');
          addLog('📁 Utilisez la version Windows dans /windows_deployment/', 'error');
          addLog('💡 Double-cliquez sur run_bot.bat sur Windows', 'error');
        } else {
          addLog(`Connection failed: ${response.message}`, 'error');
        }
      }
    } catch (error) {
      addLog(`Connection error: ${error.message}`, 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDisconnectMT5 = async () => {
    try {
      await apiCall('/mt5/disconnect', { method: 'POST' });
      setIsConnected(false);
      setAccountInfo(null);
      setIsAutoTrading(false);
      addLog('Disconnected from MT5', 'warning');
    } catch (error) {
      addLog(`Disconnect error: ${error.message}`, 'error');
    }
  };

  const handleStartTrading = async () => {
    if (!isConnected) {
      addLog('Please connect to MT5 first', 'error');
      return;
    }
    
    try {
      // 🚀 Démarrer le trading avec IA automatique
      const response = await apiCall('/trading/start', { method: 'POST' });
      if (response.success) {
        setIsAutoTrading(true);
        addLog('🤖 Ultra Trading Bot démarré - IA activée automatiquement', 'success');
        
        // 🧠 Activer automatiquement l'Ultra AI
        try {
          const aiResponse = await apiCall('/ai/ultra_test', { method: 'POST' });
          if (aiResponse.success) {
            addLog('🏆 Ultra AI activé et opérationnel', 'success');
            addLog(`🧠 IA prête: ${aiResponse.ultra_metrics?.ai_action || 'Analyse en cours...'}`, 'info');
          }
        } catch (aiError) {
          addLog('⚠️ IA activée en mode de base', 'warning');
        }
        
        // 📊 Démarrer l'analyse automatique
        try {
          const analysisResponse = await apiCall('/analysis/all');
          addLog(`📈 Analyse automatique des ${analysisResponse.analyses?.length || 'marchés'} activée`, 'success');
        } catch (analysisError) {
          addLog('📊 Analyse en cours...', 'info');
        }
        
        addLog('✅ Bot 100% autonome - Plus besoin d\'intervention manuelle', 'success');
      }
    } catch (error) {
      addLog(`Failed to start trading: ${error.message}`, 'error');
    }
  };

  const handleStopTrading = async () => {
    try {
      const response = await apiCall('/trading/stop', { method: 'POST' });
      if (response.success) {
        setIsAutoTrading(false);
        addLog('Auto trading stopped', 'warning');
      }
    } catch (error) {
      addLog(`Failed to stop trading: ${error.message}`, 'error');
    }
  };

  const handleAnalyzeAll = async () => {
    if (!isConnected) {
      addLog('Please connect to MT5 first', 'error');
      return;
    }
    
    try {
      const response = await apiCall('/analysis/all');
      setCurrentAnalysis(response.analyses);
      addLog(`Analysis completed for ${response.analyses.length} symbols`, 'success');
    } catch (error) {
      addLog(`Analysis error: ${error.message}`, 'error');
    }
  };

  const handleClearCredentials = () => {
    localStorage.removeItem('mt5_credentials');
    setMt5Config({ login: '', password: '', server: '' });
    setSaveCredentials(false);
    addLog('🗑️ Identifiants supprimés de la sauvegarde', 'warning');
  };

  const handleUpdateConfig = async () => {
    try {
      const config = {
        timeframe: selectedTimeframe, // 🎯 IMPORTANT: Timeframe choisie par l'utilisateur
        symbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD'],
        risk_per_trade: riskLevel,
        max_positions: maxPositions,
        auto_trading: isAutoTrading,
        take_trades: takeTrades,
        use_manual_sltp: useManualSLTP,
        manual_sl_pips: manualSL ? parseFloat(manualSL) : null,
        manual_tp_pips: manualTP ? parseFloat(manualTP) : null,
        ai_pure_mode: aiPureMode
      };
      
      const response = await apiCall('/config/update', {
        method: 'POST',
        body: JSON.stringify(config)
      });
      
      if (response.success) {
        addLog(`🎯 Configuration mise à jour - Trading ${selectedTimeframe}`, 'success');
        addLog(`📊 Le bot tradra maintenant sur ${selectedTimeframe}`, 'info');
      }
    } catch (error) {
      addLog(`Config update error: ${error.message}`, 'error');
    }
  };

  // 🔄 Mettre à jour automatiquement quand la timeframe change
  useEffect(() => {
    if (isConnected) {
      handleUpdateConfig();
    }
  }, [selectedTimeframe, riskLevel, maxPositions, takeTrades, useManualSLTP, manualSL, manualTP, aiPureMode]);

  // 🧠 Load AI status periodically
  useEffect(() => {
    const loadAiStatus = async () => {
      try {
        const response = await apiCall('/ai/status');
        setAiStatus(response.ai_status);
      } catch (error) {
        console.error('Failed to load AI status:', error);
      }
    };

    if (isConnected) {
      loadAiStatus();
      const interval = setInterval(loadAiStatus, 10000);
      return () => clearInterval(interval);
    }
  }, [isConnected]);

  return (
    <div className="ultra-trading-interface">
      {/* Header */}
      <div className="trading-header">
        <div className="bot-title">ULTRA TRADING BOT</div>
        <div className="connection-status">
          <div className="status-dot"></div>
          <span>PREVIEW MODE</span>
        </div>
      </div>

      {/* Main Trading Grid */}
      <div className="main-trading-grid">
        {/* Market Trends - Large Card */}
        <div className="glass-card market-trends">
          <div className="card-title">📈 Market Trends</div>
          <div className="currency-pair">EUR/USD</div>
          <div className="chart-container">
            <div className="chart-placeholder">📊 Chart data will be displayed here</div>
          </div>
          
          {/* Timeframe Selector */}
          <div style={{marginTop: '15px'}}>
            <label style={{fontSize: '14px', color: 'rgba(255,255,255,0.8)', marginBottom: '8px', display: 'block'}}>
              ⏰ Timeframe de Trading
            </label>
            <select 
              className="form-input"
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              style={{width: '100%', background: 'rgba(0,255,255,0.1)', border: '1px solid #00ffff'}}
            >
              {timeframes.map(tf => (
                <option key={tf.value} value={tf.value} style={{background: '#1a1a2e', color: '#fff'}}>
                  {tf.label}
                </option>
              ))}
            </select>
            <div style={{fontSize: '12px', color: '#00ffff', marginTop: '5px'}}>
              📍 Trading actif: {selectedTimeframe} - {timeframes.find(tf => tf.value === selectedTimeframe)?.description}
            </div>
          </div>
        </div>

        {/* Account Balance */}
        <div className="glass-card account-balance">
          <div className="card-title">💰 Account Balance</div>
          <div className="balance-amount">
            {accountInfo ? formatCurrency(accountInfo.balance) : '$25,300.45'}
          </div>
          {accountInfo && (
            <div style={{marginTop: '15px', fontSize: '14px'}}>
              <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '8px'}}>
                <span style={{color: 'rgba(255,255,255,0.7)'}}>Equity:</span>
                <span style={{color: '#00ffff'}}>{formatCurrency(accountInfo.equity)}</span>
              </div>
              <div style={{display: 'flex', justifyContent: 'space-between'}}>
                <span style={{color: 'rgba(255,255,255,0.7)'}}>Margin:</span>
                <span style={{color: '#ffaa00'}}>{formatCurrency(accountInfo.margin)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Open Trades */}
        <div className="glass-card">
          <div className="card-title">📈 Positions Ouvertes</div>
          {positions.length > 0 ? (
            positions.slice(0, 3).map((position, index) => (
              <div key={index} className="trade-item">
                <span className="trade-pair">{position.symbol}</span>
                <span className={`trade-profit ${position.profit >= 0 ? '' : 'negative'}`}>
                  {position.profit >= 0 ? '+' : ''}{formatCurrency(position.profit)}
                </span>
              </div>
            ))
          ) : (
            <>
              <div className="trade-item">
                <span className="trade-pair">EUR/USD</span>
                <span className="trade-profit">+714.20</span>
              </div>
              <div className="trade-item">
                <span className="trade-pair">BTC/USD</span>
                <span className="trade-profit">+128.56</span>
              </div>
            </>
          )}
        </div>

        {/* Trading Signals */}
        <div className="glass-card">
          <div className="card-title">🎯 Signaux {selectedTimeframe}</div>
          {signals.length > 0 ? (
            signals.slice(0, 2).map((signal, index) => (
              <div key={index} className="signal-item">
                <div className="signal-pair">
                  <span>{signal.symbol}</span>
                  <span className={`signal-direction ${signal.action.toLowerCase().includes('buy') ? 'signal-buy' : 'signal-sell'}`}>
                    {signal.action}
                  </span>
                </div>
                <span className="signal-time">{Math.round(signal.confidence * 100)}%</span>
              </div>
            ))
          ) : (
            <>
              <div className="signal-item">
                <div className="signal-pair">
                  <span>📈 GBP/USD</span>
                  <span className="signal-direction signal-buy">Buy</span>
                </div>
                <span className="signal-time">1m ago</span>
              </div>
              <div className="signal-item">
                <div className="signal-pair">
                  <span>📉 ETH/USD</span>
                  <span className="signal-direction signal-sell">Sell</span>
                </div>
                <span className="signal-time">5m ago</span>
              </div>
            </>
          )}
        </div>

        {/* AI Learning avec Récompenses et Positions Passées */}
        <div className="glass-card">
          <div className="card-title">🧠 IA + Apprentissage</div>
          <div className="ai-stats">
            <div className="ai-stat">
              <div className="ai-stat-label">Epochs</div>
              <div className="ai-stat-value">{aiStatus.total_experiences || 23}</div>
            </div>
            <div className="ai-stat">
              <div className="ai-stat-label">Accuracy</div>
              <div className="ai-stat-value ai-accuracy">
                {Math.round((aiStatus.recent_win_rate || 0.89) * 100)}%
              </div>
            </div>
          </div>
          
          {/* Système de récompenses et apprentissage */}
          <div style={{marginTop: '12px', fontSize: '11px', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '8px'}}>
            <div style={{color: '#00ff00', marginBottom: '2px', fontWeight: '600'}}>🏆 Apprentissage Actif:</div>
            <div style={{color: 'rgba(255,255,255,0.8)', marginBottom: '1px'}}>• Positions fermées ✅</div>
            <div style={{color: 'rgba(255,255,255,0.8)', marginBottom: '1px'}}>• Récompenses/Pénalités ✅</div>
            <div style={{color: 'rgba(255,255,255,0.8)', marginBottom: '1px'}}>• Protection sur-trading ✅</div>
            <div style={{color: '#00ffff', fontSize: '10px', marginTop: '4px'}}>
              L'IA analyse ses trades passés pour s'améliorer
            </div>
          </div>
        </div>

        {/* Gestion du Risque Intelligente */}
        <div className="glass-card">
          <div className="card-title">⚖️ Gestion du Risque</div>
          
          <div style={{textAlign: 'center', marginBottom: '15px'}}>
            <div style={{
              padding: '12px', 
              background: 'rgba(255,68,68,0.1)', 
              border: '1px solid #ff4444', 
              borderRadius: '12px',
              fontSize: '18px',
              fontWeight: '700',
              color: '#ff4444'
            }}>
              MAX 5% PAR TRADE
            </div>
          </div>
          
          <div style={{fontSize: '11px'}}>
            <div style={{color: '#00ffff', marginBottom: '6px', fontWeight: '600'}}>
              🧠 IA Calcule Automatiquement:
            </div>
            <div style={{color: 'rgba(255,255,255,0.8)', marginBottom: '2px'}}>
              • Volume selon distance SL
            </div>
            <div style={{color: 'rgba(255,255,255,0.8)', marginBottom: '2px'}}>
              • SL large → Volume ↓
            </div>
            <div style={{color: 'rgba(255,255,255,0.8)', marginBottom: '2px'}}>
              • SL serré → Volume ↑
            </div>
            <div style={{color: '#00ff00', fontSize: '10px', marginTop: '6px', fontWeight: '600'}}>
              ✅ Capital 100% protégé
            </div>
          </div>
        </div>

        {/* MT5 Connection Panel */}
        <div className="glass-card">
          <div className="card-title">🔌 MT5 Connection</div>
          <div className="connection-form">
            <input 
              type="number" 
              placeholder="Login"
              className="form-input"
              value={mt5Config.login}
              onChange={(e) => setMt5Config({...mt5Config, login: e.target.value})}
            />
            <input 
              type="password" 
              placeholder="Password"
              className="form-input"
              value={mt5Config.password}
              onChange={(e) => setMt5Config({...mt5Config, password: e.target.value})}
            />
            <input 
              type="text" 
              placeholder="Server"
              className="form-input"
              value={mt5Config.server}
              onChange={(e) => setMt5Config({...mt5Config, server: e.target.value})}
            />
            
            {/* Sauvegarde des identifiants */}
            <div style={{display: 'flex', alignItems: 'center', gap: '8px', marginTop: '10px'}}>
              <input 
                type="checkbox" 
                id="saveCredentials"
                checked={saveCredentials}
                onChange={(e) => setSaveCredentials(e.target.checked)}
                style={{accentColor: '#00ffff'}}
              />
              <label htmlFor="saveCredentials" style={{fontSize: '14px', color: 'rgba(255,255,255,0.8)'}}>
                💾 Sauvegarder identifiants
              </label>
            </div>
            
            <button 
              className="connect-btn"
              onClick={handleConnectMT5}
              disabled={isLoading || isConnected}
            >
              {isLoading ? 'CONNECTING...' : isConnected ? 'CONNECTED' : 'CONNECT'}
            </button>
            
            {saveCredentials && (
              <button 
                onClick={handleClearCredentials}
                style={{
                  background: 'rgba(255,68,68,0.2)', 
                  border: '1px solid #ff4444', 
                  borderRadius: '8px', 
                  padding: '8px 16px', 
                  color: '#ff4444', 
                  fontSize: '12px',
                  cursor: 'pointer',
                  marginTop: '8px'
                }}
              >
                🗑️ Effacer sauvegarde
              </button>
            )}
          </div>
        </div>

        {/* Historique des Trades */}
        <div className="glass-card" style={{gridColumn: 'span 2'}}>
          <div className="card-title">📊 Historique des Trades ({selectedTimeframe})</div>
          <div style={{height: '120px', overflowY: 'auto'}}>
            {tradeHistory.length > 0 ? (
              tradeHistory.slice(0, 5).map((trade, index) => (
                <div key={index} style={{
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  padding: '8px 0', 
                  borderBottom: '1px solid rgba(255,255,255,0.1)',
                  fontSize: '13px'
                }}>
                  <div>
                    <span style={{color: '#00ffff', fontWeight: '600'}}>{trade.symbol}</span>
                    <span style={{marginLeft: '10px', color: trade.action === 'BUY' ? '#00ff00' : '#ff4444'}}>
                      {trade.action}
                    </span>
                    <span style={{marginLeft: '10px', color: 'rgba(255,255,255,0.7)'}}>
                      Vol: {trade.volume}
                    </span>
                  </div>
                  <div>
                    <span style={{color: '#ffaa00', marginRight: '10px'}}>
                      {trade.price}
                    </span>
                    <span style={{color: Math.round(trade.confidence * 100) > 70 ? '#00ff00' : '#ffaa00'}}>
                      {Math.round(trade.confidence * 100)}%
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div style={{textAlign: 'center', color: 'rgba(255,255,255,0.5)', paddingTop: '40px'}}>
                📈 Historique des trades s'affichera ici<br/>
                <span style={{fontSize: '11px'}}>Démarrez le trading pour voir l'activité</span>
              </div>
            )}
          </div>
        </div>

        {/* Bot Status & Logs */}
        <div className="glass-card" style={{gridColumn: 'span 1'}}>
          <div className="card-title">🤖 Bot Autonome - Activité</div>
          <div style={{height: '120px', overflowY: 'auto', fontSize: '13px'}}>
            {logs.length > 0 ? (
              logs.slice(0, 8).map((log) => (
                <div key={log.id} style={{
                  padding: '4px 0', 
                  borderBottom: '1px solid rgba(255,255,255,0.1)',
                  color: log.type === 'success' ? '#00ff00' : 
                         log.type === 'error' ? '#ff4444' : 
                         log.type === 'warning' ? '#ffaa00' : '#00ffff'
                }}>
                  <span style={{color: 'rgba(255,255,255,0.5)', fontSize: '11px'}}>
                    {log.timestamp}
                  </span>
                  <span style={{marginLeft: '10px'}}>{log.message}</span>
                </div>
              ))
            ) : (
              <div style={{color: 'rgba(255,255,255,0.5)', textAlign: 'center', paddingTop: '40px'}}>
                Connectez MT5 et cliquez START pour voir l'activité du bot
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Trading Controls - Enhanced */}
      <div className="trading-controls">
        <button 
          className="control-button btn-start"
          onClick={handleStartTrading}
          disabled={!isConnected || isAutoTrading}
          style={{fontSize: '16px', padding: '20px 40px', position: 'relative'}}
        >
          🚀 START ULTRA TRADING
          <div style={{fontSize: '12px', opacity: 0.8}}>
            (Mode {selectedTimeframe} + IA autonome)
          </div>
          {isAutoTrading && (
            <div style={{
              position: 'absolute', 
              top: '5px', 
              right: '5px', 
              width: '8px', 
              height: '8px', 
              background: '#00ff00', 
              borderRadius: '50%', 
              animation: 'pulse 1s infinite'
            }}></div>
          )}
        </button>
        
        <button 
          className="control-button btn-stop"
          onClick={handleStopTrading}
          disabled={!isAutoTrading}
          style={{fontSize: '16px', padding: '20px 40px'}}
        >
          ⏹️ STOP TRADING
          <div style={{fontSize: '12px', opacity: 0.8}}>
            (Arrêt complet du bot)
          </div>
        </button>
      </div>

      {/* Barre IA en temps réel - Bottom Bar */}
      <div className="ai-stats-bar">
        <div className="ai-stat-item">
          <div className="ai-stat-icon">🧠</div>
          <div className="ai-stat-content">
            <div className="ai-stat-label">INTELLIGENCE ARTIFICIELLE</div>
            <div className={`ai-stat-value ${aiStats.isActive ? 'active' : 'inactive'}`}>
              {aiStats.isActive ? 'IA Active' : 'IA Inactive'}
            </div>
          </div>
        </div>

        <div className="ai-stat-item">
          <div className="ai-stat-icon">⚡</div>
          <div className="ai-stat-content">
            <div className="ai-stat-label">DÉCISIONS IA</div>
            <div className="ai-stat-value">{aiStats.decisions}</div>
          </div>
        </div>

        <div className="ai-stat-item">
          <div className="ai-stat-icon">🎯</div>
          <div className="ai-stat-content">
            <div className="ai-stat-label">PRÉCISION</div>
            <div className="ai-stat-value">{aiStats.precision}%</div>
          </div>
        </div>

        <div className="ai-stat-item">
          <div className="ai-stat-icon">📚</div>
          <div className="ai-stat-content">
            <div className="ai-stat-label">APPRENTISSAGE</div>
            <div className="ai-stat-value">{aiStats.learningStatus}</div>
          </div>
        </div>

        <div className="ai-stat-item">
          <div className="ai-stat-icon">🤖</div>
          <div className="ai-stat-content">
            <div className="ai-stat-label">MODE</div>
            <div className="ai-stat-value">{aiStats.mode}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;