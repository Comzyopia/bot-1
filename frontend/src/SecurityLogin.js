import React, { useState, useEffect } from 'react';

const SecurityLogin = ({ onLoginSuccess }) => {
  const [step, setStep] = useState('login'); // 'login', '2fa', 'register'
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [totpCode, setTotpCode] = useState('');
  const [qrCodeData, setQrCodeData] = useState('');
  const [manualKey, setManualKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleRegister = async () => {
    if (!userId || !password) {
      setError('User ID et mot de passe requis');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, password })
      });

      const data = await response.json();

      if (data.success) {
        setQrCodeData(data['2fa_setup'].qr_code);
        setManualKey(data['2fa_setup'].manual_key);
        setStep('2fa_setup');
      } else {
        setError('Erreur d\'enregistrement');
      }
    } catch (error) {
      setError('Erreur de connexion');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async () => {
    if (!userId || !password || !totpCode) {
      setError('Tous les champs sont requis');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          password,
          totp_code: totpCode
        })
      });

      const data = await response.json();

      if (data.success) {
        localStorage.setItem('auth_token', data.access_token);
        onLoginSuccess(data.access_token);
      } else {
        setError(data.detail || 'Échec de la connexion');
      }
    } catch (error) {
      setError('Erreur de connexion');
    } finally {
      setIsLoading(false);
    }
  };

  if (step === '2fa_setup') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex items-center justify-center p-6">
        <div className="bg-gray-800 rounded-2xl p-8 max-w-md w-full border border-gray-700">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-white text-2xl">🛡️</span>
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">Configuration 2FA</h2>
            <p className="text-gray-400">Scannez le QR code avec votre app d'authentification</p>
          </div>

          <div className="space-y-6">
            <div className="text-center">
              {qrCodeData && (
                <img src={qrCodeData} alt="QR Code 2FA" className="mx-auto mb-4 rounded-lg" />
              )}
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">Clé manuelle :</label>
              <div className="bg-gray-700 p-3 rounded-lg text-center">
                <code className="text-green-400 text-sm break-all">{manualKey}</code>
              </div>
            </div>

            <button
              onClick={() => setStep('login')}
              className="w-full py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium transition-colors"
            >
              Continuer vers la connexion
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-black flex items-center justify-center p-6">
      <div className="bg-gray-800 rounded-2xl p-8 max-w-md w-full border border-gray-700">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-white text-2xl">🏆</span>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">Ultra Trading Bot</h1>
          <p className="text-gray-400">Connexion sécurisée avec 2FA</p>
        </div>

        {error && (
          <div className="bg-red-600/20 border border-red-600 rounded-lg p-3 mb-6">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">User ID</label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none"
              placeholder="Votre identifiant"
            />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">Mot de passe</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none"
              placeholder="Votre mot de passe"
            />
          </div>

          {step === 'login' && (
            <div>
              <label className="block text-sm text-gray-400 mb-2">Code 2FA</label>
              <input
                type="text"
                value={totpCode}
                onChange={(e) => setTotpCode(e.target.value)}
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:border-purple-500 focus:outline-none text-center text-2xl tracking-widest"
                placeholder="000000"
                maxLength="6"
              />
              <p className="text-xs text-gray-500 mt-1">Code depuis votre app d'authentification</p>
            </div>
          )}

          <div className="space-y-3">
            {step === 'login' ? (
              <button
                onClick={handleLogin}
                disabled={isLoading}
                className="w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors"
              >
                {isLoading ? 'Connexion...' : '🔐 Se connecter'}
              </button>
            ) : (
              <button
                onClick={handleRegister}
                disabled={isLoading}
                className="w-full py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors"
              >
                {isLoading ? 'Enregistrement...' : '📝 S\'enregistrer'}
              </button>
            )}

            <button
              onClick={() => setStep(step === 'login' ? 'register' : 'login')}
              className="w-full py-2 text-gray-400 hover:text-white transition-colors"
            >
              {step === 'login' ? 'Créer un compte' : 'Déjà un compte ? Se connecter'}
            </button>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-gray-700">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-4 text-xs text-gray-500">
              <div className="flex items-center">
                <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                AES-256 Encryption
              </div>
              <div className="flex items-center">
                <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                2FA Protection
              </div>
              <div className="flex items-center">
                <span className="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
                Secure Sessions
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SecurityLogin;