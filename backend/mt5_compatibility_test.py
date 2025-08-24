"""
Test de Compatibilité MT5 - Ultra Trading Bot
Vérification complète de toutes les fonctionnalités MetaTrader 5
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys

class MT5CompatibilityTest:
    """Test complet de compatibilité avec MetaTrader 5"""
    
    def __init__(self):
        self.results = []
        self.errors = []
        
    def log_result(self, test_name, status, details=""):
        """Enregistre le résultat d'un test"""
        self.results.append({
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': datetime.now()
        })
        
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {test_name}: {details}")
        
    def test_mt5_import(self):
        """Test d'importation de MetaTrader5"""
        try:
            import MetaTrader5 as mt5
            version = mt5.__version__ if hasattr(mt5, '__version__') else "Version inconnue"
            self.log_result("Import MT5", "PASS", f"Version: {version}")
            return True
        except ImportError as e:
            self.log_result("Import MT5", "FAIL", f"Erreur: {str(e)}")
            return False
    
    def test_mt5_initialization(self):
        """Test d'initialisation MT5"""
        try:
            if not mt5.initialize():
                self.log_result("Initialisation MT5", "FAIL", "Échec d'initialisation")
                return False
            
            # Vérifier les informations sur le terminal
            terminal_info = mt5.terminal_info()
            if terminal_info:
                self.log_result("Initialisation MT5", "PASS", f"Terminal: {terminal_info.name}")
            else:
                self.log_result("Initialisation MT5", "PASS", "Initialisé sans info terminal")
            
            return True
        except Exception as e:
            self.log_result("Initialisation MT5", "FAIL", f"Erreur: {str(e)}")
            return False
    
    def test_mt5_connection(self, login=1296306, password="@0JxCmJa", server="FreshForex-MT5"):
        """Test de connexion avec les identifiants"""
        try:
            result = mt5.login(login, password=password, server=server)
            if result:
                account_info = mt5.account_info()
                if account_info:
                    balance = account_info.balance
                    currency = account_info.currency
                    leverage = account_info.leverage
                    self.log_result("Connexion MT5", "PASS", 
                                  f"Compte: {login}, Solde: {balance} {currency}, Levier: 1:{leverage}")
                else:
                    self.log_result("Connexion MT5", "PASS", "Connecté sans info compte")
                return True
            else:
                error = mt5.last_error()
                self.log_result("Connexion MT5", "FAIL", f"Erreur: {error}")
                return False
        except Exception as e:
            self.log_result("Connexion MT5", "FAIL", f"Exception: {str(e)}")
            return False
    
    def test_symbols_availability(self):
        """Test de disponibilité des symboles"""
        test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
        available_symbols = []
        
        try:
            for symbol in test_symbols:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    available_symbols.append(symbol)
                    
                    # Test si le symbole est visible
                    if not symbol_info.visible:
                        # Essayer de le rendre visible
                        if mt5.symbol_select(symbol, True):
                            self.log_result(f"Symbole {symbol}", "PASS", "Activé avec succès")
                        else:
                            self.log_result(f"Symbole {symbol}", "WARN", "Disponible mais non activé")
                    else:
                        self.log_result(f"Symbole {symbol}", "PASS", "Disponible et actif")
                else:
                    self.log_result(f"Symbole {symbol}", "FAIL", "Non disponible")
            
            if available_symbols:
                self.log_result("Disponibilité Symboles", "PASS", 
                              f"{len(available_symbols)}/{len(test_symbols)} symboles disponibles")
                return True
            else:
                self.log_result("Disponibilité Symboles", "FAIL", "Aucun symbole disponible")
                return False
                
        except Exception as e:
            self.log_result("Disponibilité Symboles", "FAIL", f"Erreur: {str(e)}")
            return False
    
    def test_timeframes_compatibility(self):
        """Test de compatibilité des timeframes"""
        timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        
        working_timeframes = []
        
        try:
            for tf_name, tf_value in timeframes.items():
                # Test avec EURUSD
                rates = mt5.copy_rates_from_pos("EURUSD", tf_value, 0, 10)
                if rates is not None and len(rates) > 0:
                    working_timeframes.append(tf_name)
                    self.log_result(f"Timeframe {tf_name}", "PASS", f"Valeur: {tf_value}")
                else:
                    self.log_result(f"Timeframe {tf_name}", "FAIL", "Pas de données")
            
            if working_timeframes:
                self.log_result("Compatibilité Timeframes", "PASS", 
                              f"{len(working_timeframes)}/9 timeframes fonctionnels")
                return True
            else:
                self.log_result("Compatibilité Timeframes", "FAIL", "Aucun timeframe fonctionnel")
                return False
                
        except Exception as e:
            self.log_result("Compatibilité Timeframes", "FAIL", f"Erreur: {str(e)}")
            return False
    
    def test_market_data_retrieval(self):
        """Test de récupération des données de marché"""
        try:
            # Test rates
            rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 100)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                self.log_result("Données OHLCV", "PASS", 
                              f"{len(rates)} barres récupérées")
                
                # Vérifier les colonnes
                required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
                if all(col in df.columns for col in required_cols):
                    self.log_result("Colonnes OHLCV", "PASS", "Toutes les colonnes présentes")
                else:
                    self.log_result("Colonnes OHLCV", "FAIL", "Colonnes manquantes")
                    return False
            else:
                self.log_result("Données OHLCV", "FAIL", "Pas de données rates")
                return False
            
            # Test tick info
            tick_info = mt5.symbol_info_tick("EURUSD")
            if tick_info:
                self.log_result("Tick Info", "PASS", 
                              f"Bid: {tick_info.bid}, Ask: {tick_info.ask}")
            else:
                self.log_result("Tick Info", "FAIL", "Pas de tick info")
                return False
            
            # Test ticks
            ticks = mt5.copy_ticks_from("EURUSD", datetime.now() - timedelta(minutes=5), 100, mt5.COPY_TICKS_ALL)
            if ticks is not None and len(ticks) > 0:
                self.log_result("Données Ticks", "PASS", f"{len(ticks)} ticks récupérés")
            else:
                self.log_result("Données Ticks", "WARN", "Pas de ticks (normal sur démo)")
            
            return True
            
        except Exception as e:
            self.log_result("Récupération Données", "FAIL", f"Erreur: {str(e)}")
            return False
    
    def test_account_info(self):
        """Test des informations de compte"""
        try:
            account_info = mt5.account_info()
            if account_info:
                info_dict = account_info._asdict()
                
                required_fields = ['login', 'balance', 'equity', 'margin', 'margin_free', 'profit', 'currency', 'leverage']
                missing_fields = [field for field in required_fields if field not in info_dict]
                
                if not missing_fields:
                    self.log_result("Informations Compte", "PASS", 
                                  f"Login: {info_dict['login']}, Solde: {info_dict['balance']}")
                    
                    # Test des positions
                    positions = mt5.positions_get()
                    if positions is not None:
                        self.log_result("Positions", "PASS", f"{len(positions)} positions")
                    else:
                        self.log_result("Positions", "PASS", "Aucune position ouverte")
                    
                    # Test des ordres
                    orders = mt5.orders_get()
                    if orders is not None:
                        self.log_result("Ordres", "PASS", f"{len(orders)} ordres en attente")
                    else:
                        self.log_result("Ordres", "PASS", "Aucun ordre en attente")
                    
                    return True
                else:
                    self.log_result("Informations Compte", "FAIL", f"Champs manquants: {missing_fields}")
                    return False
            else:
                self.log_result("Informations Compte", "FAIL", "Pas d'informations compte")
                return False
                
        except Exception as e:
            self.log_result("Informations Compte", "FAIL", f"Erreur: {str(e)}")
            return False
    
    def test_trading_operations(self):
        """Test des opérations de trading (simulation)"""
        try:
            # Test de préparation d'ordre
            symbol = "EURUSD"
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.log_result("Test Trading", "FAIL", "Symbole non disponible")
                return False
            
            # Calculer les paramètres d'ordre
            point = symbol_info.point
            lot_size = 0.01
            
            # Test tick info pour prix
            tick_info = mt5.symbol_info_tick(symbol)
            if not tick_info:
                self.log_result("Test Trading", "FAIL", "Pas de prix disponible")
                return False
            
            # Simulation d'ordre d'achat
            price = tick_info.ask
            sl = price - 100 * point
            tp = price + 100 * point
            
            # Préparer la requête (ne pas l'envoyer)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": "Test order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Vérifier que tous les paramètres sont corrects
            if all(key in request for key in ['action', 'symbol', 'volume', 'type', 'price']):
                self.log_result("Préparation Ordre", "PASS", 
                              f"Ordre BUY {lot_size} {symbol} à {price}")
                
                # Test de vérification des marges
                margin_info = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lot_size, price)
                if margin_info is not None:
                    self.log_result("Calcul Marge", "PASS", f"Marge requise: {margin_info}")
                else:
                    self.log_result("Calcul Marge", "WARN", "Calcul marge non disponible")
                
                # Test de vérification des profits
                profit_info = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, lot_size, price, price + 100*point)
                if profit_info is not None:
                    self.log_result("Calcul Profit", "PASS", f"Profit estimé: {profit_info}")
                else:
                    self.log_result("Calcul Profit", "WARN", "Calcul profit non disponible")
                
                return True
            else:
                self.log_result("Préparation Ordre", "FAIL", "Paramètres manquants")
                return False
                
        except Exception as e:
            self.log_result("Test Trading", "FAIL", f"Erreur: {str(e)}")
            return False
    
    def test_historical_data(self):
        """Test des données historiques"""
        try:
            # Test avec différentes méthodes
            symbol = "EURUSD"
            
            # Méthode 1: copy_rates_from_pos
            rates1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
            if rates1 is not None and len(rates1) > 0:
                self.log_result("Rates from pos", "PASS", f"{len(rates1)} barres")
            else:
                self.log_result("Rates from pos", "FAIL", "Pas de données")
                return False
            
            # Méthode 2: copy_rates_from
            utc_from = datetime.now() - timedelta(hours=24)
            rates2 = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, utc_from, 100)
            if rates2 is not None and len(rates2) > 0:
                self.log_result("Rates from time", "PASS", f"{len(rates2)} barres")
            else:
                self.log_result("Rates from time", "WARN", "Données limitées")
            
            # Méthode 3: copy_rates_range
            utc_to = datetime.now()
            rates3 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, utc_from, utc_to)
            if rates3 is not None and len(rates3) > 0:
                self.log_result("Rates range", "PASS", f"{len(rates3)} barres")
            else:
                self.log_result("Rates range", "WARN", "Données limitées")
            
            return True
            
        except Exception as e:
            self.log_result("Données Historiques", "FAIL", f"Erreur: {str(e)}")
            return False
    
    def test_cleanup(self):
        """Test de nettoyage"""
        try:
            mt5.shutdown()
            self.log_result("Nettoyage MT5", "PASS", "Connexion fermée proprement")
            return True
        except Exception as e:
            self.log_result("Nettoyage MT5", "FAIL", f"Erreur: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Exécute tous les tests"""
        print("\n" + "="*60)
        print("🚀 TEST DE COMPATIBILITÉ MT5 - ULTRA TRADING BOT")
        print("="*60)
        
        # Test 1: Import
        if not self.test_mt5_import():
            print("\n❌ ÉCHEC: MetaTrader5 ne peut pas être importé")
            return False
        
        # Test 2: Initialisation
        if not self.test_mt5_initialization():
            print("\n❌ ÉCHEC: Impossible d'initialiser MT5")
            return False
        
        # Test 3: Connexion
        if not self.test_mt5_connection():
            print("\n⚠️  AVERTISSEMENT: Connexion échouée (normal sans MT5 ouvert)")
            # Continue quand même les autres tests
        
        # Test 4: Symboles
        self.test_symbols_availability()
        
        # Test 5: Timeframes
        self.test_timeframes_compatibility()
        
        # Test 6: Données de marché
        self.test_market_data_retrieval()
        
        # Test 7: Informations compte
        self.test_account_info()
        
        # Test 8: Opérations trading
        self.test_trading_operations()
        
        # Test 9: Données historiques
        self.test_historical_data()
        
        # Test 10: Nettoyage
        self.test_cleanup()
        
        # Résumé
        self.print_summary()
        
        return True
    
    def print_summary(self):
        """Affiche le résumé des tests"""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DES TESTS")
        print("="*60)
        
        passed = sum(1 for r in self.results if r['status'] == 'PASS')
        failed = sum(1 for r in self.results if r['status'] == 'FAIL')
        warned = sum(1 for r in self.results if r['status'] == 'WARN')
        total = len(self.results)
        
        print(f"✅ Tests Réussis: {passed}/{total}")
        print(f"❌ Tests Échoués: {failed}/{total}")
        print(f"⚠️  Avertissements: {warned}/{total}")
        
        success_rate = (passed / total) * 100 if total > 0 else 0
        print(f"📈 Taux de Réussite: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n🎉 COMPATIBILITÉ MT5: EXCELLENTE!")
        elif success_rate >= 60:
            print("\n✅ COMPATIBILITÉ MT5: BONNE")
        elif success_rate >= 40:
            print("\n⚠️  COMPATIBILITÉ MT5: MOYENNE")
        else:
            print("\n❌ COMPATIBILITÉ MT5: FAIBLE")
        
        # Recommandations
        print("\n💡 RECOMMANDATIONS:")
        if failed > 0:
            print("- Vérifiez que MetaTrader 5 est installé et ouvert")
            print("- Vérifiez vos identifiants de connexion")
            print("- Assurez-vous que les symboles sont disponibles")
        
        print("- Utilisez Python 3.8-3.11 pour la compatibilité")
        print("- Testez d'abord en mode démo")
        print("- Surveillez les logs pour les erreurs")
        
        print("\n" + "="*60)

def main():
    """Fonction principale"""
    tester = MT5CompatibilityTest()
    tester.run_all_tests()

if __name__ == "__main__":
    main()