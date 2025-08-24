#!/usr/bin/env python3
"""
🚀 ULTRA TRADING BOT - COMPREHENSIVE BACKEND TEST
═══════════════════════════════════════════════════

Tests complets pour vérifier le fonctionnement des systèmes ultra-avancés:
- Ultra Reward System
- Price Action AI  
- Smart Risk Management
- API endpoints
- MT5 connection
- WebSocket functionality

Version: 3.0 - Ultra Performance Edition
"""

import requests
import json
import sys
import time
import asyncio
import websockets
from datetime import datetime
from typing import Dict, List, Optional

class UltraTradingBotTester:
    def __init__(self, base_url: str = "https://reward-bot.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api"
        self.ws_url = base_url.replace('https', 'wss') + '/api/ws'
        
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
        
        # Configuration MT5 par défaut
        self.mt5_config = {
            "login": 1296306,
            "password": "@0JxCmJa", 
            "server": "FreshForex-MT5"
        }
        
        print("🚀 Ultra Trading Bot Tester initialisé")
        print(f"📡 Backend URL: {self.base_url}")
        print(f"🔌 WebSocket URL: {self.ws_url}")
        print("=" * 60)

    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """Exécute un test et enregistre le résultat"""
        self.tests_run += 1
        print(f"\n🔍 Test {self.tests_run}: {test_name}")
        
        try:
            result = test_func(*args, **kwargs)
            if result:
                self.tests_passed += 1
                print(f"✅ PASSED - {test_name}")
                return True
            else:
                self.failed_tests.append(test_name)
                print(f"❌ FAILED - {test_name}")
                return False
                
        except Exception as e:
            self.failed_tests.append(f"{test_name}: {str(e)}")
            print(f"❌ ERROR - {test_name}: {str(e)}")
            return False

    def test_api_root(self) -> bool:
        """Test de l'endpoint racine"""
        try:
            response = requests.get(f"{self.api_base}/", timeout=10)
            
            if response.status_code != 200:
                print(f"Status code incorrect: {response.status_code}")
                return False
            
            data = response.json()
            
            # Vérifications spécifiques
            required_fields = ["message", "version", "features"]
            for field in required_fields:
                if field not in data:
                    print(f"Champ manquant: {field}")
                    return False
            
            # Vérification des features ultra-avancées
            features = data.get("features", {})
            ultra_features = [
                "ultra_reward_system",
                "price_action_ai", 
                "smart_risk_management"
            ]
            
            for feature in ultra_features:
                if not features.get(feature, False):
                    print(f"Feature ultra-avancée manquante: {feature}")
                    return False
            
            print(f"Version: {data.get('version')}")
            print(f"Features ultra-avancées: ✅")
            return True
            
        except Exception as e:
            print(f"Erreur: {e}")
            return False

    def test_status_endpoint(self) -> bool:
        """Test de l'endpoint status"""
        try:
            response = requests.get(f"{self.api_base}/status", timeout=10)
            
            if response.status_code != 200:
                print(f"Status code: {response.status_code}")
                return False
            
            data = response.json()
            
            # Vérifications de structure
            required_sections = ["bot_status", "performance_summary", "system_health"]
            for section in required_sections:
                if section not in data:
                    print(f"Section manquante: {section}")
                    return False
            
            # Vérification des systèmes ultra-avancés
            system_health = data.get("system_health", {})
            systems = system_health.get("systems_operational", {})
            
            ultra_systems = ["ultra_reward", "price_action_ai", "risk_manager"]
            for system in ultra_systems:
                if not systems.get(system, False):
                    print(f"Système ultra-avancé non opérationnel: {system}")
                    return False
            
            print("Systèmes ultra-avancés: ✅ Tous opérationnels")
            return True
            
        except Exception as e:
            print(f"Erreur: {e}")
            return False

    def test_mt5_connection(self) -> bool:
        """Test de connexion MT5"""
        try:
            response = requests.post(
                f"{self.api_base}/mt5/connect",
                json=self.mt5_config,
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"Status code: {response.status_code}")
                if response.status_code == 500:
                    error_detail = response.json().get("detail", "Erreur inconnue")
                    print(f"Détail erreur: {error_detail}")
                return False
            
            data = response.json()
            
            if not data.get("success", False):
                print(f"Connexion échouée: {data}")
                return False
            
            # Vérification des systèmes ultra-avancés initialisés
            systems_status = data.get("systems_status", {})
            required_systems = [
                "ultra_reward_system",
                "price_action_ai", 
                "smart_risk_management"
            ]
            
            for system in required_systems:
                if systems_status.get(system) != "✅ Activé":
                    print(f"Système non activé: {system}")
                    return False
            
            account_info = data.get("account_info", {})
            print(f"Compte connecté - Balance: ${account_info.get('balance', 0):.2f}")
            print(f"Login: {account_info.get('login', 'N/A')}")
            print(f"Serveur: {account_info.get('server', 'N/A')}")
            print("Systèmes ultra-avancés: ✅ Tous activés")
            
            return True
            
        except Exception as e:
            print(f"Erreur: {e}")
            return False

    def test_trading_start(self) -> bool:
        """Test de démarrage du trading"""
        try:
            response = requests.post(f"{self.api_base}/trading/start", timeout=10)
            
            if response.status_code != 200:
                print(f"Status code: {response.status_code}")
                error_detail = response.json().get("detail", "Erreur inconnue")
                print(f"Détail: {error_detail}")
                return False
            
            data = response.json()
            
            if not data.get("success", False):
                print(f"Démarrage échoué: {data}")
                return False
            
            # Vérification des systèmes initialisés
            systems = data.get("systems_initialized", [])
            expected_systems = [
                "Système de récompenses révolutionnaire",
                "IA Price Action ultra-avancée",
                "Gestion du risque intelligente"
            ]
            
            for expected in expected_systems:
                if expected not in systems:
                    print(f"Système manquant: {expected}")
                    return False
            
            print("Trading démarré avec tous les systèmes ultra-avancés ✅")
            return True
            
        except Exception as e:
            print(f"Erreur: {e}")
            return False

    def test_signals_generation(self) -> bool:
        """Test de génération de signaux Price Action"""
        try:
            # Attendre un peu pour que les signaux se génèrent
            time.sleep(3)
            
            response = requests.get(f"{self.api_base}/signals", timeout=10)
            
            if response.status_code != 200:
                print(f"Status code: {response.status_code}")
                return False
            
            data = response.json()
            
            # Vérifications de structure
            required_fields = ["signals", "total_signals", "systems_used"]
            for field in required_fields:
                if field not in data:
                    print(f"Champ manquant: {field}")
                    return False
            
            # Vérification des systèmes utilisés
            systems_used = data.get("systems_used", [])
            expected_systems = ["Price Action AI", "Ultra Reward System", "Smart Risk Management"]
            
            for system in expected_systems:
                if system not in systems_used:
                    print(f"Système non utilisé: {system}")
                    return False
            
            signals = data.get("signals", [])
            total_signals = data.get("total_signals", 0)
            
            print(f"Signaux générés: {total_signals}")
            print(f"Systèmes utilisés: {', '.join(systems_used)}")
            
            # Analyse des signaux si présents
            if signals:
                for i, signal in enumerate(signals[:3]):  # Analyse max 3 signaux
                    print(f"  Signal {i+1}: {signal.get('symbol')} {signal.get('action')} "
                          f"(Conf: {signal.get('confidence', 0)*100:.1f}%)")
                    
                    # Vérification des champs Price Action
                    pa_fields = ["pattern", "risk_reward_ratio", "expected_move_pips"]
                    for field in pa_fields:
                        if field not in signal:
                            print(f"Champ Price Action manquant: {field}")
                            return False
            
            return True
            
        except Exception as e:
            print(f"Erreur: {e}")
            return False

    def test_performance_metrics(self) -> bool:
        """Test des métriques de performance ultra-avancées"""
        try:
            response = requests.get(f"{self.api_base}/performance", timeout=10)
            
            if response.status_code != 200:
                print(f"Status code: {response.status_code}")
                return False
            
            data = response.json()
            
            # Vérifications de structure
            required_sections = [
                "system_status",
                "performance_metrics", 
                "risk_metrics",
                "ai_metrics"
            ]
            
            for section in required_sections:
                if section not in data:
                    print(f"Section manquante: {section}")
                    return False
            
            # Vérification du système ultra-avancé
            system_status = data.get("system_status", {})
            if not system_status.get("ultra_mode", False):
                print("Ultra mode non activé")
                return False
            
            systems_active = system_status.get("systems_active", {})
            ultra_systems = ["price_action_ai", "ultra_reward_system", "smart_risk_management"]
            
            for system in ultra_systems:
                if not systems_active.get(system, False):
                    print(f"Système ultra non actif: {system}")
                    return False
            
            # Vérification des métriques
            perf_metrics = data.get("performance_metrics", {})
            risk_metrics = data.get("risk_metrics", {})
            ai_metrics = data.get("ai_metrics", {})
            
            print(f"Ultra Mode: ✅ Actif")
            print(f"Win Rate: {perf_metrics.get('win_rate', 0)*100:.1f}%")
            print(f"Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Price Action Accuracy: {ai_metrics.get('success_rates', {}).get('overall_pattern', 0)*100:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"Erreur: {e}")
            return False

    def test_config_update(self) -> bool:
        """Test de mise à jour de configuration ultra-avancée"""
        try:
            ultra_config = {
                "timeframe": "M5",
                "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
                "risk_level": "moderate",
                "max_positions": 5,
                "auto_trading": False,
                "price_action_mode": True,
                "ultra_reward_system": True,
                "smart_risk_management": True,
                "adaptive_position_sizing": True,
                "multi_timeframe_analysis": True,
                "min_confidence_threshold": 0.75,
                "max_daily_risk": 0.05
            }
            
            response = requests.post(
                f"{self.api_base}/config/update",
                json=ultra_config,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"Status code: {response.status_code}")
                return False
            
            data = response.json()
            
            if not data.get("success", False):
                print(f"Mise à jour échouée: {data}")
                return False
            
            # Vérification de la nouvelle config
            new_config = data.get("new_config", {})
            ultra_systems = new_config.get("ultra_systems", {})
            
            expected_ultra = {
                "reward_system": True,
                "smart_risk": True,
                "adaptive_sizing": True
            }
            
            for system, expected in expected_ultra.items():
                if ultra_systems.get(system) != expected:
                    print(f"Configuration ultra incorrecte: {system}")
                    return False
            
            print("Configuration ultra-avancée mise à jour ✅")
            return True
            
        except Exception as e:
            print(f"Erreur: {e}")
            return False

    def test_trading_stop(self) -> bool:
        """Test d'arrêt du trading"""
        try:
            response = requests.post(f"{self.api_base}/trading/stop", timeout=10)
            
            if response.status_code != 200:
                print(f"Status code: {response.status_code}")
                return False
            
            data = response.json()
            
            if not data.get("success", False):
                print(f"Arrêt échoué: {data}")
                return False
            
            print("Trading arrêté ✅")
            return True
            
        except Exception as e:
            print(f"Erreur: {e}")
            return False

    async def test_websocket_connection(self) -> bool:
        """Test de connexion WebSocket"""
        try:
            print("Connexion WebSocket...")
            
            # Fix: Remove timeout parameter from websockets.connect
            async with websockets.connect(self.ws_url) as websocket:
                print("WebSocket connecté ✅")
                
                # Attendre le message de connexion
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(message)
                    
                    if data.get("type") != "connection_established":
                        print(f"Type de message inattendu: {data.get('type')}")
                        return False
                    
                    # Vérifier les systèmes ultra-avancés
                    systems_status = data.get("systems_status", {})
                    expected_systems = [
                        "ultra_reward_system",
                        "price_action_ai",
                        "smart_risk_management"
                    ]
                    
                    for system in expected_systems:
                        if systems_status.get(system) != "✅ Actif":
                            print(f"Système WebSocket non actif: {system}")
                            return False
                    
                    print("Systèmes ultra-avancés WebSocket: ✅")
                    
                    # Test d'abonnement aux performances
                    subscribe_msg = {"type": "subscribe_performance"}
                    await websocket.send(json.dumps(subscribe_msg))
                    
                    # Attendre la réponse
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    perf_data = json.loads(response)
                    
                    if perf_data.get("type") != "performance_update":
                        print(f"Type de réponse inattendu: {perf_data.get('type')}")
                        return False
                    
                    print("Abonnement performance: ✅")
                    return True
                    
                except asyncio.TimeoutError:
                    print("Timeout lors de la réception des messages WebSocket")
                    return False
                    
        except Exception as e:
            print(f"Erreur WebSocket: {e}")
            return False

    def run_websocket_test(self) -> bool:
        """Wrapper pour le test WebSocket"""
        try:
            return asyncio.run(self.test_websocket_connection())
        except Exception as e:
            print(f"Erreur test WebSocket: {e}")
            return False

    def run_all_tests(self):
        """Exécute tous les tests"""
        print("🚀 DÉBUT DES TESTS ULTRA TRADING BOT")
        print("=" * 60)
        
        # Tests API de base
        self.run_test("API Root Endpoint", self.test_api_root)
        self.run_test("Status Endpoint", self.test_status_endpoint)
        
        # Tests MT5 et systèmes ultra-avancés
        self.run_test("Connexion MT5 + Systèmes Ultra", self.test_mt5_connection)
        
        # Tests de trading
        self.run_test("Démarrage Trading Ultra", self.test_trading_start)
        self.run_test("Génération Signaux Price Action", self.test_signals_generation)
        self.run_test("Métriques Performance Ultra", self.test_performance_metrics)
        
        # Tests de configuration
        self.run_test("Configuration Ultra-Avancée", self.test_config_update)
        
        # Test WebSocket
        self.run_test("WebSocket Temps Réel", self.run_websocket_test)
        
        # Arrêt propre
        self.run_test("Arrêt Trading", self.test_trading_stop)
        
        # Résumé final
        self.print_final_report()

    def print_final_report(self):
        """Affiche le rapport final"""
        print("\n" + "=" * 60)
        print("📊 RAPPORT FINAL DES TESTS")
        print("=" * 60)
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        print(f"Tests exécutés: {self.tests_run}")
        print(f"Tests réussis: {self.tests_passed}")
        print(f"Tests échoués: {len(self.failed_tests)}")
        print(f"Taux de réussite: {success_rate:.1f}%")
        
        if self.failed_tests:
            print("\n❌ TESTS ÉCHOUÉS:")
            for i, failed_test in enumerate(self.failed_tests, 1):
                print(f"  {i}. {failed_test}")
        
        print("\n🎯 SYSTÈMES ULTRA-AVANCÉS TESTÉS:")
        print("  ✅ Ultra Reward System")
        print("  ✅ Price Action AI")
        print("  ✅ Smart Risk Management")
        print("  ✅ WebSocket Temps Réel")
        print("  ✅ Configuration Dynamique")
        
        if success_rate >= 90:
            print(f"\n🚀 RÉSULTAT: EXCELLENT ({success_rate:.1f}%)")
            print("Le bot Ultra Trading est prêt pour la production!")
        elif success_rate >= 75:
            print(f"\n✅ RÉSULTAT: BON ({success_rate:.1f}%)")
            print("Le bot fonctionne bien avec quelques améliorations mineures.")
        elif success_rate >= 50:
            print(f"\n⚠️ RÉSULTAT: MOYEN ({success_rate:.1f}%)")
            print("Des corrections sont nécessaires avant la production.")
        else:
            print(f"\n❌ RÉSULTAT: CRITIQUE ({success_rate:.1f}%)")
            print("Le bot nécessite des corrections majeures.")
        
        print("=" * 60)
        
        return success_rate >= 75  # Retourne True si les tests sont globalement réussis

def main():
    """Fonction principale"""
    print("🚀 ULTRA TRADING BOT - TESTS BACKEND COMPLETS")
    print("Version: 3.0 - Ultra Performance Edition")
    print("=" * 60)
    
    # Initialise le testeur
    tester = UltraTradingBotTester()
    
    # Exécute tous les tests
    success = tester.run_all_tests()
    
    # Code de sortie
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())