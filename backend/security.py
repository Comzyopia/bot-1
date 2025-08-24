"""
🛡️ Module de Sécurité Ultra-Avancé
Chiffrement AES-256, 2FA, Protection API
"""

import os
import hashlib
import secrets
import base64
import json
import time
import pyotp
import qrcode
from io import BytesIO
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from functools import wraps
import jwt
from datetime import datetime, timedelta
import logging
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logging.warning("Email functionality not available")

# Configuration sécurisée
MASTER_KEY = os.environ.get('ULTRA_MASTER_KEY', secrets.token_hex(32))
JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))
SESSION_TIMEOUT = 3600  # 1 heure

class UltraSecurityManager:
    """🛡️ Gestionnaire de sécurité ultra-avancé"""
    
    def __init__(self):
        self.active_sessions = {}
        self.failed_attempts = {}
        self.security_logs = []
        self.master_key = MASTER_KEY.encode()
        self.jwt_secret = JWT_SECRET
        
    def generate_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Générer une clé de chiffrement à partir du mot de passe"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # 100k itérations pour sécurité maximale
            backend=default_backend()
        )
        return kdf.derive(password.encode())
    
    def encrypt_mt5_credentials(self, login: str, password: str, server: str, user_password: str) -> str:
        """🔐 Chiffrement AES-256 des identifiants MT5"""
        try:
            # Créer un salt unique
            salt = os.urandom(16)
            
            # Générer la clé de chiffrement
            key = self.generate_key_from_password(user_password, salt)
            
            # Créer l'IV (Initialization Vector)
            iv = os.urandom(16)
            
            # Préparer les données à chiffrer
            credentials = {
                "login": login,
                "password": password,
                "server": server,
                "timestamp": time.time()
            }
            
            plaintext = json.dumps(credentials).encode()
            
            # Chiffrement AES-256-CBC
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Padding PKCS7
            pad_length = 16 - (len(plaintext) % 16)
            padded_plaintext = plaintext + bytes([pad_length] * pad_length)
            
            ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
            
            # Combiner salt + iv + ciphertext
            encrypted_data = salt + iv + ciphertext
            
            # Encoder en base64 pour stockage
            encrypted_b64 = base64.b64encode(encrypted_data).decode()
            
            self.log_security_event("MT5_CREDENTIALS_ENCRYPTED", {"success": True})
            
            return encrypted_b64
            
        except Exception as e:
            self.log_security_event("MT5_ENCRYPTION_ERROR", {"error": str(e)})
            raise Exception("Échec du chiffrement des identifiants")
    
    def decrypt_mt5_credentials(self, encrypted_data: str, user_password: str) -> dict:
        """🔓 Déchiffrement AES-256 des identifiants MT5"""
        try:
            # Décoder depuis base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            
            # Extraire salt, iv, et ciphertext
            salt = encrypted_bytes[:16]
            iv = encrypted_bytes[16:32]
            ciphertext = encrypted_bytes[32:]
            
            # Générer la clé de déchiffrement
            key = self.generate_key_from_password(user_password, salt)
            
            # Déchiffrement AES-256-CBC
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            
            padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Supprimer le padding PKCS7
            pad_length = padded_plaintext[-1]
            plaintext = padded_plaintext[:-pad_length]
            
            # Décoder JSON
            credentials = json.loads(plaintext.decode())
            
            # Vérifier que ce n'est pas trop ancien (24h max)
            if time.time() - credentials.get("timestamp", 0) > 86400:
                raise Exception("Identifiants expirés")
            
            self.log_security_event("MT5_CREDENTIALS_DECRYPTED", {"success": True})
            
            return credentials
            
        except Exception as e:
            self.log_security_event("MT5_DECRYPTION_ERROR", {"error": str(e)})
            raise Exception("Échec du déchiffrement des identifiants")
    
    def generate_2fa_secret(self, user_id: str) -> dict:
        """🔐 Générer un secret 2FA (TOTP)"""
        secret = pyotp.random_base32()
        
        # Créer l'URI pour QR Code
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_id,
            issuer_name="Ultra Trading Bot 🏆"
        )
        
        # Générer QR Code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convertir en base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        qr_code_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        self.log_security_event("2FA_SECRET_GENERATED", {"user_id": user_id})
        
        return {
            "secret": secret,
            "qr_code": qr_code_b64,
            "manual_entry_key": secret
        }
    
    def verify_2fa_token(self, secret: str, token: str) -> bool:
        """✅ Vérifier un code 2FA"""
        try:
            totp = pyotp.TOTP(secret)
            is_valid = totp.verify(token, valid_window=1)  # ±30 secondes
            
            self.log_security_event("2FA_VERIFICATION", {
                "success": is_valid,
                "timestamp": time.time()
            })
            
            return is_valid
            
        except Exception as e:
            self.log_security_event("2FA_ERROR", {"error": str(e)})
            return False
    
    def generate_session_token(self, user_id: str, additional_claims: dict = None) -> str:
        """🎫 Générer un token de session JWT"""
        try:
            payload = {
                "user_id": user_id,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(seconds=SESSION_TIMEOUT),
                "jti": secrets.token_hex(16),  # Unique token ID
                "permissions": ["trading", "api_access", "ai_control"]
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            
            # Stocker la session active
            self.active_sessions[payload["jti"]] = {
                "user_id": user_id,
                "created_at": time.time(),
                "last_activity": time.time()
            }
            
            self.log_security_event("SESSION_CREATED", {
                "user_id": user_id,
                "token_id": payload["jti"],
                "expires_at": payload["exp"].isoformat()
            })
            
            return token
            
        except Exception as e:
            self.log_security_event("SESSION_ERROR", {"error": str(e)})
            raise Exception("Échec de création de session")
    
    def verify_session_token(self, token: str) -> dict:
        """🔍 Vérifier un token de session"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            token_id = payload.get("jti")
            
            # Vérifier que la session est active
            if token_id not in self.active_sessions:
                raise jwt.InvalidTokenError("Session expirée")
            
            # Mettre à jour l'activité
            self.active_sessions[token_id]["last_activity"] = time.time()
            
            self.log_security_event("SESSION_VERIFIED", {
                "user_id": payload["user_id"],
                "token_id": token_id
            })
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self.log_security_event("SESSION_EXPIRED", {"token": token[:20] + "..."})
            raise Exception("Token expiré")
        except jwt.InvalidTokenError as e:
            self.log_security_event("SESSION_INVALID", {"error": str(e)})
            raise Exception("Token invalide")
    
    def revoke_session(self, token_id: str):
        """🚫 Révoquer une session"""
        if token_id in self.active_sessions:
            user_id = self.active_sessions[token_id]["user_id"]
            del self.active_sessions[token_id]
            
            self.log_security_event("SESSION_REVOKED", {
                "user_id": user_id,
                "token_id": token_id
            })
    
    def check_rate_limit(self, user_id: str, max_attempts: int = 5, window: int = 300) -> bool:
        """⏱️ Vérification du rate limiting"""
        current_time = time.time()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        # Nettoyer les anciennes tentatives
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if current_time - attempt < window
        ]
        
        return len(self.failed_attempts[user_id]) < max_attempts
    
    def record_failed_attempt(self, user_id: str):
        """📝 Enregistrer une tentative échouée"""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append(time.time())
        
        self.log_security_event("FAILED_ATTEMPT", {
            "user_id": user_id,
            "total_attempts": len(self.failed_attempts[user_id])
        })
    
    def log_security_event(self, event_type: str, details: dict):
        """📋 Logger les événements de sécurité"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "ip_hash": hashlib.sha256(str(details).encode()).hexdigest()[:16]
        }
        
        self.security_logs.append(log_entry)
        
        # Garder seulement les 1000 derniers logs
        if len(self.security_logs) > 1000:
            self.security_logs = self.security_logs[-1000:]
        
        # Logger aussi dans le système
        logging.info(f"🛡️ SECURITY: {event_type} - {details}")
    
    def get_security_logs(self, limit: int = 100) -> list:
        """📊 Récupérer les logs de sécurité"""
        return self.security_logs[-limit:]
    
    def hash_password(self, password: str) -> str:
        """🔐 Hasher un mot de passe avec salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """✅ Vérifier un mot de passe hashé"""
        try:
            salt, hash_hex = hashed.split(':')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash.hex() == hash_hex
        except:
            return False

# Instance globale du gestionnaire de sécurité
security_manager = UltraSecurityManager()

def require_auth(f):
    """🛡️ Décorateur pour protection API"""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        from fastapi import Request, HTTPException
        
        # Extraire le token depuis les headers
        request = kwargs.get('request') or args[0] if args else None
        if not request:
            raise HTTPException(status_code=401, detail="No request context")
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Token manquant")
        
        token = auth_header.split(' ')[1]
        
        try:
            payload = security_manager.verify_session_token(token)
            kwargs['current_user'] = payload
            return await f(*args, **kwargs)
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))
    
    return decorated_function

def require_trading_permission(f):
    """🎯 Décorateur pour protection des routes de trading"""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        current_user = kwargs.get('current_user')
        if not current_user or 'trading' not in current_user.get('permissions', []):
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Permission de trading requise")
        
        return await f(*args, **kwargs)
    
    return decorated_function