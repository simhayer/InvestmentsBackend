"""
Optional verification of Plaid webhooks using the Plaid-Verification JWT header.

When PLAID_WEBHOOK_VERIFY is enabled, the webhook handler rejects requests that
fail signature or body-hash verification (recommended for production).
"""

import hashlib
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Optional: use PyJWT + JWK for verification. If not available, verification is skipped.
try:
    import jwt  # noqa: F401
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False

from services.plaid.plaid_config import client

# In-memory cache: kid -> { "key": jwk_dict, "exp": expiry_timestamp }
_key_cache: dict = {}
_KEY_CACHE_TTL = 300  # 5 minutes


def _decode_jwt_unverified(token: str) -> tuple[dict, dict]:
    """Decode JWT header and payload without verifying. Returns (header, payload)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT structure")
        import base64
        def b64_decode(s: str) -> bytes:
            pad = 4 - len(s) % 4
            if pad != 4:
                s += "=" * pad
            return base64.urlsafe_b64decode(s)
        header = json.loads(b64_decode(parts[0]).decode())
        payload = json.loads(b64_decode(parts[1]).decode())
        return header, payload
    except Exception as e:
        logger.warning("Failed to decode Plaid webhook JWT: %s", e)
        raise


def _get_verification_key(key_id: str) -> Optional[dict]:
    """Fetch webhook verification key from Plaid (with cache). Returns JWK dict or None."""
    now = time.time()
    if key_id in _key_cache and _key_cache[key_id]["exp"] > now:
        return _key_cache[key_id]["key"]
    try:
        from plaid.model.webhook_verification_key_get_request import WebhookVerificationKeyGetRequest
        req = WebhookVerificationKeyGetRequest(key_id=key_id)
        resp = client.webhook_verification_key_get(req)
        key = resp.key
        # Convert to dict if it's an object with attributes
        if hasattr(key, "to_dict"):
            key_dict = key.to_dict()
        elif hasattr(key, "__dict__"):
            key_dict = {k: v for k, v in key.__dict__.items() if not k.startswith("_")}
        else:
            key_dict = dict(key)
        _key_cache[key_id] = {"key": key_dict, "exp": now + _KEY_CACHE_TTL}
        return key_dict
    except ImportError:
        # SDK may not have this endpoint in older versions; try direct API call
        pass
    except Exception as e:
        logger.warning("Failed to fetch Plaid webhook verification key: %s", e)
        return None

    # Fallback: call Plaid API directly (when SDK has no webhook_verification_key_get)
    try:
        import httpx
        client_id = os.getenv("PLAID_CLIENT_ID")
        secret = os.getenv("PLAID_SECRET")
        host = getattr(client.api_client.configuration, "host", None)
        if host is None or not isinstance(host, str):
            host = "https://sandbox.plaid.com" if os.getenv("PLAID_ENV", "sandbox").strip().lower() != "production" else "https://production.plaid.com"
        base_url = host if host.startswith("http") else f"https://{host}"
        if not client_id or not secret:
            return None
        url = f"{base_url}/webhook_verification_key/get"
        with httpx.Client() as http:
            r = http.post(
                url,
                json={"key_id": key_id},
                headers={"PLAID-CLIENT-ID": client_id, "PLAID-SECRET": secret},
                timeout=10.0,
            )
        if r.status_code != 200:
            return None
        data = r.json()
        key_dict = data.get("key")
        if key_dict:
            _key_cache[key_id] = {"key": key_dict, "exp": now + _KEY_CACHE_TTL}
        return key_dict
    except Exception as e:
        logger.warning("Direct Plaid webhook key fetch failed: %s", e)
        return None


def verify_plaid_webhook(body: bytes, plaid_verification_header: str) -> bool:
    """
    Verify Plaid webhook using Plaid-Verification JWT and body SHA-256.
    Returns True if verification passes, False otherwise.
    """
    if not _JWT_AVAILABLE:
        logger.warning("PyJWT not installed; cannot verify Plaid webhook. Install PyJWT and cryptography.")
        return False
    if not plaid_verification_header or not plaid_verification_header.strip():
        return False
    try:
        header, payload = _decode_jwt_unverified(plaid_verification_header)
        if header.get("alg") != "ES256":
            logger.warning("Plaid webhook JWT alg is not ES256: %s", header.get("alg"))
            return False
        kid = header.get("kid")
        if not kid:
            return False
        key_dict = _get_verification_key(kid)
        if not key_dict:
            return False
        # Verify JWT signature and iat (max 5 min old)
        try:
            import jwt as jwt_lib
            # Build public key from JWK for ES256
            from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicNumbers
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            import base64

            def b64url_decode(s: str) -> int:
                pad = 4 - len(s) % 4
                if pad != 4:
                    s += "=" * pad
                return int.from_bytes(base64.urlsafe_b64decode(s), "big")

            x = b64url_decode(key_dict["x"])
            y = b64url_decode(key_dict["y"])
            from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1
            numbers = EllipticCurvePublicNumbers(x, y, SECP256R1())
            pub_key = numbers.public_key(default_backend())
            decoded = jwt_lib.decode(
                plaid_verification_header,
                pub_key,
                algorithms=["ES256"],
                options={"verify_exp": False},
                leeway=0,
            )
            iat = decoded.get("iat")
            if iat is not None and (time.time() - iat) > 300:
                logger.warning("Plaid webhook JWT too old (iat=%s)", iat)
                return False
        except Exception as e:
            logger.warning("Plaid webhook JWT signature verification failed: %s", e)
            return False
        # Compare body SHA-256 (Plaid uses raw body, 2-space JSON)
        body_sha = hashlib.sha256(body).hexdigest()
        claimed = payload.get("request_body_sha256", "")
        if not claimed or not body_sha:
            return False
        # Constant-time comparison
        if len(claimed) != len(body_sha):
            return False
        result = 0
        for a, b in zip(claimed.encode(), body_sha.encode()):
            result |= a ^ b
        return result == 0
    except Exception as e:
        logger.warning("Plaid webhook verification error: %s", e)
        return False


def is_webhook_verification_enabled() -> bool:
    """Require explicit opt-in. When webhook URL is localhost, verification is skipped so production keys work locally."""
    if os.getenv("PLAID_WEBHOOK_VERIFY", "").strip().lower() not in ("1", "true", "yes"):
        return False
    webhook_url = (os.getenv("PLAID_WEBHOOK_URL") or "").strip().lower()
    if "localhost" in webhook_url or "127.0.0.1" in webhook_url:
        logger.debug("Plaid webhook URL is local; skipping verification so production can be used locally")
        return False
    return True
