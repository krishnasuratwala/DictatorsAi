import requests
import hmac
import hashlib
import json
import logging
import os

# In production, load these from os.environ
BTCPAY_URL = os.getenv("BTCPAY_URL")
STORE_ID = os.getenv("BTCPAY_STORE_ID")
API_KEY = os.getenv("BTCPAY_API_KEY")
# Load secret from env (Production) or leave None (Insecure/Test)
WEBHOOK_SECRET = os.getenv("BTCPAY_WEBHOOK_SECRET", None)

def create_invoice(amount, currency, metadata=None):
    """
    Creates an invoice on BTCPay Server.
    metadata: dict containing custom fields (e.g. {'userId': '123', 'plan': 'commander'})
    """
    url = f"{BTCPAY_URL}/api/v1/stores/{STORE_ID}/invoices"
    headers = {
        "Authorization": f"token {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "amount": str(amount),
        "currency": currency,
        "metadata": metadata or {},
        "checkout": {
            "speedPolicy": "HighSpeed", # Recommend for digital goods
            "redirectAutomatically": True,
           # "redirectAutomatically": True,
            "redirectURL": os.getenv("FRONTEND_URL", "https://frontend-4ok4.vercel.app/")
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"BTCPay Invoice Creation Failed: {e}")
        if hasattr(e, 'response') and e.response:
            logging.error(f"Response: {e.response.text}")
        raise e

def verify_webhook_signature(payload_body, sig_header, secret):
    """
    Verifies that the webhook came from BTCPay Server.
    sig_header: The 'BTCPay-Sig' header value.
    secret: The webhook secret configured in BTCPay.
    """
    if not secret:
        logging.warning("Webhook received but no secret configured. Skipping signature verification (INSECURE).")
        return True 
        
    computed_sig = "sha256=" + hmac.new(
        key=secret.encode('utf-8'),
        msg=payload_body,
        digestmod=hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(computed_sig, sig_header)
