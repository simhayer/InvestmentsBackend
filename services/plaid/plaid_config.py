import os
from plaid import Configuration, ApiClient, Environment
from plaid.api import plaid_api
from dotenv import load_dotenv

load_dotenv()

_plaid_env_name = os.getenv("PLAID_ENV", "sandbox").strip().lower()
_ENV_MAP = {
    "sandbox": Environment.Sandbox,
    "production": Environment.Production,
}
_plaid_host = _ENV_MAP.get(_plaid_env_name, Environment.Sandbox)
_client_id = os.getenv("PLAID_CLIENT_ID")
_secret = os.getenv("PLAID_SECRET")


configuration = Configuration(
    host=_plaid_host,
    api_key={
        "clientId": _client_id,
        "secret": _secret,
    },
)

api_client = ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)