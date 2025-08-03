import os
from plaid import Configuration, ApiClient, Environment
from plaid.api import plaid_api
from dotenv import load_dotenv

load_dotenv()

configuration = Configuration(
    host=Environment.Sandbox,
    api_key={
        "clientId": os.getenv("PLAID_CLIENT_ID"),
        "secret": os.getenv("PLAID_SECRET"),
    },
)

api_client = ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)