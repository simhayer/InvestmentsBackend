import os
from fastapi import APIRouter, HTTPException, Body
from dotenv import load_dotenv
from plaid import Configuration, ApiClient, Environment
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from database import get_db
from models.access_token import UserAccess
from sqlalchemy.orm import Session
from fastapi import Depends
from plaid.model.investments_holdings_get_request import InvestmentsHoldingsGetRequest
from pydantic import BaseModel

# Load .env variables
load_dotenv()

# Configure Plaid client
configuration = Configuration(
    host=Environment.Sandbox,  # ✅ This is now correct
    api_key={
        "clientId": os.getenv("PLAID_CLIENT_ID"),
        "secret": os.getenv("PLAID_SECRET"),
    },
)

api_client = ApiClient(configuration)
client = plaid_api.PlaidApi(api_client)

# Create FastAPI router
router = APIRouter()

class LinkTokenRequest(BaseModel):
    user_id: str

@router.post("/create-link-token")
async def create_link_token(body: LinkTokenRequest):
    user_id = body.user_id
    try:
        request = LinkTokenCreateRequest(
            products=[Products("investments")],
            client_name="Investment Tracker",
            country_codes=[CountryCode("US"), CountryCode("CA")],
            language="en",
            user=LinkTokenCreateRequestUser(client_user_id=user_id),
        )
        response = client.link_token_create(request)
        return {"link_token": response.link_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating link token: {e}")


@router.post("/exchange-token")
async def exchange_token(
    public_token: str = Body(...),
    user_id: str = Body(...),
    db: Session = Depends(get_db),
):
    try:
        request = ItemPublicTokenExchangeRequest(public_token=public_token)
        response = client.item_public_token_exchange(request)
        access_token = response.access_token
        item_id = response.item_id

        # Upsert the token for the user
        token_entry = db.query(UserAccess).filter_by(user_id=user_id).first()
        if token_entry:
            token_entry.access_token = access_token
            token_entry.item_id = item_id
        else:
            token_entry = UserAccess(
                user_id=user_id,
                access_token=access_token,
                item_id=item_id
            )
            db.add(token_entry)

        db.commit()

        return {"access_token": access_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {e}")
    
@router.get("/investments")
async def get_investments(user_id: str, db: Session = Depends(get_db)):
    try:
        token_entry = db.query(UserAccess).filter_by(user_id=user_id).first()
        if not token_entry:
            raise HTTPException(status_code=404, detail="User access token not found")

        request = InvestmentsHoldingsGetRequest(access_token=token_entry.access_token)
        response = client.investments_holdings_get(request)

        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch investments: {e}")
