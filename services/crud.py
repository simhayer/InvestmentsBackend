from models.user import User
from models.holding import Holding
from sqlalchemy.orm import Session
from services.auth import get_password_hash

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, email: str, password: str):
    hashed_pw = get_password_hash(password)
    user = User(email=email, hashed_password=hashed_pw)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def create_holding(
    db: Session,
    user_id: int,
    symbol: str,
    quantity: float,
    purchase_price: float,
    type_: str
) -> Holding:
    holding = Holding(
        symbol=symbol,
        quantity=quantity,
        purchase_price=purchase_price,
        type=type_, 
        user_id=user_id,
    )
    db.add(holding)
    db.commit()
    db.refresh(holding)
    return holding

