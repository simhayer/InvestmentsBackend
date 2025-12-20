# services/crud.py
from sqlalchemy.orm import Session
from models.user import User
from models.holding import Holding

def get_user_by_supabase_id(db: Session, supabase_user_id: str) -> User | None:
    return db.query(User).filter(User.supabase_user_id == supabase_user_id).first()

def get_or_create_user(db: Session, supabase_user_id: str, email: str | None) -> User:
    user = get_user_by_supabase_id(db, supabase_user_id)
    if user:
        # optional: keep email synced
        if email and user.email != email:
            user.email = email
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    if not email:
        # Supabase JWT usually includes email, but handle just in case
        email = f"{supabase_user_id}@no-email.local"

    user = User(email=email, supabase_user_id=supabase_user_id, hashed_password=None)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# Keep this if you still want it (not needed for auth anymore)
def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()

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
