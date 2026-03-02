from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from schemas.watchlist import WatchlistCreate, WatchlistItemCreate, WatchlistOut, WatchlistUpdate
from services.supabase_auth import get_current_db_user
from services.watchlist_service import (
    add_watchlist_item,
    create_watchlist,
    delete_watchlist,
    get_watchlist,
    list_watchlists,
    remove_watchlist_item,
    update_watchlist,
)

router = APIRouter()


@router.get("", response_model=List[WatchlistOut])
def get_user_watchlists(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    return list_watchlists(db, user.id)


@router.post("", response_model=WatchlistOut, status_code=status.HTTP_201_CREATED)
def create_user_watchlist(
    payload: WatchlistCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    try:
        return create_watchlist(
            db,
            user.id,
            name=payload.name,
            is_default=payload.is_default,
            symbols=payload.symbols,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/{watchlist_id}", response_model=WatchlistOut)
def get_user_watchlist(
    watchlist_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    watchlist = get_watchlist(db, user.id, watchlist_id)
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    return watchlist


@router.patch("/{watchlist_id}", response_model=WatchlistOut)
def update_user_watchlist(
    watchlist_id: int,
    payload: WatchlistUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    try:
        return update_watchlist(
            db,
            user.id,
            watchlist_id,
            name=payload.name,
            is_default=payload.is_default,
        )
    except ValueError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc))


@router.post("/{watchlist_id}/items", response_model=WatchlistOut)
def add_user_watchlist_item(
    watchlist_id: int,
    payload: WatchlistItemCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    try:
        return add_watchlist_item(
            db,
            user.id,
            watchlist_id,
            symbol=payload.symbol,
            note=payload.note,
        )
    except ValueError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 409
        raise HTTPException(status_code=status_code, detail=str(exc))


@router.delete("/{watchlist_id}/items/{symbol}", response_model=WatchlistOut)
def delete_user_watchlist_item(
    watchlist_id: int,
    symbol: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    try:
        return remove_watchlist_item(db, user.id, watchlist_id, symbol=symbol)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user_watchlist(
    watchlist_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_db_user),
):
    try:
        delete_watchlist(db, user.id, watchlist_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
