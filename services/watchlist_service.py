from __future__ import annotations

from typing import Iterable, List

from sqlalchemy.orm import Session, selectinload

from models.watchlist import Watchlist, WatchlistItem


def _normalize_symbol(value: str) -> str:
    symbol = (value or "").strip().upper()
    if not symbol or len(symbol) > 20:
        raise ValueError("symbol must be 1-20 characters")
    return symbol


def _normalize_symbols(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = _normalize_symbol(value)
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def list_watchlists(db: Session, user_id: int) -> List[Watchlist]:
    return (
        db.query(Watchlist)
        .options(selectinload(Watchlist.items))
        .filter(Watchlist.user_id == user_id)
        .order_by(Watchlist.is_default.desc(), Watchlist.created_at.asc(), Watchlist.id.asc())
        .all()
    )


def get_watchlist(db: Session, user_id: int, watchlist_id: int) -> Watchlist | None:
    return (
        db.query(Watchlist)
        .options(selectinload(Watchlist.items))
        .filter(Watchlist.user_id == user_id, Watchlist.id == watchlist_id)
        .first()
    )


def _clear_default_watchlists(db: Session, user_id: int, exclude_id: int | None = None) -> None:
    query = db.query(Watchlist).filter(Watchlist.user_id == user_id, Watchlist.is_default.is_(True))
    if exclude_id is not None:
        query = query.filter(Watchlist.id != exclude_id)
    for watchlist in query.all():
        watchlist.is_default = False


def create_watchlist(
    db: Session,
    user_id: int,
    *,
    name: str,
    is_default: bool = False,
    symbols: Iterable[str] | None = None,
) -> Watchlist:
    existing = (
        db.query(Watchlist)
        .filter(Watchlist.user_id == user_id, Watchlist.name == name)
        .first()
    )
    if existing:
        raise ValueError("Watchlist with this name already exists")

    should_default = is_default or not db.query(Watchlist).filter(Watchlist.user_id == user_id).first()
    if should_default:
        _clear_default_watchlists(db, user_id)

    watchlist = Watchlist(user_id=user_id, name=name, is_default=should_default)
    db.add(watchlist)
    db.flush()

    for symbol in _normalize_symbols(symbols or []):
        db.add(WatchlistItem(watchlist_id=watchlist.id, symbol=symbol))

    db.commit()
    return get_watchlist(db, user_id, watchlist.id)  # type: ignore[return-value]


def update_watchlist(
    db: Session,
    user_id: int,
    watchlist_id: int,
    *,
    name: str | None = None,
    is_default: bool | None = None,
) -> Watchlist:
    watchlist = get_watchlist(db, user_id, watchlist_id)
    if not watchlist:
        raise ValueError("Watchlist not found")

    if name is not None and name != watchlist.name:
        duplicate = (
            db.query(Watchlist)
            .filter(Watchlist.user_id == user_id, Watchlist.name == name, Watchlist.id != watchlist_id)
            .first()
        )
        if duplicate:
            raise ValueError("Watchlist with this name already exists")
        watchlist.name = name

    if is_default is True:
        _clear_default_watchlists(db, user_id, exclude_id=watchlist.id)
        watchlist.is_default = True
    elif is_default is False:
        watchlist.is_default = False

    db.commit()
    return get_watchlist(db, user_id, watchlist_id)  # type: ignore[return-value]


def delete_watchlist(db: Session, user_id: int, watchlist_id: int) -> None:
    watchlist = get_watchlist(db, user_id, watchlist_id)
    if not watchlist:
        raise ValueError("Watchlist not found")

    was_default = watchlist.is_default
    db.delete(watchlist)
    db.commit()

    if was_default:
        replacement = (
            db.query(Watchlist)
            .filter(Watchlist.user_id == user_id)
            .order_by(Watchlist.created_at.asc(), Watchlist.id.asc())
            .first()
        )
        if replacement:
            replacement.is_default = True
            db.commit()


def add_watchlist_item(
    db: Session,
    user_id: int,
    watchlist_id: int,
    *,
    symbol: str,
    note: str | None = None,
) -> Watchlist:
    watchlist = get_watchlist(db, user_id, watchlist_id)
    if not watchlist:
        raise ValueError("Watchlist not found")

    normalized_symbol = _normalize_symbol(symbol)
    exists = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.watchlist_id == watchlist_id, WatchlistItem.symbol == normalized_symbol)
        .first()
    )
    if exists:
        raise ValueError("Symbol already exists in watchlist")

    db.add(WatchlistItem(watchlist_id=watchlist_id, symbol=normalized_symbol, note=note))
    db.commit()
    return get_watchlist(db, user_id, watchlist_id)  # type: ignore[return-value]


def remove_watchlist_item(db: Session, user_id: int, watchlist_id: int, *, symbol: str) -> Watchlist:
    watchlist = get_watchlist(db, user_id, watchlist_id)
    if not watchlist:
        raise ValueError("Watchlist not found")

    normalized_symbol = _normalize_symbol(symbol)
    item = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.watchlist_id == watchlist_id, WatchlistItem.symbol == normalized_symbol)
        .first()
    )
    if not item:
        raise ValueError("Symbol not found in watchlist")

    db.delete(item)
    db.commit()
    return get_watchlist(db, user_id, watchlist_id)  # type: ignore[return-value]


def get_watchlist_symbols(db: Session, user_id: int, watchlist_id: int) -> tuple[Watchlist, list[str]]:
    watchlist = get_watchlist(db, user_id, watchlist_id)
    if not watchlist:
        raise ValueError("Watchlist not found")
    symbols = [item.symbol for item in watchlist.items]
    return watchlist, symbols
