# cache/crypto_catalog.py
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class CryptoCoin:
    symbol: str
    name: str | None = None

class CryptoCatalog:
    def __init__(self):
        self._coins: List[CryptoCoin] = []

    def set(self, coins: List[CryptoCoin]):
        self._coins = coins

    def search(self, q: str, limit: int = 5) -> List[CryptoCoin]:
        q_raw = (q or "").strip()
        if not q_raw:
            return []

        q_upper = q_raw.upper()
        q_lower = q_raw.lower()

        def name_lower(c: CryptoCoin) -> str:
            return (c.name or "").lower()

        # 1) exact symbol match first
        exact_symbol = [c for c in self._coins if c.symbol == q_upper]
        if len(exact_symbol) >= limit:
            return exact_symbol[:limit]

        used = {c.symbol for c in exact_symbol}

        # 2) symbol prefix
        sym_prefix = [c for c in self._coins if c.symbol.startswith(q_upper) and c.symbol not in used]
        used |= {c.symbol for c in sym_prefix}
        if len(exact_symbol) + len(sym_prefix) >= limit:
            return (exact_symbol + sym_prefix)[:limit]

        # 3) name prefix (e.g. "bit" -> "Bitcoin")
        name_prefix = [
            c for c in self._coins
            if c.symbol not in used and c.name and name_lower(c).startswith(q_lower)
        ]
        used |= {c.symbol for c in name_prefix}
        if len(exact_symbol) + len(sym_prefix) + len(name_prefix) >= limit:
            return (exact_symbol + sym_prefix + name_prefix)[:limit]

        # 4) symbol contains
        sym_contains = [
            c for c in self._coins
            if c.symbol not in used and q_upper in c.symbol
        ]
        used |= {c.symbol for c in sym_contains}
        if len(exact_symbol) + len(sym_prefix) + len(name_prefix) + len(sym_contains) >= limit:
            return (exact_symbol + sym_prefix + name_prefix + sym_contains)[:limit]

        # 5) name contains
        name_contains = [
            c for c in self._coins
            if c.symbol not in used and c.name and q_lower in name_lower(c)
        ]

        return (exact_symbol + sym_prefix + name_prefix + sym_contains + name_contains)[:limit]


crypto_catalog = CryptoCatalog()
