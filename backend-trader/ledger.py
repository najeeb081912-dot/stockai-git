"""Account endpoints (fake money). Every route needs a signed-in user."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import ledgerstore as store
from auth import get_current_user

store.init()

router = APIRouter(prefix="/accounts", tags=["accounts"])

_price_provider = None


def set_price_provider(fn):
    """fn(list_of_symbols) -> {symbol: live price or None}"""
    global _price_provider
    _price_provider = fn


def live_prices(symbols):
    symbols = sorted(set(symbols))

    if not symbols or _price_provider is None:
        return {}

    return _price_provider(symbols)


def check_slot(slot):
    if slot not in store.SLOTS:
        raise HTTPException(status_code=404, detail="Account not found")


class TradeIn(BaseModel):
    symbol: str
    side: str
    shares: int


class CashIn(BaseModel):
    action: str
    amount: float


@router.get("")
def list_accounts(user=Depends(get_current_user)):
    store.ensure_accounts(user["id"])

    prices = live_prices(store.held_symbols(user["id"]))

    return {
        "accounts": [
            store.summary(user["id"], slot, prices) for slot in store.SLOTS
        ]
    }


@router.get("/{slot}")
def get_account(slot: int, user=Depends(get_current_user)):
    check_slot(slot)
    store.ensure_accounts(user["id"])

    prices = live_prices(store.held_symbols(user["id"]))

    return store.summary(user["id"], slot, prices, include_trades=True)


@router.post("/{slot}/trade")
def place_trade(slot: int, body: TradeIn, user=Depends(get_current_user)):
    check_slot(slot)
    store.ensure_accounts(user["id"])

    try:
        symbol = store.normalize_symbol(body.symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # fetch the price first so no database lock is held during the network call
    price = live_prices([symbol]).get(symbol)

    if not price:
        raise HTTPException(
            status_code=503,
            detail=f"No live price available for {symbol} right now",
        )

    try:
        return store.trade(user["id"], slot, symbol, body.side, body.shares, price)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{slot}/cash")
def move_cash(slot: int, body: CashIn, user=Depends(get_current_user)):
    check_slot(slot)
    store.ensure_accounts(user["id"])

    try:
        return store.cash_move(user["id"], slot, body.action, body.amount)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
