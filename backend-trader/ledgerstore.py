"""Fake-money account ledger: accounts.db (kept separate from users.db).

All money is stored as integer cents. Shares are whole numbers.
Nothing here touches real money or a real brokerage.
"""

import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
ACCOUNTS_DB = DATA_DIR / "accounts.db"

STARTING_CASH_CENTS = int(float(os.getenv("STARTING_CASH", "100000")) * 100)
SLOTS = (1, 2, 3)
MAX_SHARES = 1_000_000
MAX_CASH_MOVE_CENTS = 10_000_000 * 100

SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ACCOUNTS_DB, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def init():
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id            INTEGER NOT NULL,
                slot               INTEGER NOT NULL,
                name               TEXT NOT NULL,
                cash_cents         INTEGER NOT NULL,
                net_deposits_cents INTEGER NOT NULL,
                created_at         INTEGER NOT NULL,
                UNIQUE (user_id, slot)
            );
            CREATE TABLE IF NOT EXISTS positions (
                account_id INTEGER NOT NULL
                           REFERENCES accounts(id) ON DELETE CASCADE,
                symbol     TEXT NOT NULL,
                shares     INTEGER NOT NULL,
                cost_cents INTEGER NOT NULL,
                PRIMARY KEY (account_id, symbol)
            );
            CREATE TABLE IF NOT EXISTS trades (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id     INTEGER NOT NULL
                               REFERENCES accounts(id) ON DELETE CASCADE,
                symbol         TEXT NOT NULL,
                side           TEXT NOT NULL,
                shares         INTEGER NOT NULL,
                price_cents    INTEGER NOT NULL,
                total_cents    INTEGER NOT NULL,
                realized_cents INTEGER,
                created_at     INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cash_moves (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id   INTEGER NOT NULL
                             REFERENCES accounts(id) ON DELETE CASCADE,
                kind         TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                created_at   INTEGER NOT NULL
            );
            """
        )


def to_cents(value):
    return int(round(float(value) * 100))


def dollars(cents):
    return round(cents / 100, 2)


def normalize_symbol(symbol):
    symbol = (symbol or "").strip().upper().lstrip("$")

    if not SYMBOL_RE.match(symbol):
        raise ValueError("Invalid stock symbol")

    return symbol


def ensure_accounts(user_id):
    """Every user gets Account 1-3 with starting fake cash on first use."""
    now = int(time.time())

    with connect() as db:
        with transaction(db):
            for slot in SLOTS:
                cur = db.execute(
                    "INSERT OR IGNORE INTO accounts "
                    "(user_id, slot, name, cash_cents, net_deposits_cents, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, slot, f"Account {slot}",
                     STARTING_CASH_CENTS, STARTING_CASH_CENTS, now),
                )

                if cur.rowcount == 1:
                    db.execute(
                        "INSERT INTO cash_moves (account_id, kind, amount_cents, created_at) "
                        "VALUES (?, 'start', ?, ?)",
                        (cur.lastrowid, STARTING_CASH_CENTS, now),
                    )


def _account(db, user_id, slot):
    row = db.execute(
        "SELECT * FROM accounts WHERE user_id = ? AND slot = ?",
        (user_id, slot),
    ).fetchone()

    if not row:
        raise LookupError("Account not found")

    return row


def held_symbols(user_id):
    with connect() as db:
        rows = db.execute(
            "SELECT DISTINCT p.symbol FROM positions p "
            "JOIN accounts a ON a.id = p.account_id WHERE a.user_id = ?",
            (user_id,),
        ).fetchall()

    return [r["symbol"] for r in rows]


def summary(user_id, slot, prices, include_trades=False):
    """prices: {symbol: live price or None}."""
    with connect() as db:
        acct = _account(db, user_id, slot)

        pos_rows = db.execute(
            "SELECT symbol, shares, cost_cents FROM positions "
            "WHERE account_id = ? ORDER BY symbol",
            (acct["id"],),
        ).fetchall()

        trade_rows = []

        if include_trades:
            trade_rows = db.execute(
                "SELECT symbol, side, shares, price_cents, total_cents, "
                "realized_cents, created_at FROM trades "
                "WHERE account_id = ? ORDER BY id DESC LIMIT 20",
                (acct["id"],),
            ).fetchall()

    positions = []
    positions_value = 0

    for p in pos_rows:
        price = prices.get(p["symbol"])
        value_cents = (
            p["shares"] * to_cents(price) if price else p["cost_cents"]
        )
        pnl = value_cents - p["cost_cents"]
        positions_value += value_cents

        positions.append({
            "symbol": p["symbol"],
            "shares": p["shares"],
            "avg_cost": round(p["cost_cents"] / p["shares"] / 100, 4),
            "price": price,
            "price_missing": not price,
            "market_value": dollars(value_cents),
            "unrealized_pl": dollars(pnl),
            "unrealized_pl_pct": round(pnl / p["cost_cents"] * 100, 2)
            if p["cost_cents"] else 0.0,
        })

    total_cents = acct["cash_cents"] + positions_value
    total_pl = total_cents - acct["net_deposits_cents"]

    result = {
        "slot": acct["slot"],
        "name": acct["name"],
        "cash": dollars(acct["cash_cents"]),
        "positions_value": dollars(positions_value),
        "total_value": dollars(total_cents),
        "net_deposits": dollars(acct["net_deposits_cents"]),
        "total_pl": dollars(total_pl),
        "total_pl_pct": round(total_pl / acct["net_deposits_cents"] * 100, 2)
        if acct["net_deposits_cents"] > 0 else 0.0,
        "positions": positions,
    }

    if include_trades:
        result["recent_trades"] = [
            {
                "symbol": t["symbol"],
                "side": t["side"],
                "shares": t["shares"],
                "price": dollars(t["price_cents"]),
                "total": dollars(t["total_cents"]),
                "realized_pl": dollars(t["realized_cents"])
                if t["realized_cents"] is not None else None,
                "time": t["created_at"],
            }
            for t in trade_rows
        ]

    return result


def trade(user_id, slot, symbol, side, shares, price):
    symbol = normalize_symbol(symbol)
    side = (side or "").lower()

    if side not in ("buy", "sell"):
        raise ValueError("Side must be buy or sell")

    if not isinstance(shares, int) or not 1 <= shares <= MAX_SHARES:
        raise ValueError(f"Shares must be a whole number from 1 to {MAX_SHARES}")

    price_cents = to_cents(price) if price else 0

    if price_cents <= 0:
        raise ValueError("No valid live price")

    total = shares * price_cents
    now = int(time.time())

    with connect() as db:
        with transaction(db):
            acct = _account(db, user_id, slot)

            pos = db.execute(
                "SELECT shares, cost_cents FROM positions "
                "WHERE account_id = ? AND symbol = ?",
                (acct["id"], symbol),
            ).fetchone()

            realized = None

            if side == "buy":
                if total > acct["cash_cents"]:
                    raise ValueError(
                        f"Not enough cash: need ${dollars(total):,.2f}, "
                        f"have ${dollars(acct['cash_cents']):,.2f}"
                    )

                cash_after = acct["cash_cents"] - total

                if pos:
                    db.execute(
                        "UPDATE positions SET shares = shares + ?, "
                        "cost_cents = cost_cents + ? "
                        "WHERE account_id = ? AND symbol = ?",
                        (shares, total, acct["id"], symbol),
                    )
                else:
                    db.execute(
                        "INSERT INTO positions (account_id, symbol, shares, cost_cents) "
                        "VALUES (?, ?, ?, ?)",
                        (acct["id"], symbol, shares, total),
                    )
            else:
                held = pos["shares"] if pos else 0

                if held < shares:
                    raise ValueError(f"You hold {held} shares of {symbol}")

                # average-cost accounting
                cost_removed = (pos["cost_cents"] * shares + held // 2) // held
                realized = total - cost_removed
                cash_after = acct["cash_cents"] + total

                if shares == held:
                    db.execute(
                        "DELETE FROM positions WHERE account_id = ? AND symbol = ?",
                        (acct["id"], symbol),
                    )
                else:
                    db.execute(
                        "UPDATE positions SET shares = shares - ?, "
                        "cost_cents = cost_cents - ? "
                        "WHERE account_id = ? AND symbol = ?",
                        (shares, cost_removed, acct["id"], symbol),
                    )

            db.execute(
                "UPDATE accounts SET cash_cents = ? WHERE id = ?",
                (cash_after, acct["id"]),
            )
            db.execute(
                "INSERT INTO trades (account_id, symbol, side, shares, price_cents, "
                "total_cents, realized_cents, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (acct["id"], symbol, side, shares, price_cents, total, realized, now),
            )

    return {
        "symbol": symbol,
        "side": side,
        "shares": shares,
        "price": dollars(price_cents),
        "total": dollars(total),
        "cash_after": dollars(cash_after),
        "realized_pl": dollars(realized) if realized is not None else None,
    }


def cash_move(user_id, slot, action, amount):
    action = (action or "").lower()

    if action not in ("deposit", "withdraw"):
        raise ValueError("Action must be deposit or withdraw")

    cents = to_cents(amount)

    if not 0 < cents <= MAX_CASH_MOVE_CENTS:
        raise ValueError("Amount must be greater than 0 and at most $10,000,000")

    now = int(time.time())

    with connect() as db:
        with transaction(db):
            acct = _account(db, user_id, slot)

            if action == "withdraw":
                if cents > acct["cash_cents"]:
                    raise ValueError(
                        f"Not enough cash: have ${dollars(acct['cash_cents']):,.2f}"
                    )
                delta = -cents
            else:
                delta = cents

            db.execute(
                "UPDATE accounts SET cash_cents = cash_cents + ?, "
                "net_deposits_cents = net_deposits_cents + ? WHERE id = ?",
                (delta, delta, acct["id"]),
            )
            db.execute(
                "INSERT INTO cash_moves (account_id, kind, amount_cents, created_at) "
                "VALUES (?, ?, ?, ?)",
                (acct["id"], action, cents, now),
            )

        new_cash = acct["cash_cents"] + delta

    return {"action": action, "amount": dollars(cents), "cash_after": dollars(new_cash)}
