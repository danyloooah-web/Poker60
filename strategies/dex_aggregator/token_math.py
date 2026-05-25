"""Shared token metadata and unit conversion helpers."""

from __future__ import annotations

TOKEN_DECIMALS: dict[str, int] = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 18,
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": 8,
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,
    # Base
    "0x4200000000000000000000000000000000000006": 18,  # WETH
    "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": 6,   # USDC
    "0x50c5725949a6f0c72e6c4a641f24049a917db0cb": 18,  # DAI
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf": 8,   # cbBTC
    "0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca": 6,   # USDbC
}


def get_decimals(token_address: str) -> int:
    """Return token decimals, defaulting to 18 for unknown assets."""
    return TOKEN_DECIMALS.get(token_address.lower(), 18)


def to_human(raw_amount: int, decimals: int) -> float:
    """Convert an integer token amount to its decimal form."""
    if decimals == 0:
        return float(raw_amount)
    return raw_amount / (10 ** decimals)


def to_raw(human_amount: float, decimals: int) -> int:
    """Convert a decimal token amount to its raw integer form."""
    return int(human_amount * (10 ** decimals))
