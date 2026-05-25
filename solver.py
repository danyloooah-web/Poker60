"""Danylo Minotaur solver for Subnet 112."""

from __future__ import annotations

import logging
import os
from typing import Any

from strategies.dex_aggregator.baseline_solver import BaselineSwapSolver, _KNOWN_POOLS
from minotaur_subnet.sdk.intent_solver import MarketSnapshot, SolverMetadata

logger = logging.getLogger(__name__)

SOLVER_NAME = os.environ.get("MINOTAUR_SOLVER_NAME", "danylo-minotaur-solver")
SOLVER_VERSION = os.environ.get("MINOTAUR_SOLVER_VERSION", "1.2.0")
SOLVER_AUTHOR = os.environ.get("MINOTAUR_SOLVER_AUTHOR", "danyloooah")


class MinerSolver(BaselineSwapSolver):
    """Cross-DEX routing solver tuned for Base DAI pairs and benchmark scoring."""

    def initialize(self, config: dict) -> None:
        super().initialize(config)
        self._pool_cache_ttl = float(config.get("pool_cache_ttl", 6.0))
        if self._processor is not None:
            self._processor.slippage_bps = int(config.get("slippage_bps", 50))

    def _get_pool_states(
        self,
        chain_id: int,
        snapshot: MarketSnapshot | None,
    ) -> dict[str, dict[str, Any]]:
        """Merge RPC-known pools with snapshot pools for fuller route coverage."""
        pool_states: dict[str, dict[str, Any]] = {}

        if self._rpc_urls.get(chain_id):
            pool_states.update(self._discover_pools(chain_id))
            w3 = self._get_web3(chain_id)
            if w3 is not None:
                seen = {k.lower() for k in pool_states}
                for addr in _KNOWN_POOLS.get(chain_id, []):
                    if addr.lower() in seen:
                        continue
                    state = self._query_pool_state(w3, addr)
                    if state is not None:
                        pool_states[addr] = state
                        seen.add(addr.lower())

        if snapshot is not None and snapshot.pool_states:
            for addr, state in snapshot.pool_states.items():
                if addr.lower() not in {k.lower() for k in pool_states}:
                    pool_states[addr] = state

        return pool_states

    def metadata(self) -> SolverMetadata:
        base = super().metadata()
        return SolverMetadata(
            name=SOLVER_NAME,
            version=SOLVER_VERSION,
            author=SOLVER_AUTHOR,
            description=(
                "BaselineSwapSolver v1.2 with Base WETH/DAI pool seeding, "
                "merged RPC+snapshot pool states, and 50 bps slippage."
            ),
            supported_chains=base.supported_chains,
            supported_intent_types=base.supported_intent_types,
        )


SOLVER_CLASS = MinerSolver
