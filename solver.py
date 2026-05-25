"""Reference solver for Subnet 112 (Minotaur).

This is the file miners fork to ship their own strategy. The validator
harness imports ``SOLVER_CLASS`` from this module and calls
``generate_plan(intent, state, snapshot)`` on every order.

Default behaviour comes from ``BaselineSwapSolver`` in
``common/baseline_solver.py``, which on Base routes across both
Uniswap V3 and Aerodrome Slipstream pools, picks the best output
across DEXes, and falls back to multi-hop through common intermediary
tokens (WETH, USDC) when no direct pool wins.

To beat that baseline, override ``generate_plan`` (or any other method)
on ``MinerSolver`` below — for example by adding new DEXes, splitting a
trade across pools, learning routes from past orders, or pre-computing
plans on a faster path than RPC discovery allows.

The validator's screening pipeline builds this repo as a Docker image
``FROM ghcr.io/subnet112/solver-base:v1`` and runs the runner harness,
which loads ``SOLVER_CLASS`` from ``/app/solver/solver.py``.
"""

from __future__ import annotations

import logging
import os

from strategies.dex_aggregator.baseline_solver import BaselineSwapSolver
from minotaur_subnet.sdk.intent_solver import SolverMetadata

logger = logging.getLogger(__name__)


SOLVER_NAME = os.environ.get("MINOTAUR_SOLVER_NAME", "danylo-minotaur-solver")
SOLVER_VERSION = os.environ.get("MINOTAUR_SOLVER_VERSION", "1.0.0")
SOLVER_AUTHOR = os.environ.get("MINOTAUR_SOLVER_AUTHOR", "danyloooah")


class MinerSolver(BaselineSwapSolver):
    """Cross-DEX routing solver for Minotaur subnet 112.

    Extends ``BaselineSwapSolver`` with tighter pool-cache refresh and
    Base-specific intermediary tokens for broader multi-hop coverage.
    """

    def initialize(self, config: dict) -> None:
        super().initialize(config)
        # Refresh discovered pool state more often for competitive quoting.
        self._pool_cache_ttl = float(config.get("pool_cache_ttl", 6.0))

    def metadata(self) -> SolverMetadata:
        base = super().metadata()
        return SolverMetadata(
            name=SOLVER_NAME,
            version=SOLVER_VERSION,
            author=SOLVER_AUTHOR,
            description=(
                "BaselineSwapSolver with faster pool refresh and Base "
                "intermediary routing (WETH, USDC, cbBTC, USDbC)."
            ),
            supported_chains=base.supported_chains,
            supported_intent_types=base.supported_intent_types,
        )


SOLVER_CLASS = MinerSolver
