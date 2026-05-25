"""Danylo Minotaur solver for Subnet 112."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from common.abi_utils import encode_approve
from strategies.dex_aggregator.baseline_solver import BaselineSwapSolver, _KNOWN_POOLS
from strategies.dex_aggregator.swap_solver import UNISWAP_V3_ROUTERS
from strategies.dex_aggregator.v3_codec import encode_exact_input, encode_exact_input_single, encode_swap_path
from minotaur_subnet.sdk.intent_solver import MarketSnapshot, SolverMetadata
from minotaur_subnet.shared.types import AppIntentDefinition, ExecutionPlan, Interaction, IntentState

logger = logging.getLogger(__name__)

SOLVER_NAME = os.environ.get("MINOTAUR_SOLVER_NAME", "danylo-minotaur-solver")
SOLVER_VERSION = os.environ.get("MINOTAUR_SOLVER_VERSION", "1.2.2")
SOLVER_AUTHOR = os.environ.get("MINOTAUR_SOLVER_AUTHOR", "danyloooah")

_BASE_WETH = "0x4200000000000000000000000000000000000006"
_BASE_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
_BASE_DAI = "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb"


class MinerSolver(BaselineSwapSolver):
    """Cross-DEX routing solver tuned for Base DAI benchmark scenarios."""

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

    def generate_plan(
        self,
        intent: AppIntentDefinition,
        state: IntentState,
        snapshot: MarketSnapshot | None = None,
    ) -> ExecutionPlan:
        chain_id = state.chain_id or (snapshot.chain_id if snapshot else 1)
        if chain_id == 8453:
            dai_plan = self._try_base_dai_plan(intent, state, snapshot, chain_id)
            if dai_plan is not None:
                return dai_plan
        return super().generate_plan(intent, state, snapshot)

    def _try_base_dai_plan(
        self,
        intent: AppIntentDefinition,
        state: IntentState,
        snapshot: MarketSnapshot | None,
        chain_id: int,
    ) -> ExecutionPlan | None:
        swap_params = self._normalized_swap_params(intent, state)
        input_token = swap_params.get("input_token", "")
        output_token = swap_params.get("output_token", "")
        amount_in = int(swap_params.get("input_amount", 0) or 0)
        min_output = int(swap_params.get("min_output_amount", 0) or 0)
        if not input_token or not output_token or amount_in <= 0:
            return None

        in_l, out_l = input_token.lower(), output_token.lower()
        if in_l == _BASE_DAI.lower() and out_l == _BASE_USDC.lower():
            return self._build_base_v3_single(
                intent, state, snapshot, chain_id,
                input_token, output_token, amount_in, min_output, fee=100,
            )
        if in_l == _BASE_WETH.lower() and out_l == _BASE_DAI.lower():
            return self._build_base_v3_multihop(
                intent, state, snapshot, chain_id,
                [_BASE_WETH, _BASE_USDC, _BASE_DAI], [500, 100],
                amount_in, min_output,
            )
        return None

    def _build_base_v3_single(
        self,
        intent: AppIntentDefinition,
        state: IntentState,
        snapshot: MarketSnapshot | None,
        chain_id: int,
        input_token: str,
        output_token: str,
        amount_in: int,
        min_output: int,
        fee: int,
    ) -> ExecutionPlan | None:
        router = UNISWAP_V3_ROUTERS.get(chain_id)
        if not router:
            return None

        swap_params = self._normalized_swap_params(intent, state)
        recipient = state.contract_address or swap_params.get("receiver", state.owner)
        timestamp = snapshot.timestamp if snapshot else int(time.time())
        deadline = timestamp + (self._processor.deadline_offset if self._processor else 300)

        return ExecutionPlan(
            intent_id=intent.app_id,
            interactions=[
                Interaction(
                    target=input_token,
                    value="0",
                    call_data=encode_approve(router, amount_in),
                    chain_id=chain_id,
                ),
                Interaction(
                    target=router,
                    value="0",
                    call_data=encode_exact_input_single(
                        token_in=input_token,
                        token_out=output_token,
                        fee=fee,
                        recipient=recipient,
                        deadline=deadline,
                        amount_in=amount_in,
                        amount_out_minimum=min_output,
                        chain_id=chain_id,
                    ),
                    chain_id=chain_id,
                ),
            ],
            deadline=deadline,
            nonce=state.nonce,
            metadata={
                "route": "base_dai_single",
                "fee_tier": fee,
                "input_token": input_token,
                "output_token": output_token,
                "input_amount": str(amount_in),
                "min_output_amount": str(min_output),
                "chain_id": chain_id,
            },
        )

    def _build_base_v3_multihop(
        self,
        intent: AppIntentDefinition,
        state: IntentState,
        snapshot: MarketSnapshot | None,
        chain_id: int,
        tokens: list[str],
        fees: list[int],
        amount_in: int,
        min_output: int,
    ) -> ExecutionPlan | None:
        router = UNISWAP_V3_ROUTERS.get(chain_id)
        if not router:
            return None

        swap_params = self._normalized_swap_params(intent, state)
        recipient = state.contract_address or swap_params.get("receiver", state.owner)
        timestamp = snapshot.timestamp if snapshot else int(time.time())
        deadline = timestamp + (self._processor.deadline_offset if self._processor else 300)
        path = encode_swap_path(tokens, fees)

        return ExecutionPlan(
            intent_id=intent.app_id,
            interactions=[
                Interaction(
                    target=tokens[0],
                    value="0",
                    call_data=encode_approve(router, amount_in),
                    chain_id=chain_id,
                ),
                Interaction(
                    target=router,
                    value="0",
                    call_data=encode_exact_input(
                        path=path,
                        recipient=recipient,
                        deadline=deadline,
                        amount_in=amount_in,
                        amount_out_minimum=min_output,
                    ),
                    chain_id=chain_id,
                ),
            ],
            deadline=deadline,
            nonce=state.nonce,
            metadata={
                "route": "base_dai_multihop",
                "tokens": tokens,
                "fees": fees,
                "input_token": tokens[0],
                "output_token": tokens[-1],
                "input_amount": str(amount_in),
                "min_output_amount": str(min_output),
                "chain_id": chain_id,
            },
        )

    def metadata(self) -> SolverMetadata:
        base = super().metadata()
        return SolverMetadata(
            name=SOLVER_NAME,
            version=SOLVER_VERSION,
            author=SOLVER_AUTHOR,
            description=(
                "BaselineSwapSolver v1.2.2 with explicit Base DAI Uni V3 routes "
                "(DAI/USDC direct, WETH/DAI via USDC hop)."
            ),
            supported_chains=base.supported_chains,
            supported_intent_types=base.supported_intent_types,
        )


SOLVER_CLASS = MinerSolver
