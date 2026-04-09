from __future__ import annotations

import asyncio
import json

import websockets

from decision_service.inference import predict_ex_ante
from decision_service.models import ExAnteDecisionRequest


def _parse_message(raw: str) -> ExAnteDecisionRequest:
    tokens = raw.strip().split()
    if len(tokens) < 5:
        raise ValueError("insufficient_tokens")
    hand = tokens[0]
    position = tokens[1]
    stack_bb = float(tokens[2])
    pot_bb_before = float(tokens[3])
    num_players = int(tokens[4])
    limpers = int(tokens[5]) if len(tokens) > 5 and tokens[5].isdigit() else 0
    open_size_bb = float(tokens[6]) if len(tokens) > 6 else 0.0
    return ExAnteDecisionRequest(
        hand=hand,
        position=position,
        stack_bb=stack_bb,
        pot_bb_before=pot_bb_before,
        num_players=num_players,
        limpers=limpers,
        open_size_bb=open_size_bb,
    )


async def handle_connection(websocket):
    async for raw_message in websocket:
        try:
            request = _parse_message(raw_message)
            result = predict_ex_ante(request)
            await websocket.send(json.dumps(result.__dict__, ensure_ascii=False))
        except Exception as exc:
            await websocket.send(json.dumps({"action": "WARNING", "confidence": 0.0, "warning": str(exc), "abstained": True}, ensure_ascii=False))


async def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    async with websockets.serve(handle_connection, host, port, ping_interval=20, ping_timeout=20, compression=None):
        await asyncio.Future()
