"""
revenue_automation/dispatch/dispatcher.py
==========================================
ActionDispatcher — executa as ações de uma PolicyDecision nos canais certos.

Cada ação pode ter múltiplos canais. O dispatcher tenta cada canal em sequência.
Se um canal falha, passa para o próximo sem abortar (fallback seguro).
Sempre tenta o canal CONSOLE como fallback final.

Retorna InterventionRecord completo pronto para o audit trail.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..schemas.models import (
    DispatchChannel,
    DispatchResult,
    InterventionContext,
    InterventionRecord,
    PolicyDecision,
    SelectedAction,
)
from .channels import api_hook, console, email, file as file_ch, slack

log = logging.getLogger("dispatch.dispatcher")

# Mapa canal → função de envio
_CHANNEL_MAP = {
    DispatchChannel.CONSOLE:  lambda code, p: console.send(code, p),
    DispatchChannel.FILE:     lambda code, p: file_ch.send(code, p),
    DispatchChannel.SLACK:    lambda code, p: slack.send(code, p),
    DispatchChannel.EMAIL:    lambda code, p: email.send(code, p),
    DispatchChannel.API_HOOK: lambda code, p: api_hook.send(code, p),
}


class ActionDispatcher:
    """
    Dispatcher que envia ações para os canais definidos pela política.

    Sempre garante que ao menos o canal CONSOLE é tentado.
    Em caso de falha de um canal, registra o erro e continua nos demais.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run   # se True, apenas loga sem enviar

    def dispatch(
        self,
        decision:        PolicyDecision,
        ctx:             InterventionContext,
        previous_score:  Optional[float] = None,
        current_score:   Optional[float] = None,
    ) -> InterventionRecord:
        """
        Executa todas as ações do PolicyDecision.
        Retorna InterventionRecord com resultados de todos os canais.
        """
        correlation_id    = uuid4().hex
        all_results:       List[DispatchResult] = []
        actions_taken:     List[str]            = []
        channels_used:     List[str]            = []
        reasons:           List[str]            = []
        payload_summary:   Dict[str, Any]       = {}

        for action in decision.actions:
            action_code = action.code.value
            actions_taken.append(action_code)
            reasons.append(action.reason)

            results = self._dispatch_action(action, ctx)
            all_results.extend(results)

            for r in results:
                if r.success:
                    channels_used.append(r.channel.value)

            payload_summary[action_code] = action.payload

        record = InterventionRecord(
            correlation_id  = correlation_id,
            account_id      = ctx.account_id,
            session_id      = ctx.session_id,
            timestamp       = datetime.now(timezone.utc).isoformat(),
            previous_score  = previous_score,
            current_score   = current_score,
            churn_risk      = ctx.churn_risk,
            risk_level      = ctx.risk_level,
            actions_taken   = actions_taken,
            channels_used   = list(set(channels_used)),
            reasons         = list(dict.fromkeys(reasons)),  # dedup preserving order
            dispatch_results= [asdict(r) for r in all_results],
            payload_summary = payload_summary,
        )

        log.info(
            "Intervention %s | account=%s | actions=%s | channels=%s",
            correlation_id[:8], ctx.account_id, actions_taken, record.channels_used,
        )
        return record

    def _dispatch_action(
        self,
        action: SelectedAction,
        ctx:    InterventionContext,
    ) -> List[DispatchResult]:
        """Tenta cada canal da ação. Garante fallback para CONSOLE."""
        results:  List[DispatchResult] = []
        code      = action.code.value
        payload   = action.payload
        channels  = list(action.channels)

        # Garante CONSOLE sempre presente
        if DispatchChannel.CONSOLE not in channels:
            channels.append(DispatchChannel.CONSOLE)

        for channel in channels:
            if self.dry_run:
                results.append(DispatchResult(
                    channel      = channel,
                    action_code  = code,
                    success      = True,
                    message      = f"[DRY RUN] {code} via {channel.value}",
                    payload_sent = payload,
                ))
                continue

            fn = _CHANNEL_MAP.get(channel)
            if fn is None:
                log.warning("Canal desconhecido: %s", channel)
                continue

            try:
                result = fn(code, payload)
            except Exception as e:
                log.error("Canal %s falhou inesperadamente: %s", channel, e)
                result = DispatchResult(
                    channel      = channel,
                    action_code  = code,
                    success      = False,
                    message      = "Unexpected error",
                    payload_sent = payload,
                    error        = str(e),
                )

            results.append(result)

            if not result.success:
                log.warning(
                    "Canal %s falhou para %s: %s",
                    channel.value, code, result.error or result.message,
                )

        return results
