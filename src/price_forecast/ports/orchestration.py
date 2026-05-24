"""Orchestration port — start a training run on whatever orchestrator the
deployment uses (GitHub Actions today; Argo, Jenkins, Airflow, Step Functions
tomorrow). Business logic depends only on this port."""

from __future__ import annotations

from abc import ABC, abstractmethod


class OrchestrationAdapter(ABC):
    @abstractmethod
    def dispatch_training(
        self,
        *,
        trigger_id: str,
        category: str,
        project: str,
        model_name: str,
        bucket: str,
        prefix: str,
        auto_promote: bool,
    ) -> None:
        """Kick off a training run for trigger_id.

        Implementations should be idempotent w.r.t. trigger_id — calling twice
        with the same id MUST NOT enqueue two runs.

        Raise RuntimeError when the orchestrator refuses the request (auth,
        rate-limit, persistent failure). Callers treat that as a hard failure
        and write a failed.json marker so consumers don't poll forever.
        """
