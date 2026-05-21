"""Prometheus-format metrics — written by hand to avoid a prometheus_client dep.

The text format is stable: ``# HELP``, ``# TYPE``, then ``metric_name{labels} value``
lines, one per metric series, separated by newlines. Prometheus and Grafana
Agent scrape this without issue.

Counters are thread-safe via a single lock. Gauges are atomic in Python
(attribute writes), so no lock needed for those.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Metrics:
    # Counters
    predict_total: int = 0
    predict_errors: int = 0
    predict_schema_errors: int = 0
    batch_predict_total: int = 0
    batch_predict_errors: int = 0
    batch_predict_schema_errors: int = 0
    reload_total: int = 0
    reload_errors: int = 0
    # Gauges
    model_loaded: int = 0  # 0 = not loaded, 1 = loaded
    last_reload_unixtime: float = 0.0
    # Latency (sum + count for an "average" view; histograms would need bucketing)
    predict_latency_sum_ms: float = 0.0
    predict_latency_count: int = 0
    # Lock
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def inc_predict(self, latency_ms: float, *, ok: bool, schema_error: bool = False) -> None:
        with self._lock:
            self.predict_total += 1
            self.predict_latency_sum_ms += latency_ms
            self.predict_latency_count += 1
            if not ok:
                self.predict_errors += 1
            if schema_error:
                self.predict_schema_errors += 1

    def inc_batch(self, *, ok: bool, schema_error: bool = False) -> None:
        with self._lock:
            self.batch_predict_total += 1
            if not ok:
                self.batch_predict_errors += 1
            if schema_error:
                self.batch_predict_schema_errors += 1

    def inc_reload(self, *, ok: bool) -> None:
        with self._lock:
            self.reload_total += 1
            if not ok:
                self.reload_errors += 1

    def set_model_loaded(self, loaded: bool) -> None:
        with self._lock:
            self.model_loaded = 1 if loaded else 0
            if loaded:
                self.last_reload_unixtime = time.time()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, *, app_id: str, model_version: str) -> str:
        labels = f'app_id="{app_id}",model_version="{model_version}"'
        lines = [
            "# HELP price_forecast_predict_total Total /predict calls.",
            "# TYPE price_forecast_predict_total counter",
            f"price_forecast_predict_total{{{labels}}} {self.predict_total}",
            "# HELP price_forecast_predict_errors_total Total failing /predict calls.",
            "# TYPE price_forecast_predict_errors_total counter",
            f"price_forecast_predict_errors_total{{{labels}}} {self.predict_errors}",
            "# HELP price_forecast_predict_schema_errors_total Total /predict requests rejected on schema.",
            "# TYPE price_forecast_predict_schema_errors_total counter",
            f"price_forecast_predict_schema_errors_total{{{labels}}} {self.predict_schema_errors}",
            "# HELP price_forecast_batch_predict_total Total /predict/batch calls.",
            "# TYPE price_forecast_batch_predict_total counter",
            f"price_forecast_batch_predict_total{{{labels}}} {self.batch_predict_total}",
            "# HELP price_forecast_batch_predict_errors_total Total failing /predict/batch calls.",
            "# TYPE price_forecast_batch_predict_errors_total counter",
            f"price_forecast_batch_predict_errors_total{{{labels}}} {self.batch_predict_errors}",
            "# HELP price_forecast_batch_predict_schema_errors_total Total /predict/batch requests rejected on schema.",
            "# TYPE price_forecast_batch_predict_schema_errors_total counter",
            f"price_forecast_batch_predict_schema_errors_total{{{labels}}} {self.batch_predict_schema_errors}",
            "# HELP price_forecast_reload_total Total model reload attempts.",
            "# TYPE price_forecast_reload_total counter",
            f"price_forecast_reload_total{{{labels}}} {self.reload_total}",
            "# HELP price_forecast_reload_errors_total Total failed model reloads.",
            "# TYPE price_forecast_reload_errors_total counter",
            f"price_forecast_reload_errors_total{{{labels}}} {self.reload_errors}",
            "# HELP price_forecast_model_loaded 1 if a model is currently loaded.",
            "# TYPE price_forecast_model_loaded gauge",
            f"price_forecast_model_loaded{{{labels}}} {self.model_loaded}",
            "# HELP price_forecast_last_reload_unixtime Unix timestamp of last successful reload.",
            "# TYPE price_forecast_last_reload_unixtime gauge",
            f"price_forecast_last_reload_unixtime{{{labels}}} {self.last_reload_unixtime}",
            "# HELP price_forecast_predict_latency_sum_ms Sum of /predict latencies (ms).",
            "# TYPE price_forecast_predict_latency_sum_ms counter",
            f"price_forecast_predict_latency_sum_ms{{{labels}}} {self.predict_latency_sum_ms:.3f}",
            "# HELP price_forecast_predict_latency_count Number of /predict latency observations.",
            "# TYPE price_forecast_predict_latency_count counter",
            f"price_forecast_predict_latency_count{{{labels}}} {self.predict_latency_count}",
            "",
        ]
        return "\n".join(lines)
