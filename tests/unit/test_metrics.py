"""Metrics rendering + counter behavior."""

from __future__ import annotations

from price_forecast.metrics import Metrics


def test_counters_increment():
    m = Metrics()
    m.inc_predict(12.5, ok=True)
    m.inc_predict(8.0, ok=False, schema_error=True)
    m.inc_batch(ok=True)
    m.inc_reload(ok=False)
    m.set_model_loaded(True)
    assert m.predict_total == 2
    assert m.predict_errors == 1
    assert m.predict_schema_errors == 1
    assert m.predict_latency_count == 2
    assert m.predict_latency_sum_ms == 20.5
    assert m.batch_predict_total == 1
    assert m.reload_total == 1
    assert m.reload_errors == 1
    assert m.model_loaded == 1
    assert m.last_reload_unixtime > 0


def test_render_contains_all_metrics():
    m = Metrics()
    m.inc_predict(5.0, ok=True)
    m.set_model_loaded(True)
    body = m.render(project="product_dq", model_name="price_forecast", model_version="v42")
    for needle in (
        "price_forecast_predict_total",
        "price_forecast_predict_errors_total",
        "price_forecast_predict_schema_errors_total",
        "price_forecast_batch_predict_total",
        "price_forecast_reload_total",
        "price_forecast_model_loaded",
        "price_forecast_last_reload_unixtime",
        "price_forecast_predict_latency_sum_ms",
        'project="product_dq"',
        'model_name="price_forecast"',
        'model_version="v42"',
        "# HELP",
        "# TYPE",
    ):
        assert needle in body, f"missing {needle!r} in:\n{body}"


def test_render_is_prometheus_parseable_shape():
    body = Metrics().render(project="p", model_name="m", model_version="none")
    lines = body.splitlines()
    assert len(lines) >= 30
    assert "" not in lines[:-1] or lines.count("") <= 1
