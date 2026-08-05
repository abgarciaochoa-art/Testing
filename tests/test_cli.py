"""Tests del CLI (`earnings_alpha.cli`) y del tearsheet (`earnings_alpha.reports`).

Todo corre **sin red y sin credenciales**, contra el proveedor sintético
(contrato §0.5). Cuatro familias:

1. **Subcomandos end-to-end.** Cada subcomando (`universe show|refresh`,
   `data status`, `factors list|compute`, `events scan`, `backtest run` en ambos
   modos, `collector run --dry-run`, `report`) devuelve 0 y produce la salida
   prometida.
2. **Códigos de salida.** Error de configuración/uso → 2, error de datos o de
   proveedor (`EarningsAlphaError`) → 1; nada se degrada a un 0 con tabla vacía.
3. **Tearsheet.** El HTML contiene todas las secciones (`SECTION_IDS`), es
   autocontenido (ninguna referencia a hosts externos), trae tema claro y
   oscuro, y los insumos inválidos fallan de forma explícita.
4. **Refresh append-only offline.** `universe refresh --from-csv` añade fechas
   nuevas y el `--dry-run` no toca el fichero.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from earnings_alpha.cli import LocalCsvUniverseSource, build_parser, main
from earnings_alpha.errors import DataQualityError
from earnings_alpha.reports import (
    SECTION_IDS,
    MetricWithCI,
    ic_metrics,
    render_tearsheet,
    sharpe_metrics,
    write_tearsheet,
)

# Mercado sintético pequeño y rápido, compartido por todos los subcomandos.
MKT_ARGS = [
    "--seed", "7",
    "--tickers", "12",
    "--start", "2021-01-04",
    "--end", "2022-12-30",
]

#: Todos los identificadores de sección del tearsheet completo.
ALL_SECTIONS = ("equity", "drawdown", "ic", "quantiles", "caar", "grid", "metrics", "params")


# ===========================================================================
# Utilidades de los tests
# ===========================================================================


def _fake_history_csv(path: Path, *rows: tuple[str, list[str]]) -> Path:
    """Escribe un CSV de composición histórica sintético (date,tickers)."""
    lines = ["date,tickers"]
    lines.extend(f'{day},"{",".join(tickers)}"' for day, tickers in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _tickers(n: int = 500) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _assert_self_contained(html: str) -> None:
    """El HTML no puede referenciar ningún host externo (ni CDNs ni fuentes)."""
    lowered = html.lower()
    assert "http" not in lowered, "el tearsheet referencia una URL externa"
    assert "//cdn" not in lowered
    assert "src=" not in lowered, "no debe haber recursos cargados por src"
    assert "<link" not in lowered, "no debe haber hojas de estilo externas"
    assert "@import" not in lowered


# ===========================================================================
# 1. universe
# ===========================================================================


class TestUniverse:
    def test_show_returns_zero_and_reports_membership(self, capsys) -> None:
        rc = main(["universe", "show", "--date", "2010-06-30", "--limit", "5"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "2010-06-30" in out
        assert "miembros: 445" in out  # pertenencia PIT real del fichero semilla
        assert "sector" in out

    def test_show_before_history_fails_explicitly(self, capsys) -> None:
        rc = main(["universe", "show", "--date", "1990-01-01"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "InsufficientHistory" in err

    def test_refresh_dry_run_does_not_write(self, tmp_path, capsys) -> None:
        base = _tickers()
        hist = _fake_history_csv(tmp_path / "hist.csv", ("2024-01-02", base))
        before = hist.read_bytes()
        newer = _fake_history_csv(
            tmp_path / "nuevo.csv",
            ("2024-01-02", base),
            ("2024-01-03", [*base[:-1], "NUEVO"]),
        )
        rc = main([
            "universe", "refresh", "--dry-run",
            "--history", str(hist), "--from-csv", str(newer),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "dry-run" in out
        assert "2024-01-03" in out
        assert hist.read_bytes() == before, "el dry-run jamás toca el fichero"

    def test_refresh_appends_only_new_dates(self, tmp_path, capsys) -> None:
        base = _tickers()
        hist = _fake_history_csv(tmp_path / "hist.csv", ("2024-01-02", base))
        original_first_line = hist.read_text().splitlines()[1]
        newer = _fake_history_csv(
            tmp_path / "nuevo.csv",
            ("2024-01-02", base),
            ("2024-01-03", [*base[:-1], "NUEVO"]),
        )
        rc = main([
            "universe", "refresh", "--history", str(hist), "--from-csv", str(newer),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "fechas añadidas: 2024-01-03" in out
        lines = hist.read_text().splitlines()
        assert len(lines) == 3  # cabecera + snapshot original + snapshot nuevo
        assert lines[1] == original_first_line, "append-only: la historia no se reescribe"
        assert lines[2].startswith("2024-01-03")
        assert "NUEVO" in lines[2]

    def test_refresh_without_reachable_sources_fails(self, tmp_path, capsys) -> None:
        hist = _fake_history_csv(tmp_path / "hist.csv", ("2024-01-02", _tickers()))
        rc = main([
            "universe", "refresh", "--history", str(hist),
            "--from-csv", str(tmp_path / "no_existe.csv"),
        ])
        err = capsys.readouterr().err
        assert rc == 1
        assert "ProviderUnavailable" in err

    def test_local_csv_source_snapshot(self, tmp_path) -> None:
        path = _fake_history_csv(tmp_path / "h.csv", ("2024-01-02", _tickers(501)))
        source = LocalCsvUniverseSource(path)
        assert source.available()
        snap = source.fetch_snapshot()
        assert snap.as_of.isoformat() == "2024-01-02"
        assert snap.n_members == 501
        assert not LocalCsvUniverseSource(tmp_path / "nada.csv").available()


# ===========================================================================
# 2. data status
# ===========================================================================


class TestDataStatus:
    def test_status_reports_providers_and_datasets(self, capsys) -> None:
        rc = main(["data", "status"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "synthetic" in out
        assert "disponible SIEMPRE" in out
        # el consenso real del repo y su advertencia PIT obligatoria
        assert "consenso_master.parquet" in out
        assert "ADVERTENCIA PIT" in out
        assert "NO para momentum de revisiones" in out
        # los proveedores sin credenciales se señalan, no se ocultan
        assert "faltan variables de entorno" in out


# ===========================================================================
# 3. factors
# ===========================================================================


class TestFactors:
    def test_list_contains_both_families(self, capsys) -> None:
        rc = main(["factors", "list"])
        out = capsys.readouterr().out
        assert rc == 0
        # ángulo evento y ángulo fundamental, ambos registrados
        for name in ("sue_analyst", "pead", "piotroski_f", "accruals_cf", "fcf_yield"):
            assert name in out

    def test_compute_reports_ic_with_band_and_writes_csv(self, tmp_path, capsys) -> None:
        out_csv = tmp_path / "sue.csv"
        rc = main(["factors", "compute", "sue_analyst", *MKT_ARGS, "--output", str(out_csv)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "IC" in out
        assert "IC95%" in out  # la banda es obligatoria (contrato §3.8)
        assert "Newey-West" in out
        assert out_csv.exists()
        frame = pd.read_csv(out_csv)
        assert {"date", "ticker", "sue_analyst"}.issubset(frame.columns)
        assert len(frame) > 1000

    def test_compute_unknown_factor_is_config_error(self, capsys) -> None:
        rc = main(["factors", "compute", "factor_inexistente", *MKT_ARGS])
        err = capsys.readouterr().err
        assert rc == 2
        assert "factor desconocido" in err


# ===========================================================================
# 4. events scan
# ===========================================================================


class TestEventsScan:
    def test_scan_with_ground_truth(self, tmp_path, capsys) -> None:
        out_csv = tmp_path / "scan.csv"
        rc = main([
            "events", "scan", *MKT_ARGS,
            "--leak-fraction", "0.3", "--top", "5", "--ground-truth",
            "--output", str(out_csv),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "top 5 eventos por score" in out
        assert "AUC-ROC" in out
        assert "precisión@" in out
        # la nota de tasa base del informe §12.4 acompaña siempre a la validación
        assert "PPV" in out
        # fuentes ausentes declaradas, nunca inventadas
        assert "form4" in out
        dump = pd.read_csv(out_csv)
        assert "informed_trading_score" in dump.columns
        assert dump["event_id"].is_unique

    def test_scan_directional_mode(self, capsys) -> None:
        rc = main(["events", "scan", *MKT_ARGS, "--mode", "directional", "--top", "3"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "directional" in out


# ===========================================================================
# 5. backtest run
# ===========================================================================


class TestBacktestRun:
    def test_cross_mode_prints_sharpe_with_ci(self, capsys) -> None:
        rc = main([
            "backtest", "run", "--mode", "cross", *MKT_ARGS, "--quantiles", "3",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Sharpe anualizado" in out
        assert "IC95%" in out  # ningún Sharpe sin banda de error
        assert "PSR(0)" in out
        assert "costes acumulados" in out

    def test_event_mode_prints_grid_and_multiplicity_warning(self, capsys) -> None:
        rc = main([
            "backtest", "run", "--mode", "event", *MKT_ARGS,
            "--entry-offsets", "1", "--exit-offsets", "5,10",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mean_net" in out
        assert "mean_net_ci_low" in out
        assert "multiplicidad" in out  # aviso obligatorio de la rejilla

    def test_event_mode_prepositioning_requires_declaration(self, capsys) -> None:
        """Entrar antes del anuncio sin declarar el calendario es LookAheadError."""
        rc = main([
            "backtest", "run", "--mode", "event", *MKT_ARGS,
            "--entry-offsets", "-3", "--exit-offsets", "5",
        ])
        captured = capsys.readouterr()
        assert rc == 1
        assert "LookAheadError" in captured.err
        assert "nota PIT" in captured.out

    def test_event_mode_prepositioning_with_declaration_runs(self, capsys) -> None:
        rc = main([
            "backtest", "run", "--mode", "event", *MKT_ARGS,
            "--entry-offsets", "-3", "--exit-offsets", "5",
            "--calendar-known",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mean_net" in out


# ===========================================================================
# 6. collector (passthrough)
# ===========================================================================


class TestCollector:
    def test_run_dry_run_writes_nothing(self, tmp_path, capsys) -> None:
        root = tmp_path / "store"
        rc = main([
            "collector", "run", "--dry-run",
            "--date", "2026-08-05",
            "--root", str(root),
            "--options-source", "synthetic",
            "--calendar-provider", "synthetic",
            "--budget", "100",
            "--baseline", "5",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "DRY-RUN" in out
        assert "plan de la pasada 'close'" in out
        assert not any(root.rglob("*.parquet")), "el dry-run no escribe nada"

    def test_collector_without_args_is_usage_error(self, capsys) -> None:
        rc = main(["collector"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "run --dry-run" in err


# ===========================================================================
# 7. report (tearsheet completo end-to-end)
# ===========================================================================


@pytest.fixture(scope="module")
def report_html(tmp_path_factory) -> str:
    """Genera el tearsheet completo una sola vez para todos los asserts."""
    out = tmp_path_factory.mktemp("report") / "tearsheet.html"
    rc = main([
        "report", *MKT_ARGS,
        "--quantiles", "3",
        "--entry-offsets", "1", "--exit-offsets", "5,10",
        "--caar-groups", "3",
        "--output", str(out),
    ])
    assert rc == 0
    return out.read_text(encoding="utf-8")


class TestReportCommand:
    def test_contains_every_section(self, report_html: str) -> None:
        for sec_id in ALL_SECTIONS:
            assert f'id="{sec_id}"' in report_html, f"falta la sección {sec_id!r}"

    def test_is_self_contained(self, report_html: str) -> None:
        _assert_self_contained(report_html)

    def test_has_light_and_dark_theme(self, report_html: str) -> None:
        assert "prefers-color-scheme: dark" in report_html
        assert 'data-theme="dark"' in report_html
        assert 'data-theme="light"' in report_html
        assert "theme-toggle" in report_html

    def test_charts_are_hand_drawn_svg(self, report_html: str) -> None:
        assert report_html.count("<svg") >= 5
        # coordenadas numéricas válidas en todos los trazos
        for points in re.findall(r'points="([^"]+)"', report_html):
            for pair in points.split():
                x, y = pair.split(",")
                assert np.isfinite(float(x)) and np.isfinite(float(y))

    def test_grid_table_present_with_ci_columns(self, report_html: str) -> None:
        assert "run_grid" in report_html
        assert "mean_net_ci_low" in report_html
        assert "mean_net_ci_high" in report_html
        # advertencia de multiplicidad de la rejilla (validation_methodology §7)
        assert "Hochberg" in report_html

    def test_metrics_table_reports_ci(self, report_html: str) -> None:
        assert "IC 95%" in report_html
        assert "Sharpe anualizado" in report_html
        assert "PSR" in report_html

    def test_params_echo_for_reproducibility(self, report_html: str) -> None:
        assert "seed" in report_html
        assert "leak_fraction" in report_html
        assert "synthetic" in report_html


# ===========================================================================
# 8. Tearsheet como biblioteca
# ===========================================================================


def _toy_inputs() -> dict[str, object]:
    dates = pd.bdate_range("2021-01-04", periods=120)
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(5e-4, 0.01, len(dates)), index=dates)
    ic = pd.Series(rng.normal(0.03, 0.12, len(dates)), index=dates)
    quants = pd.DataFrame(
        {f"q{i}": rng.normal(2e-4 * i, 0.01, len(dates)) for i in range(1, 4)}, index=dates
    )
    tau = pd.Index(np.arange(-5, 11), name="tau")
    caar = pd.DataFrame(
        {"caar": np.cumsum(rng.normal(1e-3, 2e-3, len(tau))), "caar_se": 0.003},
        index=tau,
    )
    idx = pd.MultiIndex.from_product([[1], [5, 10]], names=["entry_offset", "exit_offset"])
    grid = pd.DataFrame(
        {
            "status": ["ok", "ok"],
            "n_events": [40, 40],
            "hit_rate": [0.55, 0.52],
            "mean_net": [0.004, -0.001],
            "mean_net_ci_low": [0.001, -0.004],
            "mean_net_ci_high": [0.007, 0.002],
        },
        index=idx,
    )
    return {
        "returns": rets,
        "ic": ic,
        "quantile_returns": quants,
        "caar": caar,
        "grid": grid,
        "metrics": [MetricWithCI("Sharpe anualizado", 1.1, 0.2, 2.0)],
        "params": {"seed": 3},
    }


class TestTearsheetLibrary:
    def test_all_sections_rendered(self) -> None:
        html = render_tearsheet(title="prueba", **_toy_inputs())
        for sec_id in SECTION_IDS:
            assert f'id="{sec_id}"' in html
        _assert_self_contained(html)

    def test_partial_inputs_render_partial_sections(self) -> None:
        data = _toy_inputs()
        html = render_tearsheet(title="solo equity", returns=data["returns"])
        assert 'id="equity"' in html
        assert 'id="drawdown"' in html  # derivado de la curva de equity
        for absent in ("ic", "quantiles", "caar", "grid", "metrics"):
            assert f'id="{absent}"' not in html

    def test_empty_tearsheet_is_an_error(self) -> None:
        with pytest.raises(DataQualityError, match="ninguna sección"):
            render_tearsheet(title="vacío")

    def test_metric_ci_validation(self) -> None:
        with pytest.raises(DataQualityError, match="ci_low"):
            MetricWithCI("m", 1.0, ci_low=2.0, ci_high=1.0)
        with pytest.raises(DataQualityError, match="ambos extremos"):
            MetricWithCI("m", 1.0, ci_low=0.5)

    def test_bare_numbers_rejected_in_metrics(self) -> None:
        data = _toy_inputs()
        with pytest.raises(DataQualityError, match="MetricWithCI"):
            render_tearsheet(title="x", returns=data["returns"], metrics=[1.23])  # type: ignore[list-item]

    def test_grid_requires_run_grid_schema(self) -> None:
        bad = pd.DataFrame({"mean_net": [0.1]})
        with pytest.raises(DataQualityError, match="entry_offset"):
            render_tearsheet(title="x", grid=bad)

    def test_caar_multiindex_renders_one_series_per_group(self) -> None:
        tau = np.arange(-3, 6)
        idx = pd.MultiIndex.from_product([["Q1", "Q3"], tau], names=["group", "tau"])
        caar = pd.DataFrame(
            {"caar": np.linspace(-0.01, 0.02, len(idx)), "caar_se": 0.004}, index=idx
        )
        html = render_tearsheet(title="caar", caar=caar)
        assert "Q1" in html
        assert "Q3" in html
        assert 'class="band"' in html  # banda ±1,96·SE por grupo

    def test_write_tearsheet_creates_parents(self, tmp_path) -> None:
        out = tmp_path / "sub" / "dir" / "ts.html"
        data = _toy_inputs()
        path = write_tearsheet(out, title="w", returns=data["returns"])
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_helper_converters_duck_type(self) -> None:
        class FakeSharpe:
            sharpe_annualized = 1.5
            ci_low = 0.5
            ci_high = 2.5
            psr_zero = 0.97
            method = "mertens"
            n_obs = 500

        metrics = sharpe_metrics(FakeSharpe())
        assert metrics[0].ci_low == 0.5
        summary = {
            "mean_ic": 0.05, "ci_low": 0.01, "ci_high": 0.09, "t_nw": 2.4,
            "nw_lags": 5, "n_periods": 400, "ic_ir_annualized": 1.2, "horizon": 5,
        }
        ics = ic_metrics(summary)
        assert ics[0].ci_high == 0.09
        with pytest.raises(DataQualityError):
            sharpe_metrics(object())


# ===========================================================================
# 9. Parser y códigos de salida
# ===========================================================================


class TestParser:
    def test_no_arguments_is_usage_error(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_unknown_command_is_usage_error(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["comando_falso"])
        assert exc.value.code == 2

    def test_bad_date_is_usage_error(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["universe", "show", "--date", "31/12/2020"])
        assert exc.value.code == 2

    def test_bad_offset_list_is_usage_error(self) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["backtest", "run", "--mode", "event", "--entry-offsets", "a,b"])
        assert exc.value.code == 2

    def test_every_documented_subcommand_exists(self) -> None:
        parser = build_parser()
        text = parser.format_help()
        for cmd in ("universe", "data", "factors", "events", "backtest", "collector", "report"):
            assert cmd in text
