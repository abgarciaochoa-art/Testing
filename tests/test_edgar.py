"""Tests sin red de `data.edgar` y `data.fundamentals`.

Todos los fixtures son JSON/XML recortados a mano en `tests/fixtures/edgar/`
(sec_edgar.md §11.4) y el tráfico va contra `ScriptedTransport`: **ningún test
abre un socket**. La lista de fixtures ejercita exactamente las trampas del
informe: paginación de `files[]`, arrays desalineados, fronteras BMO/AMC
(16:01 ET → AMC, 08:30 ET → BMO, viernes tarde → lunes), vintages y reexpresión,
el 10-Q que mezcla 3M/6M/comparativos con el mismo `fy`/`fp`, la migración de
tag de ingresos (ASC 606), los códigos de Form 4 (P/S frente a A/F/M), el 403 de
la SEC que en realidad es limitación de tasa, y el `relation="gte"` de la
búsqueda a texto completo.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from earnings_alpha.data.base import HttpResponse, ScriptedTransport
from earnings_alpha.data.edgar import (
    EdgarProvider,
    company_tickers_to_frame,
    companyfacts_to_frame,
    classify_periods,
    detect_earnings_8k,
    facts_to_fundamental_facts,
    filed_available_at,
    form4_code_distribution,
    form4_to_frame,
    frame_to_earnings_events,
    insider_net_buys,
    parse_acceptance_datetime,
    parse_form4,
    submissions_to_frame,
)
from earnings_alpha.data.fundamentals import (
    CANONICAL_COLUMNS,
    EODHDFundamentals,
    EdgarFundamentals,
    FMPFundamentals,
    FundamentalsService,
    build_default_registry,
    derive_q4,
    fundamentals_panel,
    reconcile_total_debt,
    resolve_company_series,
    resolve_company_tag,
)
from earnings_alpha.errors import (
    DataQualityError,
    InsufficientHistory,
    ProviderUnavailable,
    RateLimited,
)
from earnings_alpha.types import FundamentalFact, Session

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"

SEC_UA = "earnings-alpha tests test@example.com"


def _load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _load_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _subs_frame() -> pd.DataFrame:
    return submissions_to_frame(
        _load_json("submissions_recent.json"), [_load_json("submissions_page_001.json")]
    )


def _facts_frame() -> pd.DataFrame:
    return companyfacts_to_frame(_load_json("companyfacts_vintages.json"))


def _dispatch(routes: dict[str, bytes]) -> ScriptedTransport:
    """Transporte guionizado que enruta por subcadena de la URL."""

    def handler(request: Any) -> HttpResponse:
        for needle, body in routes.items():
            if needle in request.url:
                return HttpResponse(200, content=body, url=request.url)
        msg = f"URL no guionizada en el test: {request.url}"
        raise AssertionError(msg)

    return ScriptedTransport(handler, repeat_last=True)


# ===========================================================================
# 1. Instantes: acceptanceDateTime y filed
# ===========================================================================


class TestParseAcceptanceDatetime:
    def test_sufijo_z_es_hora_de_nueva_york(self) -> None:
        # EDGAR sirve la hora del Este con un sufijo Z engañoso: 16:35 ET en
        # mayo (EDT, UTC-4) son las 20:35 UTC.
        out = parse_acceptance_datetime("2024-05-01T16:35:12.000Z")
        assert out == datetime(2024, 5, 1, 20, 35, 12)

    def test_horario_de_invierno(self) -> None:
        out = parse_acceptance_datetime("2024-01-15T16:35:12.000Z")
        assert out == datetime(2024, 1, 15, 21, 35, 12)  # EST: UTC-5

    def test_desplazamiento_explicito_se_respeta(self) -> None:
        out = parse_acceptance_datetime("2024-05-01T16:35:12-04:00")
        assert out == datetime(2024, 5, 1, 20, 35, 12)

    def test_formato_compacto_sgml(self) -> None:
        out = parse_acceptance_datetime("20240501163512")
        assert out == datetime(2024, 5, 1, 20, 35, 12)

    def test_vacio_o_basura_falla_explicitamente(self) -> None:
        with pytest.raises(DataQualityError):
            parse_acceptance_datetime("")
        with pytest.raises(DataQualityError):
            parse_acceptance_datetime("no-es-una-fecha")

    def test_assume_eastern_false(self) -> None:
        out = parse_acceptance_datetime("2024-05-01T16:35:12", assume_eastern=False)
        assert out == datetime(2024, 5, 1, 16, 35, 12)


class TestFiledAvailableAt:
    def test_corte_general_1730_et(self) -> None:
        # 17:30 EDT (agosto) = 21:30 UTC: cota superior del instante real.
        assert filed_available_at("2024-08-01", "10-Q") == datetime(2024, 8, 1, 21, 30)

    def test_corte_general_en_invierno(self) -> None:
        assert filed_available_at("2024-01-15", "10-K") == datetime(2024, 1, 15, 22, 30)

    def test_formularios_de_insiders_hasta_2200_et(self) -> None:
        # Los Form 3/4/5 conservan el filingDate hasta las 22:00 ET, que en
        # verano cruza la medianoche UTC.
        assert filed_available_at("2024-08-01", "4") == datetime(2024, 8, 2, 2, 0)


# ===========================================================================
# 2. Identidad
# ===========================================================================


class TestCompanyTickers:
    def test_forma_columnar(self) -> None:
        df = company_tickers_to_frame(_load_json("company_tickers_exchange.json"))
        assert list(df.columns) == ["cik", "ticker", "name", "exchange"]
        assert len(df) == 6
        # BRK-B (convención EDGAR) -> BRK.B (convención del repo).
        assert "BRK.B" in set(df["ticker"])
        assert "BRK-B" not in set(df["ticker"])
        # CIK->ticker es uno-a-muchos: GOOGL y GOOG comparten emisor.
        alphabet = df.loc[df["cik"] == "0001652044", "ticker"]
        assert set(alphabet) == {"GOOGL", "GOOG"}

    def test_forma_objeto(self) -> None:
        payload = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
        }
        df = company_tickers_to_frame(payload)
        assert set(df["ticker"]) == {"AAPL", "MSFT"}
        assert set(df["cik"]) == {"0000320193", "0000789019"}
        assert df["exchange"].isna().all()

    def test_payload_invalido(self) -> None:
        with pytest.raises(DataQualityError):
            company_tickers_to_frame({"0": {"cik_str": 1, "title": "sin ticker"}})


# ===========================================================================
# 3. submissions
# ===========================================================================


class TestSubmissions:
    def test_paginacion_incluye_el_historico(self) -> None:
        df = _subs_frame()
        assert len(df) == 17  # 15 recientes + 2 de la página extra (trampa nº 5)
        assert df["filing_date"].min() == pd.Timestamp("2010-02-26")

    def test_accepted_at_es_utc_desde_hora_del_este(self) -> None:
        df = _subs_frame()
        row = df.loc[df["accession"] == "0001111111-24-000044"].iloc[0]
        assert row["accepted_at"] == pd.Timestamp("2024-05-01 20:35:12")

    def test_arrays_desalineados_lanzan(self) -> None:
        # Un pd.DataFrame(bloque) desplazaría fechas en silencio (trampa nº 4).
        with pytest.raises(DataQualityError, match="desalineado"):
            submissions_to_frame(_load_json("submissions_misaligned.json"))

    def test_bloque_sin_accession_lanza(self) -> None:
        with pytest.raises(DataQualityError):
            submissions_to_frame({"filings": {"recent": {"form": ["8-K"]}, "files": []}})


# ===========================================================================
# 4. 8-K item 2.02: sesiones y fecha negociable
# ===========================================================================


class TestEarnings8K:
    @pytest.fixture()
    def events(self) -> pd.DataFrame:
        return detect_earnings_8k(_subs_frame())

    def _row(self, events: pd.DataFrame, report_date: str) -> pd.Series:
        hit = events.loc[events["report_date"] == pd.Timestamp(report_date)]
        assert len(hit) == 1, f"esperaba 1 evento para {report_date}, hay {len(hit)}"
        return hit.iloc[0]

    def test_numero_de_eventos(self, events: pd.DataFrame) -> None:
        # 7 trimestres en `recent` + 1 en la página de 2010 (paginación).
        assert len(events) == 8

    def test_amc_1635(self, events: pd.DataFrame) -> None:
        row = self._row(events, "2024-03-30")
        assert row["session"] == "amc"
        assert row["tradable"] == pd.Timestamp("2024-05-02")  # sesión siguiente
        # y es el 8-K correcto: ni el 8-K/A ni el de items "12.02".
        assert row["accession"] == "0001111111-24-000044"

    def test_frontera_1601_es_amc(self, events: pd.DataFrame) -> None:
        # 16:01 ET: la subasta de cierre ya se ejecutó -> AMC, siguiente sesión.
        row = self._row(events, "2024-09-28")
        assert row["session"] == "amc"
        assert row["tradable"] == pd.Timestamp("2024-10-25")

    def test_frontera_0830_es_bmo(self, events: pd.DataFrame) -> None:
        # 08:30 ET < apertura -> BMO, negociable la misma sesión.
        row = self._row(events, "2025-03-29")
        assert row["session"] == "bmo"
        assert row["tradable"] == pd.Timestamp("2025-04-24")

    def test_bmo_0715(self, events: pd.DataFrame) -> None:
        row = self._row(events, "2023-12-30")
        assert row["session"] == "bmo"
        assert row["tradable"] == pd.Timestamp("2024-02-01")

    def test_dmh_marca_baja_confianza(self, events: pd.DataFrame) -> None:
        # 09:41 ET, mercado abierto: probablemente era BMO y el 8-K llegó tarde;
        # se marca para que EventBacktest pueda excluirlo (sec_edgar.md §7.3).
        row = self._row(events, "2023-09-30")
        assert row["session"] == "dmh"
        assert bool(row["low_confidence_session"]) is True
        assert row["tradable"] == pd.Timestamp("2023-11-01")

    def test_viernes_tarde_negociable_el_lunes(self, events: pd.DataFrame) -> None:
        row = self._row(events, "2024-12-28")
        assert row["session"] == "amc"
        assert pd.Timestamp(row["tradable"]).dayofweek == 0
        assert row["tradable"] == pd.Timestamp("2025-02-03")

    def test_duplicado_gana_el_primero_en_aceptarse(self, events: pd.DataFrame) -> None:
        # Dos 8-K 2.02 para el mismo reportDate: el primero movió el precio.
        row = self._row(events, "2024-06-29")
        assert row["accession"] == "0001111111-24-000075"
        assert row["accepted_at"] == pd.Timestamp("2024-07-25 20:05:00")
        assert row["tradable"] == pd.Timestamp("2024-07-26")

    def test_enmienda_8ka_excluida(self, events: pd.DataFrame) -> None:
        assert "0001111111-24-000061" not in set(events["accession"])

    def test_items_se_tokeniza(self, events: pd.DataFrame) -> None:
        # "12.02" no contiene el token "2.02"; un `in` sobre la cadena cruda sí
        # habría casado (y habría ganado la deduplicación por fecha).
        assert "0001111111-24-000041" not in set(events["accession"])
        # y el 7.01 puro tampoco entra.
        assert "0001111111-24-000091A" not in set(events["accession"])

    def test_evento_antiguo_de_la_paginacion(self, events: pd.DataFrame) -> None:
        row = self._row(events, "2010-06-30")
        assert row["session"] == "amc"
        assert row["tradable"] == pd.Timestamp("2010-07-21")

    def test_columnas_obligatorias(self) -> None:
        with pytest.raises(DataQualityError, match="submissions_to_frame"):
            detect_earnings_8k(pd.DataFrame({"form": ["8-K"]}))

    def test_frame_a_earnings_events(self, events: pd.DataFrame) -> None:
        out = frame_to_earnings_events(events, "TSTA", "0001111111")
        assert len(out) == 8
        by_pe = {e.period_end: e for e in out}
        ev = by_pe[date(2024, 3, 30)]
        assert ev.session is Session.AMC
        assert ev.fiscal_quarter == "2024Q1"
        assert ev.announced_at == datetime(2024, 5, 1, 20, 35, 12)
        assert ev.cik == "0001111111"
        assert ev.eps_actual is None  # el 8-K aporta el timestamp, no cifras


# ===========================================================================
# 5. companyfacts: vintages, fy/fp, duraciones
# ===========================================================================


class TestCompanyFacts:
    def test_vintages_y_reexpresion(self) -> None:
        df = _facts_frame()
        rev_q1 = df[
            (df["concept"] == "Revenues") & (df["end"] == pd.Timestamp("2017-04-01"))
        ].sort_values("filed")
        assert len(rev_q1) == 2
        assert list(rev_q1["n_vintages"]) == [2, 2]
        assert list(rev_q1["is_restated"]) == [False, True]  # cambia 100 -> 95
        assert (rev_q1["first_filed"] == pd.Timestamp("2017-05-03")).all()
        assert bool(rev_q1.iloc[1]["is_amendment"]) is True  # 10-K/A

    def test_republicacion_sin_cambio_no_es_reexpresion(self) -> None:
        payload = {
            "cik": 1111111,
            "entityName": "Testa Corp",
            "facts": {
                "us-gaap": {
                    "Assets": {
                        "label": "Assets",
                        "units": {
                            "USD": [
                                {"end": "2024-03-31", "val": 100, "accn": "a-1",
                                 "fy": 2024, "fp": "Q1", "form": "10-Q",
                                 "filed": "2024-05-01"},
                                {"end": "2024-03-31", "val": 100, "accn": "a-2",
                                 "fy": 2024, "fp": "Q2", "form": "10-Q",
                                 "filed": "2024-08-01"},
                            ]
                        },
                    }
                }
            },
        }
        df = companyfacts_to_frame(payload)
        assert not df["is_restated"].any()
        assert (df["n_vintages"] == 2).all()

    def test_fy_fp_no_determinan_el_periodo(self) -> None:
        # El 10-Q de 2024Q2 publica 3M, YTD y comparativos, TODOS con fy=2024
        # fp=Q2 (trampa nº 7). Solo (start, end) separa el trimestre.
        df = _facts_frame()
        ni = df[df["concept"] == "NetIncomeLoss"]
        assert (ni["fy"] == 2024).all()
        assert (ni["fp"] == "Q2").all()
        classified = classify_periods(ni)
        quarters = classified[classified["period_kind"] == "quarter"]
        cumulative = classified[classified["period_kind"] == "cumulative"]
        assert len(quarters) == 2  # 2024Q2 y el comparativo 2023Q2
        assert len(cumulative) == 2  # los dos acumulados de 6 meses
        current = quarters[quarters["end"] == pd.Timestamp("2024-06-30")]
        assert len(current) == 1
        assert current.iloc[0]["value"] == 20_000_000

    def test_available_at_desde_filed_con_corte(self) -> None:
        df = _facts_frame()
        first = df[
            (df["concept"] == "Revenues")
            & (df["end"] == pd.Timestamp("2017-04-01"))
            & (df["filed"] == pd.Timestamp("2017-05-03"))
        ].iloc[0]
        # 17:30 EDT del día `filed` = 21:30 UTC; jamás medianoche del filingDate.
        assert first["available_at"] == pd.Timestamp("2017-05-03 21:30:00")

    def test_instantaneos_sin_start(self) -> None:
        df = _facts_frame()
        debt = df[df["concept"] == "LongTermDebtNoncurrent"]
        assert debt["is_instant"].all()
        assert classify_periods(debt)["period_kind"].eq("instant").all()

    def test_tag_obsoleto_marcado(self) -> None:
        df = _facts_frame()
        assert df.loc[df["concept"] == "AccountsPayable", "deprecated"].all()
        assert not df.loc[df["concept"] == "Assets", "deprecated"].any()

    def test_sin_facts_lanza(self) -> None:
        with pytest.raises(DataQualityError):
            companyfacts_to_frame({"cik": 1, "facts": {}})

    def test_facts_a_fundamental_facts_con_acceptance_exacto(self) -> None:
        df = classify_periods(_facts_frame())
        target = df[
            (df["concept"] == "NetIncomeLoss") & (df["period_kind"] == "quarter")
        ]
        subs = pd.DataFrame(
            {
                "accession": ["0001111111-24-000042"],
                "accepted_at": [pd.Timestamp("2024-08-01 20:45:00")],
            }
        )
        facts = facts_to_fundamental_facts(target, "TSTA", submissions=subs)
        assert all(isinstance(f, FundamentalFact) for f in facts)
        current = next(f for f in facts if f.period_end == date(2024, 6, 30))
        # available_at exacto del acceptanceDateTime, no el corte de 17:30.
        assert current.available_at == datetime(2024, 8, 1, 20, 45)
        assert current.accession == "0001111111-24-000042"
        assert current.form == "10-Q"
        assert current.fiscal_period == "2024Q2"


# ===========================================================================
# 6. fundamentals: cascada estable, deuda, Q4, vintages
# ===========================================================================


class TestResolveCompanyTag:
    def test_migracion_asc606_sin_salto(self) -> None:
        facts = _facts_frame()
        tags, rows = resolve_company_tag(facts, "revenue")
        assert tags == [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
        ]
        # Serie as-first-reported continua: 12 trimestres sin hueco ni escalón.
        first = rows[(rows["filed"] == rows["first_filed"])].sort_values("end")
        quarters = first[first["duration_days"].between(80, 100)]
        assert len(quarters) == 12
        expected = [100 + 4 * i for i in range(12)]
        assert [v / 1e6 for v in quarters["value"]] == expected
        # El tag antiguo cubre 2017; el moderno, 2018 en adelante.
        assert (
            quarters.loc[quarters["end"] <= pd.Timestamp("2017-12-30"), "concept_used"]
            == "Revenues"
        ).all()
        assert (
            quarters.loc[quarters["end"] >= pd.Timestamp("2018-01-01"), "concept_used"]
            == "RevenueFromContractWithCustomerExcludingAssessedTax"
        ).all()
        assert quarters.loc[
            quarters["end"] >= pd.Timestamp("2018-01-01"), "tag_switch"
        ].all()

    def test_divergencia_en_el_solape_lanza(self) -> None:
        facts = _facts_frame().copy()
        mask = (facts["concept"] == "Revenues") & (
            facts["end"] == pd.Timestamp("2018-03-31")
        )
        facts.loc[mask, "value"] = 200_000_000  # 72% de divergencia
        with pytest.raises(DataQualityError, match="no se empalman"):
            resolve_company_tag(facts, "revenue")

    def test_concepto_ausente_lanza_insufficient_history(self) -> None:
        with pytest.raises(InsufficientHistory):
            resolve_company_tag(_facts_frame(), "capex")

    def test_concepto_desconocido(self) -> None:
        with pytest.raises(DataQualityError):
            resolve_company_tag(_facts_frame(), "ebitda_ajustado_magico")

    def test_un_solo_tag_cubre_todo(self) -> None:
        tags, rows = resolve_company_tag(_facts_frame(), "net_income")
        assert tags == ["NetIncomeLoss"]
        assert not rows["tag_switch"].any()


class TestReconcileTotalDebt:
    def test_componentes_antes_que_agregado(self) -> None:
        out = reconcile_total_debt(_facts_frame())
        jun = out[out["end"] == pd.Timestamp("2024-06-30")].iloc[0]
        # 800 no corriente + 200 corriente + 50 corto plazo; NUNCA el agregado
        # LongTermDebt=800 aunque exista (omitiría la parte corriente).
        assert jun["value"] == 1_050_000_000
        assert jun["concept_used"] == (
            "sum(LongTermDebtNoncurrent+LongTermDebtCurrent+ShortTermBorrowings)"
        )

    def test_fallback_al_agregado_cuando_faltan_componentes(self) -> None:
        out = reconcile_total_debt(_facts_frame())
        dec = out[out["end"] == pd.Timestamp("2023-12-31")].iloc[0]
        assert dec["value"] == 900_000_000
        assert dec["concept_used"] == "LongTermDebtAndFinanceLeaseObligations"

    def test_sin_tags_de_deuda(self) -> None:
        facts = _facts_frame()
        sin_deuda = facts[~facts["concept"].str.contains("Debt|Borrowings")]
        with pytest.raises(InsufficientHistory):
            reconcile_total_debt(sin_deuda)


class TestDeriveQ4:
    def test_q4_es_fy_menos_q123(self) -> None:
        resolved = resolve_company_series(_facts_frame(), ["cfo"])
        q4 = derive_q4(resolved)
        assert len(q4) == 1
        row = q4.iloc[0]
        assert row["end"] == pd.Timestamp("2023-12-31")
        assert row["value"] == 17_000_000  # 50 - (10+12+11)
        assert bool(row["is_derived"]) is True
        assert row["period_kind"] == "quarter"
        # available_at del 10-K (2024-02-20 + corte 17:30 EST = 22:30 UTC),
        # no el cierre del trimestre: el Q4 se conoce al publicarse el anual.
        assert row["available_at"] == pd.Timestamp("2024-02-20 22:30:00")

    def test_no_deriva_saldos_de_balance(self) -> None:
        resolved = resolve_company_series(_facts_frame(), ["total_assets"])
        assert len(derive_q4(resolved)) == 0

    def test_columnas_obligatorias(self) -> None:
        with pytest.raises(DataQualityError):
            derive_q4(pd.DataFrame({"concept": ["cfo"]}))


class TestFundamentalsPanel:
    def _canon(self) -> pd.DataFrame:
        resolved = resolve_company_series(_facts_frame(), ["revenue"])
        return (
            resolved.rename(columns={"end": "period_end"})
            .loc[:, ["concept", "period_end", "available_at", "value", "accession"]]
            .assign(ticker="TSTA")
        )

    def test_original_es_lo_primero_reportado(self) -> None:
        panel = fundamentals_panel(self._canon(), vintage="original")
        q1 = panel[panel["period_end"] == pd.Timestamp("2017-04-01")]
        assert q1.iloc[0]["value"] == 100_000_000

    def test_pit_respeta_el_corte(self) -> None:
        canon = self._canon()
        antes = fundamentals_panel(canon, as_of=date(2017, 6, 1), vintage="pit")
        q1 = antes[antes["period_end"] == pd.Timestamp("2017-04-01")]
        assert q1.iloc[0]["value"] == 100_000_000  # la reexpresión aún no existía
        despues = fundamentals_panel(canon, as_of=date(2018, 3, 1), vintage="pit")
        q1 = despues[despues["period_end"] == pd.Timestamp("2017-04-01")]
        assert q1.iloc[0]["value"] == 95_000_000  # ya publicada: conocible

    def test_pit_sin_as_of_lanza(self) -> None:
        with pytest.raises(DataQualityError, match="as_of"):
            fundamentals_panel(self._canon(), vintage="pit")

    def test_latest_es_lookahead_documentado(self) -> None:
        panel = fundamentals_panel(self._canon(), vintage="latest")
        q1 = panel[panel["period_end"] == pd.Timestamp("2017-04-01")]
        assert q1.iloc[0]["value"] == 95_000_000
        assert panel.attrs["vintage"] == "latest"

    def test_as_of_anterior_a_todo_lanza(self) -> None:
        with pytest.raises(InsufficientHistory):
            fundamentals_panel(self._canon(), as_of=date(2000, 1, 1), vintage="pit")

    def test_vacio_lanza(self) -> None:
        with pytest.raises(InsufficientHistory):
            fundamentals_panel(self._canon().iloc[0:0], vintage="original")


# ===========================================================================
# 7. Form 4
# ===========================================================================


class TestForm4Parsing:
    def test_valores_envueltos_y_footnote(self) -> None:
        parsed = parse_form4(_load_bytes("form4_codes.xml").decode())
        assert parsed["issuer"]["ticker"] == "TSTA"
        assert parsed["issuer"]["cik"] == "0001111111"
        assert len(parsed["owners"]) == 1
        assert parsed["owners"][0]["is_officer"] is True
        assert parsed["owners"][0]["officer_title"] == "Chief Financial Officer"
        txns = parsed["transactions"]
        assert len(txns) == 6
        compra = next(t for t in txns if t["code"] == "P")
        # El precio convive con un <footnoteId/> dentro del mismo elemento.
        assert compra["price"] == 50.0
        assert compra["shares"] == 1000
        assert parsed["footnotes"]["F1"].startswith("Weighted average")

    def test_reporting_owner_como_lista(self) -> None:
        parsed = parse_form4(_load_bytes("form4_multi_owner.xml").decode())
        assert len(parsed["owners"]) == 2
        assert parsed["owners"][0]["is_ten_percent_owner"] is True
        # El ticker con clase se normaliza a la convención del repo.
        assert parsed["issuer"]["ticker"] == "TSTA.A"
        assert parsed["transactions"][0]["shares"] == pytest.approx(34.1689)

    def test_xml_sucio_se_sanea(self) -> None:
        # El fixture multi-owner trae un `&` sin escapar en natureOfOwnership;
        # además se inyectan caracteres de control.
        dirty = _load_bytes("form4_multi_owner.xml").decode().replace(
            "SMITH JOHN", "SMITH\x01 JOHN"
        )
        parsed = parse_form4(dirty)
        assert parsed["owners"][1]["name"] == "SMITH JOHN"

    def test_no_ownership_document(self) -> None:
        with pytest.raises(DataQualityError):
            parse_form4("<html><body>no soy un form 4</body></html>")

    def test_frame_no_duplica_por_propietario(self) -> None:
        parsed = parse_form4(_load_bytes("form4_multi_owner.xml").decode())
        df = form4_to_frame(parsed, accession="ACC9", available_at="2024-05-02 22:30")
        assert len(df) == 1  # una transacción conjunta = UNA fila
        assert df.iloc[0]["n_owners"] == 2
        assert df.iloc[0]["owner_ciks"] == "0002222222;0003333333"
        assert bool(df.iloc[0]["is_director"]) is True


class TestInsiderNetBuys:
    def _frame(self) -> pd.DataFrame:
        parsed = parse_form4(_load_bytes("form4_codes.xml").decode())
        return form4_to_frame(
            parsed, accession="ACC1", available_at=pd.Timestamp("2024-05-03 21:15")
        )

    def test_solo_p_y_s_discrecionales(self) -> None:
        out = insider_net_buys(self._frame())
        row = out.loc[("TSTA", pd.Timestamp("2024-05-03"))]
        assert row["buy_value"] == 50_000  # P: 1000 x 50
        assert row["sell_value"] == 26_000  # S del 05-02, no ligada
        # La S del 05-03 va con la M del mismo filing y fecha: ejercicio
        # preprogramado, fuera del neto (trampa nº 9 / 10b5-1).
        assert row["sell_linked_value"] == 104_000  # 2000 x 52
        assert row["net_buy_value"] == 24_000
        assert row["n_routine_trades"] == 3  # A, F, M
        assert row["n_buy_trades"] == 1
        assert row["n_distinct_buyers"] == 1

    def test_distribucion_de_codigos(self) -> None:
        dist = form4_code_distribution(self._frame())
        assert dist.to_dict() == {"S": 2, "P": 1, "M": 1, "A": 1, "F": 1}

    def test_agrega_por_publicacion_no_por_transaccion(self) -> None:
        # Todas las transacciones (01..03 de mayo) colapsan a la fecha de
        # publicación del Form 4: "compras PUBLICADAS", no "realizadas".
        out = insider_net_buys(self._frame())
        assert list(out.index.get_level_values("date").unique()) == [
            pd.Timestamp("2024-05-03")
        ]

    def test_sin_available_at_lanza(self) -> None:
        parsed = parse_form4(_load_bytes("form4_codes.xml").decode())
        df = form4_to_frame(parsed, accession="ACC1", available_at=None)
        with pytest.raises(DataQualityError, match="look-ahead"):
            insider_net_buys(df)

    def test_vacio_lanza(self) -> None:
        with pytest.raises(InsufficientHistory):
            insider_net_buys(self._frame().iloc[0:0])


# ===========================================================================
# 8. EdgarProvider contra transporte guionizado
# ===========================================================================


class TestEdgarProvider:
    def test_sin_user_agent_no_toca_la_red(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEC_USER_AGENT", raising=False)
        transport = _dispatch({})
        provider = EdgarProvider(transport=transport)
        assert provider.available() is False
        with pytest.raises(ProviderUnavailable) as exc:
            provider.company_facts("1111111")
        assert "SEC_USER_AGENT" in exc.value.missing_env
        assert transport.n_calls == 0

    def test_403_con_cuerpo_de_tasa_es_rate_limited(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEC_USER_AGENT", SEC_UA)
        transport = ScriptedTransport(
            HttpResponse(403, content=_load_bytes("rate_limit_403.html")),
            repeat_last=True,
        )
        provider = EdgarProvider(transport=transport)
        with pytest.raises(RateLimited):
            provider.company_facts("1111111")

    def test_403_generico_es_provider_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SEC_USER_AGENT", SEC_UA)
        transport = ScriptedTransport(
            HttpResponse(403, content=b"<html>Forbidden</html>"), repeat_last=True
        )
        provider = EdgarProvider(transport=transport)
        with pytest.raises(ProviderUnavailable):
            provider.company_facts("1111111")

    def _provider(self, monkeypatch: pytest.MonkeyPatch) -> EdgarProvider:
        monkeypatch.setenv("SEC_USER_AGENT", SEC_UA)
        routes = {
            "submissions/CIK0001111111-submissions-001.json": _load_bytes(
                "submissions_page_001.json"
            ),
            "submissions/CIK0001111111.json": _load_bytes("submissions_recent.json"),
            "companyfacts/CIK0001111111.json": _load_bytes(
                "companyfacts_vintages.json"
            ),
            "company_tickers_exchange.json": _load_bytes(
                "company_tickers_exchange.json"
            ),
            "form4-officer.xml": _load_bytes("form4_codes.xml"),
            "form4-director.xml": _load_bytes("form4_multi_owner.xml"),
            "search-index": _load_bytes("fts_gte.json"),
        }
        return EdgarProvider(transport=_dispatch(routes))

    def test_submissions_pagina_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider(monkeypatch)
        df = provider.submissions("1111111")
        assert len(df) == 17
        assert df["filing_date"].min() == pd.Timestamp("2010-02-26")

    def test_require_history_from(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider(monkeypatch)
        provider.submissions("1111111", require_history_from=date(2011, 1, 1))
        with pytest.raises(InsufficientHistory):
            provider.submissions("1111111", require_history_from=date(2005, 1, 1))

    def test_company_facts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider(monkeypatch)
        df = provider.company_facts("1111111")
        assert df["is_restated"].any()
        assert df["cik"].eq("0001111111").all()

    def test_cik_for_normaliza(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider(monkeypatch)
        assert provider.cik_for("BRK-B") == "0001067983"
        assert provider.cik_for("brk.b") == "0001067983"
        assert provider.cik_for("NOEXISTE") is None

    def test_earnings_events_e2e(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider(monkeypatch)
        events = provider.earnings_8k("1111111")
        assert len(events) == 8
        assert all(e.ticker == "TSTA" for e in events)

    def test_form4_ventana_por_publicacion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = self._provider(monkeypatch)
        df = provider.form4("1111111", date(2024, 5, 1), date(2024, 5, 4))
        assert len(df) == 7  # 6 del officer + 1 del multi-owner
        assert df["available_at"].notna().all()
        agg = insider_net_buys(df)
        # El Form 4 del officer se aceptó a las 21:30 ET del 03 (01:30 UTC del
        # 04): la feature agrega por publicación, no por transactionDate.
        assert ("TSTA", pd.Timestamp("2024-05-04")) in agg.index
        assert agg.loc[("TSTA", pd.Timestamp("2024-05-04")), "net_buy_value"] == 24_000

    def test_form4_ventana_vacia_es_dataframe_tipado(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = self._provider(monkeypatch)
        df = provider.form4("1111111", date(2019, 1, 1), date(2019, 1, 31))
        assert len(df) == 0
        assert "code" in df.columns  # ausencia verificada, no panel roto

    def test_full_text_search_hit_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider(monkeypatch)
        df = provider.full_text_search("earnings", forms="8-K")
        assert df.attrs["hit_cap"] is True
        assert df.attrs["total"] is None  # el techo NO es un recuento
        assert df.attrs["total_lower_bound"] == 10_000
        assert df.iloc[0]["accession"] == "0001111111-24-000044"
        assert df.iloc[0]["document"] == "tsta-8k-q1-2024.htm"

    def test_frames_valida_el_periodo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider(monkeypatch)
        with pytest.raises(DataQualityError, match="periodo"):
            provider.frames("us-gaap", "Revenues", "USD", "2019Q1")

    def test_frames_marca_no_pit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEC_USER_AGENT", SEC_UA)
        payload = {
            "taxonomy": "us-gaap",
            "tag": "AccountsPayableCurrent",
            "ccp": "CY2019Q1I",
            "uom": "USD",
            "pts": 2,
            "data": [
                {"accn": "a-1", "cik": 1555538, "entityName": "X", "loc": "US-IL",
                 "end": "2019-03-31", "val": 78300000},
                {"accn": "a-2", "cik": 11199, "entityName": "Y", "loc": "US-WI",
                 "end": "2019-03-31", "val": 465700000},
            ],
        }
        provider = EdgarProvider(
            transport=ScriptedTransport(
                HttpResponse(200, content=json.dumps(payload).encode())
            )
        )
        df = provider.frames("us-gaap", "AccountsPayableCurrent", "USD", "CY2019Q1I")
        assert df.attrs["not_point_in_time"] is True
        assert len(df) == 2
        assert set(df["cik"]) == {"0001555538", "0000011199"}

    def test_archive_url_convenciones(self) -> None:
        # data.sec.gov exige CIK con ceros; Archives lo exige SIN ceros y el
        # accession sin guiones (trampa nº 1).
        url = EdgarProvider.archive_url(
            "0000320193", "0000320193-24-000006", "aapl-20231230.htm"
        )
        assert url == (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019324000006/aapl-20231230.htm"
        )


# ===========================================================================
# 9. Capa alta: EdgarFundamentals, FMP, EODHD, servicio combinado
# ===========================================================================


class TestEdgarFundamentals:
    def test_tabla_canonica_completa(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEC_USER_AGENT", SEC_UA)
        edgar = EdgarProvider(
            transport=_dispatch(
                {"companyfacts/CIK0001111111.json": _load_bytes("companyfacts_vintages.json")}
            )
        )
        fund = EdgarFundamentals(edgar=edgar)
        out = fund.quarterly_facts("TSTA", cik="0001111111")
        assert list(out.columns) == list(CANONICAL_COLUMNS)
        assert out.attrs["pit_quality"] == "vintages"
        by_concept = set(out["concept"])
        assert {"revenue", "net_income", "cfo", "total_debt", "total_assets"} <= by_concept
        # Q4 de CFO derivado del 10-K con su marca.
        q4 = out[(out["concept"] == "cfo") & (out["period_end"] == pd.Timestamp("2023-12-31"))]
        assert len(q4) == 1
        assert bool(q4.iloc[0]["is_derived"]) is True
        assert q4.iloc[0]["value"] == 17_000_000
        # La reexpresión de 2017Q1 sigue presente como vintage adicional.
        rev_q1 = out[
            (out["concept"] == "revenue")
            & (out["period_end"] == pd.Timestamp("2017-04-01"))
        ]
        assert len(rev_q1) == 2
        assert rev_q1["is_restated"].sum() == 1

    def test_cik_irresoluble(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEC_USER_AGENT", SEC_UA)
        edgar = EdgarProvider(
            transport=_dispatch(
                {"company_tickers_exchange.json": _load_bytes("company_tickers_exchange.json")}
            )
        )
        fund = EdgarFundamentals(edgar=edgar)
        with pytest.raises(ProviderUnavailable, match="deslistado"):
            fund.quarterly_facts("LEHMQ")


class TestFMPFundamentals:
    def _provider(self, monkeypatch: pytest.MonkeyPatch) -> FMPFundamentals:
        monkeypatch.setenv("FMP_API_KEY", "clave-de-prueba")
        routes = {
            "income-statement": _load_bytes("fmp_income_statement.json"),
            "balance-sheet-statement": _load_bytes("fmp_balance_sheet.json"),
            "cash-flow-statement": _load_bytes("fmp_cash_flow.json"),
        }
        return FMPFundamentals(transport=_dispatch(routes))

    def test_tabla_canonica(self, monkeypatch: pytest.MonkeyPatch) -> None:
        out = self._provider(monkeypatch).quarterly_facts("TSTA")
        assert list(out.columns) == list(CANONICAL_COLUMNS)
        assert out.attrs["pit_quality"] == "filing-date"
        rev = out[out["concept"] == "revenue"].sort_values("period_end")
        assert list(rev["value"]) == [145_000_000, 150_000_000]
        # acceptedDate "2024-08-01 16:45:12" es ET de pared -> 20:45 UTC.
        q2 = rev[rev["period_end"] == pd.Timestamp("2024-06-29")].iloc[0]
        assert q2["available_at"] == pd.Timestamp("2024-08-01 20:45:12")
        assert (out["is_restated"] == False).all()  # noqa: E712 - sin vintages

    def test_sin_clave_lanza(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        provider = FMPFundamentals(transport=_dispatch({}))
        with pytest.raises(ProviderUnavailable) as exc:
            provider.quarterly_facts("TSTA")
        assert "FMP_API_KEY" in exc.value.missing_env


class TestEODHDFundamentals:
    def test_tabla_canonica(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EODHD_API_KEY", "token-de-prueba")
        payload = {
            "General": {"Code": "TSTA"},
            "Financials": {
                "Income_Statement": {
                    "quarterly": {
                        "2024-06-30": {
                            "date": "2024-06-30",
                            "filing_date": "2024-08-01",
                            "currency_symbol": "USD",
                            "totalRevenue": "150000000.00",
                            "netIncome": "20000000.00",
                        },
                        "2024-03-31": {
                            "date": "2024-03-31",
                            "filing_date": None,  # sin fecha: se descarta
                            "totalRevenue": "145000000.00",
                        },
                    }
                },
                "Balance_Sheet": {"quarterly": {}},
                "Cash_Flow": {"quarterly": {}},
            },
        }
        provider = EODHDFundamentals(
            transport=ScriptedTransport(
                HttpResponse(200, content=json.dumps(payload).encode()),
                repeat_last=True,
            )
        )
        out = provider.quarterly_facts("TSTA")
        assert out.attrs["pit_quality"] == "restated"
        assert set(out["concept"]) == {"revenue", "net_income"}
        assert len(out) == 2  # el trimestre sin filing_date no entra
        assert (out["available_at"] == pd.Timestamp("2024-08-01 21:30:00")).all()

    def test_sin_clave_lanza(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EODHD_API_KEY", raising=False)
        provider = EODHDFundamentals(transport=_dispatch({}))
        with pytest.raises(ProviderUnavailable) as exc:
            provider.quarterly_facts("TSTA")
        assert "EODHD_API_KEY" in exc.value.missing_env


class _FakeSource:
    """Fuente falsa para probar la preferencia y el fallback del servicio."""

    kinds = ("fundamentals",)

    def __init__(self, name: str, table: pd.DataFrame | None, *, fail: bool = False):
        self.name = name
        self._table = table
        self._fail = fail
        self.calls = 0

    def available(self) -> bool:
        return True

    def quarterly_facts(self, ticker: str, *, cik: str | None = None) -> pd.DataFrame:
        self.calls += 1
        if self._fail:
            raise ProviderUnavailable(self.name, "caído a propósito")
        out = self._table.copy()
        out.attrs["pit_quality"] = "vintages" if self.name == "sec" else "restated"
        return out


def _mini_canonical(ticker: str, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "concept": ["revenue"],
            "period_end": [pd.Timestamp("2024-06-30")],
            "available_at": [pd.Timestamp("2024-08-01 21:30")],
            "value": [value],
            "fiscal_period": ["2024Q2"],
            "unit": ["USD"],
            "form": ["10-Q"],
            "accession": ["a-1"],
            "is_restated": [False],
            "concept_used": ["Revenues"],
            "is_derived": [False],
            "source": ["fake"],
        }
    )


class TestFundamentalsService:
    def test_prefiere_edgar(self) -> None:
        from earnings_alpha.data.base import ProviderRegistry

        sec = _FakeSource("sec", _mini_canonical("TSTA", 1.0))
        fmp = _FakeSource("fmp", _mini_canonical("TSTA", 2.0))
        registry = ProviderRegistry()
        registry.register("fundamentals", sec, 100)
        registry.register("fundamentals", fmp, 50)
        service = FundamentalsService(registry)
        out = service.facts("TSTA")
        assert out.iloc[0]["value"] == 1.0
        assert sec.calls == 1
        assert fmp.calls == 0

    def test_fallback_cuando_edgar_cae(self) -> None:
        from earnings_alpha.data.base import ProviderRegistry

        sec = _FakeSource("sec", None, fail=True)
        fmp = _FakeSource("fmp", _mini_canonical("TSTA", 2.0))
        registry = ProviderRegistry()
        registry.register("fundamentals", sec, 100)
        registry.register("fundamentals", fmp, 50)
        service = FundamentalsService(registry)
        out = service.facts("TSTA")
        assert out.iloc[0]["value"] == 2.0
        # El intento fallido queda auditado en la bitácora del registro.
        assert any(a.provider == "sec" and not a.ok for a in registry.attempts)

    def test_panel_registra_huecos(self) -> None:
        from earnings_alpha.data.base import ProviderRegistry

        class _Selective(_FakeSource):
            def quarterly_facts(self, ticker: str, *, cik: str | None = None):
                if ticker == "MISS":
                    raise InsufficientHistory("sin datos")
                return super().quarterly_facts(ticker, cik=cik)

        src = _Selective("sec", _mini_canonical("TSTA", 1.0))
        registry = ProviderRegistry()
        registry.register("fundamentals", src, 100)
        service = FundamentalsService(registry)
        panel = service.panel(["TSTA", "MISS"], vintage="original")
        assert panel.attrs["missing"] == ["MISS"]
        assert panel.attrs["pit_quality_by_ticker"] == {"TSTA": "vintages"}
        assert len(panel) == 1

    def test_registro_por_defecto_ordena_sec_fmp_eodhd(self) -> None:
        registry = build_default_registry()
        names = [p.name for p in registry.providers_for("fundamentals")]
        assert names == ["sec", "fmp", "eodhd"]
