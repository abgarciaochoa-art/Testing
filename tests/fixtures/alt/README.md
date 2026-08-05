# Fixtures de `tests/test_data_alt.py`

Respuestas grabadas **a mano contra la forma documentada** de cada API (no
contra la API viva: el proxy de este entorno bloquea todos los dominios
financieros; véase `docs/ARCHITECTURE.md` §5). Los endpoints y nombres de campo
proceden de `docs/research/data_sources.md` (§5.3 calendario, §6.2 consenso,
§7.2 opciones, §8.2 short interest, §9.4 ATS) y las cifras del episodio son
plausibles pero sintéticas.

Escenario común: resultados de AAPL del trimestre fiscal terminado el
2020-06-27, anunciados AMC el 2020-08-27 — el mismo episodio (split 4:1 el
2020-08-31) que usan los fixtures de `tests/fixtures/prices/`.

| Fichero | API que imita |
|---|---|
| `fmp_earnings_calendar.json` | FMP `/stable/earnings-calendar` |
| `fmp_earnings_aapl.json` | FMP `/stable/earnings?symbol=AAPL` |
| `fmp_analyst_estimates_aapl.json` | FMP `/stable/analyst-estimates` |
| `finnhub_earnings_calendar.json` | Finnhub `/calendar/earnings` |
| `finnhub_eps_estimate.json` | Finnhub `/stock/eps-estimate` |
| `finnhub_revenue_estimate.json` | Finnhub `/stock/revenue-estimate` |
| `eodhd_earnings_calendar.json` | EODHD `/api/calendar/earnings` |
| `nasdaq_calendar_day.json` | `api.nasdaq.com/api/calendar/earnings?date=` |
| `polygon_options_snapshot_p1.json` / `_p2.json` | Polygon `/v3/snapshot/options/{u}` (paginado) |
| `polygon_contracts_asof.json` | Polygon `/v3/reference/options/contracts?as_of=` |
| `polygon_underlying_agg.json` | Polygon `/v2/aggs/ticker/AAPL/range/1/day/...` |
| `orats_strikes.json` | ORATS `/datav2/hist/strikes` |
| `tradier_expirations.json` / `tradier_chain_*.json` | Tradier `/v1/markets/options/*` |
| `finra_token.json` | FINRA EWS OAuth2 (client credentials) |
| `finra_short_interest.json` | FINRA Query API `consolidatedShortInterest` |
| `finra_weekly_summary.json` | FINRA Query API `weeklySummary` |
| `regsho_cnms_20200902.txt` | `cdn.finra.org/equity/regsho/daily/CNMSshvol20200902.txt` |
| `regsho_malformed.txt` | fichero Reg SHO con líneas corruptas |
