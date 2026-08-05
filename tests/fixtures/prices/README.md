# Fixtures de precios (`tests/fixtures/prices/`)

Fixtures **construidos a mano contra la forma documentada** de cada API de
precios (el contenedor de desarrollo no alcanza las APIs financieras, contrato
`docs/ARCHITECTURE.md` §5, así que no son grabaciones de tráfico real). La forma
de cada respuesta está verificada contra la documentación pública del proveedor
(2026-08) y citada en el docstring del adaptador correspondiente en
`earnings_alpha/data/prices.py`.

Todos los ficheros describen el MISMO episodio de mercado sintético, calcado del
split de AAPL de agosto de 2020: ocho sesiones NYSE (2020-08-24 → 2020-09-02),
un **split 4:1 efectivo el 2020-08-31** y un **dividendo de 0.205 USD con fecha
ex 2020-09-02**. Como todos los proveedores describen el mismo suceso en su
propio formato (Yahoo retro-ajustado, Polygon crudo + eventos v3, Tiingo con
`divCash`/`splitFactor` diarios, Stooq ya ajustado, Alpaca raw/all), los tests
pueden exigir que los seis adaptadores produzcan **exactamente el mismo panel**.

`ground_truth.json` contiene la serie as-traded y los factores de retorno total
CRSP `(close_t*split_t + div_t)/close_{t-1}` calculados una sola vez; es la
referencia contra la que se comprueban los adaptadores.

Regenerables con `python3 tests/fixtures/prices/_make_fixtures.py` (determinista, sin red).

| Fichero | API imitada |
|---|---|
| `yfinance_aapl_chart.json` | Yahoo `v8/finance/chart` (precios en base actual + events) |
| `yfinance_not_found.json` | Yahoo `chart.error` para símbolo inexistente (HTTP 404) |
| `polygon_aggs_aapl_page{1,2}.json` | Polygon `/v2/aggs` `adjusted=false`, paginado con `next_url` |
| `polygon_{dividends,splits}_aapl.json` | Polygon `/v3/reference/*` |
| `polygon_aggs_empty.json` | Polygon aggs sin resultados |
| `tiingo_aapl.json` | Tiingo `/tiingo/daily/{t}/prices` (as-traded + divCash/splitFactor) |
| `tiingo_not_found.json` | Tiingo detail «Not found» |
| `stooq_aapl.csv` | Stooq `q/d/l` CSV ya ajustado |
| `stooq_no_data.txt` | Stooq sin datos |
| `alpaca_bars_raw_page{1,2}.json` | Alpaca `/v2/stocks/bars` `adjustment=raw`, paginado |
| `alpaca_bars_all.json` | Alpaca `adjustment=all` |
| `ground_truth.json` | serie as-traded y factores CRSP de referencia |
