# ASP Demand Prediction API

ロカール・LightGBMベースのアクセス数とAPIの使われ方（誰が、どのくらいの量を使うか）をシステム的に自動予測
AWS **Athena → SageMaker** パイプラインのコスト節約の目的

**Data flow** (how a forecast is computed):

```
ASPログファイル (S3 or local) → 前処理 (parse + aggregate) → 予測変数データ (カレンダー)
    → LightGBM トレーニング → 実測値との比較・予測
```

**システム**（ポートとアダプタ）— 1つの**コアエンジン**（基本機能）が2つの
**フロントエンド**によって駆動される。JS UIはHTTP経由でFastAPI（薄いラッパー）と通信し、
CLIはプロセス内でエンジンを直接呼び出す。
FastAPIは**JS UIのバックエンド**でもあり、Curlで処理を受けることもできる。

```
Frontends      JS UI (/ui)            CLI (uv run asp-demand …)
                   │ HTTP              │ in-process
Wrapper        FastAPI (:8000)         │   ← thin HTTP adapter
                   └──────────┬────────┘
                              ▼
Core engine    asp_demand —  calendar · preprocess · train · predict · backtest · visualize
                              │   (generate_calendar / aggregate.run / train_model /
                              ▼    forecast / backtest / plot_forecast)
Storage        data/runs/<ts>/ (artifacts)  ·  S3 / local logs  ·  data/cache
```

目標：ELBごとのリクエスト数を、**1時間ごと、3時間ごと、6時間ごと、12時間ごと、1日ごと**の粒度で計測する。
予測値は**分位数**（P50/P90/P95）で、
学習済みの実行結果は
1. **実績値との比較** backtest
2. **未来予測** forecast

## セットアップ

1. `.env.example`を `.env`としてコピーし、設定に合わせて変更:

   ```bash
   ASP_RAW_LOG_URI=s3://your-bucket/aplb-access-log
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_DEFAULT_REGION=ap-northeast-1            # must match the bucket's region
   AWS_ENDPOINT_URL=https://s3.ap-northeast-1.amazonaws.com   # optional / on-prem S3-compatible
   ```

2. Dev Container内部で

```bash
uv sync                       # install runtime + dev deps
```

## GUIクイックスタート

```bash
uv run uvicorn asp_demand.api.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/ui/   (option picker)   ·   /docs (Swagger)
```

## Dev Container CLIクイックスタート

**基本手順**
```bash
# 1. 確定済みの日本カレンダー（祝日／曜日フラグ）を生成 (2020年から2050年まで)
uv run python scripts/gen_japan_calendar.py 2020 2050

# 2. ログを時間別および日別の集計値に集計する（UTCフォルダ範囲）。
uv run asp-demand preprocess --start 2025-05-06 --end 2025-05-06

# 3. モデル　トレーニング
uv run asp-demand train --granularity hourly
uv run asp-demand train --granularity daily

# 4. CLIから予測
uv run asp-demand predict --granularity hourly --horizon 24
```

**選択的コマンド**

```bash
uv run asp-demand preprocess --start 2025-01-01 --end 2025-03-31   # 前処理
uv run asp-demand train   --granularity hourly                     # トレーニング単位 hourly | 3h | 6h | 12h | daily
uv run asp-demand predict --granularity 6h --horizon 4             # 6時間分予測 ⇒ forecast_6h.csv
# 指定された日付の期間を予測:
uv run asp-demand predict --granularity hourly --from 2025-01-01 --to 2025-01-03
# バックテスト：過去のデータに基づいて予測を行い、実際の結果と比較する（MAE/MAPE）
uv run asp-demand backtest --granularity hourly --from 2025-05-06 --to 2025-05-06
# 視覚化：Plotlyチャート（実績値＋予測値 または、予測値のみ） -> <run>/forecast_hourly.html
uv run asp-demand plot --granularity hourly
uv run asp-demand plot --granularity daily --height 1200   # even taller
uv run asp-demand plot --granularity hourly --kind backtest          # 実績値 vs. 予測値
# 特定のディレクトリを対象とする
uv run asp-demand train   --granularity hourly --run-dir data/runs/260619131540
```

## 出力レイアウト (data)

```
data/
  data/calendar/japan_calendar.csv  # Japan calander data
  cache/<utc-date>.parquet          # shared per-day aggregation cache (reused across runs)
  runs/
    latest -> 260619131540          # relative symlink to the newest run
    260619131540/                   # <yymmddhhmmss>
      hourly.parquet  3h.parquet  6h.parquet  12h.parquet  daily.parquet   # preprocess (all granularities)
      hourly.txt      hourly_features.json  metrics_hourly.json   # train output (per granularity)
      forecast_hourly.csv                 # predict output (written by default)
```

1. /cache
一度前処理されたデータはキャッシュに書き込まれ、**迅速な前処理プロセスのために再前処理を行わない**ようにする。
前処理をやり直したい場合は、キャッシュを消去すること(GUI: refresh cacheをチェック, CLI: キャッシュをクリアするコマンドを実行「前処理のパフォーマンスとキャッシュ」参照 )

2. /calendar
確定済みの日本カレンダー（祝日／曜日フラグ）のデータを生成し、学習情報として活用
平日、、、、、、連休、年末
| フラグ名 | 日の名 |
|---|---|
| is_weekend | 週末 |
| is_holiday | 祝日 |
| is_business_day | 営業日 |
| day_before_holiday | 祝日前日 |
| day_after_holiday | 祝日翌日 |
| in_long_weekend | 週末連休 |
| is_nenmatsu | 年末年始 |

3. /runs
すべての`preprocess`はタイムスタンプ付きの**実行ディレクトリ**を作成し、`latest`をそのディレクトリに向けます
`train`と`predict`は実行ディレクトリ（デフォルト：`latest`）上で動作します。実行は自己完結型です。


## タイムゾーンと曜日の境界

需要は設定可能なタイムゾーンごとに分類されます。`start`/`end`は**そのタイムゾーンにおける**日付です。:

```bash
uv run asp-demand preprocess --start 2025-01-05 --end 2025-01-15 --tz Asia/Tokyo  # default (JST)
uv run asp-demand preprocess --start 2025-01-05 --end 2025-01-15 --tz UTC         # matches Athena
```

## 前処理のパフォーマンスとキャッシュ

ログの読み込みがボトルネックとなっている（gzip圧縮と解析処理はCPU負荷が高く、S3はネットワークI/Oも加えます）。
そのため、前処理では**プロセスプール**（スレッドではなく真の並列処理。GILにより解析処理は直列化されます）を使用してファイルを読み込み、**各UTCフォルダの日付を**　`data/cache/<tz>/<date>.parquet`にキャッシュする（実行間で共有されます）。これにより、同じ日付のデータが二度集計されることはない。

```bash
# ワーカー数を調整（デフォルト：min(32, cores)）。S3（I/Oバウンド）の場合は、ワーカー数を増やしも良い。
uv run asp-demand preprocess --start 2025-01-01 --end 2025-12-31 --workers 64

# 強制的に再読み込みを行い、日ごとのキャッシュは無視する
uv run asp-demand preprocess --start 2025-05-06 --end 2025-05-06 --refresh

# キャッシュをクリアする（すべてのタイムゾーン/特定のタイムゾーン/レガシー孤立ファイル）
uv run asp-demand cache-clean --dry-run
uv run asp-demand cache-clean --tz UTC
uv run asp-demand cache-clean --orphaned
```

バンドルされた日（576ファイル）での計測結果：**シリアル処理125秒 → 約6秒**（ワーカー数32）。
キャッシュされた再実行は**約0.5秒**。
シリアル処理で約63時間かかる1年分の処理が数時間に短縮され、後続の実行では新しく追加された日のみが読み込まれる。

## FastAPIコントロールプレーン＋UI

すべてのタスクはHTTP経由で実行可能で、プレーンなJavaScriptで作成されたオプションピッカーが同一オリジンから提供される（ビルドステップなし、CORSなし）。時間のかかるタスクは、ポーリングによって監視されるバックグラウンド**ジョブ**として実行される。

```bash
uv run uvicorn asp_demand.api.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/ui/   (option picker)   ·   /docs (Swagger)
```

| エンドポイント | タスク | モード |
|---|---|---|
| `POST /calendar` | 確定済みの日本カレンダー（祝日／曜日フラグ）を生成 | sync |
| `POST /preprocess` | ログ前処理・新しいディレクトリ | **job** → `job_id` |
| `POST /train` | トレーニング | **job** → `job_id` |
| `POST /predict` | 予測P50/P90/P95 | sync |
| `POST /backtest` | 過去の期間における予測値 + MAE/MAPEと実績値の比較 | sync |
| `GET /visualize` | 視覚化：Plotlyチャート.html 生成（実績値＋予測値 または、予測値のみ） (`kind=forecast\|backtest`) | sync |
| `GET /runs` · `GET /jobs/{id}` | リスト実行 · ジョブのポーリング | sync |

```bash
# async job: kick off + poll
curl -X POST localhost:8000/preprocess -H 'content-type: application/json' \
  -d '{"start":"2025-05-06","end":"2025-05-06","tz":"UTC"}'      # -> {"job_id": "..."}
curl localhost:8000/jobs/<job_id>                                 # status + result
# sync predict
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"granularity":"hourly","from_date":"2025-05-07","to_date":"2025-05-07"}'
```

## Docker / コンテナレジストリ

A multi-stage `Dockerfile` builds a slim CPU image (uv + locked deps, non-root user)
that serves the API by default. Build and push via the `Makefile`:

```bash
make build                                    # local image asp-demand:latest
make build TAG=v0.1.0

# push to a registry (ECR example)
make ecr-login REGISTRY=<acct>.dkr.ecr.ap-northeast-1.amazonaws.com AWS_REGION=ap-northeast-1
make release  REGISTRY=<acct>.dkr.ecr.ap-northeast-1.amazonaws.com/asp-demand TAG=v0.1.0
```

`docker compose` serves the API + UI:

```bash
docker compose up api      # serve API + /ui on :8000 (mount ./data; runs persist on the host)
```

`data/` is mounted as a volume, so artifacts persist on the host and the image stays small
(raw logs are never baked in — see `.dockerignore`).

## Unitテスト & 品質管理

```bash
uv run pytest
uv run ruff check .
uv run mypy asp_demand
```

## レイアウト

| Path | Role |
|------|------|
| `asp_demand/preprocessing/` | parse `.gz` ALB logs, aggregate to hourly/daily (Athena replacement) |
| `asp_demand/features/` | Japan calendar join + time/lag/rolling features |
| `asp_demand/model/` | LightGBM quantile train + forecast/backtest (SageMaker replacement) |
| `asp_demand/viz.py` | interactive Plotly charts (forecast / backtest) |
| `asp_demand/api/` | FastAPI control plane (`main.py`), job manager (`jobs.py`), vanilla-JS UI (`static/`) |
| `asp_demand/cli.py` | `asp-demand preprocess\|train\|predict\|backtest\|plot\|cache-clean` |
| `scripts/gen_japan_calendar.py` | build the committed calendar CSV |
| `Dockerfile`, `docker-compose.yml`, `Makefile` | container build + registry push |
