# Experimental FastAPI Probe

BO Forge includes an experimental optional FastAPI probe around the internal
`CampaignAppService`. It is for local or trusted-network API exploration only.
It is not a stable public API and does not replace the Streamlit workbench.
Do not expose it directly to the public internet.

For the supported and deferred workflow combinations around the API probe, see
[CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md).

The API has no built-in auth, no database, no multi-user state coordination,
and no persistent staged server state. Do not expose it directly to the public
internet.

## Install

```bash
pip install "bo-forge[api]"
```

From a development checkout:

```bash
./.venv/bin/pip install -e ".[dev]"
```

## Launch

Local-only:

```bash
bo-forge-api --root . --host 127.0.0.1 --port 8765
```

Trusted LAN or lab server:

```bash
bo-forge-api --root . --host 0.0.0.0 --port 8765
```

Wildcard or non-loopback hosts expose the probe to the network. Use only on a
trusted LAN, VPN, or SSH tunnel. Anyone who can reach the API can read and write
campaign files under the configured root directory through the exposed campaign
operations.

SSH tunnel:

```bash
bo-forge-api --root . --host 127.0.0.1 --port 8765
ssh -L 8765:127.0.0.1:8765 user@server
```

Then open `http://127.0.0.1:8765/docs` on the client machine.

## Root-Bound Paths

The probe uses a root-bound path model.

The launcher requires an existing root directory:

```bash
bo-forge-api --root /path/to/campaign-workspace
```

Requests must use relative `config_path` and `log_path` values. Absolute paths
and paths that escape the root after symlink resolution are rejected with
structured JSON errors and do not touch files.

Example campaign reference:

```json
{
  "config_path": "configs/01_simple_2d_maximise_logei.yaml",
  "log_path": "examples/01_simple_2d_maximise_logei_campaign_log.csv"
}
```

## Endpoints

Health:

```bash
curl http://127.0.0.1:8765/health
```

Validate a campaign:

```bash
curl -X POST http://127.0.0.1:8765/campaign/validation \
  -H "Content-Type: application/json" \
  -d '{"config_path":"configs/01_simple_2d_maximise_logei.yaml","log_path":"examples/01_simple_2d_maximise_logei_campaign_log.csv"}'
```

Summarise a campaign:

```bash
curl -X POST http://127.0.0.1:8765/campaign/summary \
  -H "Content-Type: application/json" \
  -d '{"config_path":"configs/01_simple_2d_maximise_logei.yaml","log_path":"examples/01_simple_2d_maximise_logei_campaign_log.csv"}'
```

Generate dry-run suggestions without mutating the CSV:

```bash
curl -X POST http://127.0.0.1:8765/campaign/suggestions/dry-run \
  -H "Content-Type: application/json" \
  -d '{"config_path":"configs/01_simple_2d_maximise_logei.yaml","log_path":"examples/01_simple_2d_maximise_logei_campaign_log.csv","batch_size":1}'
```

Contextual campaigns pass context values in the same dry-run request:

```json
{
  "config_path": "configs/16_contextual_logei.yaml",
  "log_path": "examples/16_contextual_logei_campaign_log.csv",
  "batch_size": 1,
  "context_values": {
    "feedstock_acidity": 0.25
  }
}
```

The dry-run response includes a stateless staged bundle:

```json
{
  "staged_bundle": {
    "...": "exact dry-run staged_bundle"
  }
}
```

Send that exact bundle to `/campaign/suggestions/append` to append through the
existing `CampaignAppService.append_staged()` path:

```json
{
  "config_path": "configs/01_simple_2d_maximise_logei.yaml",
  "log_path": "examples/01_simple_2d_maximise_logei_campaign_log.csv",
  "staged_bundle": {
    "...": "exact dry-run staged_bundle"
  }
}
```

Clients should not edit the staged bundle. Append rechecks the staged bundle's
embedded `config_fingerprint` and `log_fingerprint`; append does not use a
separate `expected_log_fingerprint`. Contextual dry-runs also record
the supplied `context_values` in the staged bundle so trusted clients can retain
the context used to generate the suggestions.

Staged bundles are fingerprint and integrity checked. They are not
authenticated, signed, or server-issued. A trusted client can craft a
schema-valid staged bundle, so this probe must only be used with trusted clients
on localhost or trusted networks. Signed staged bundles and server-side staged
state are out of scope for this experimental probe.

Review and observation mutations require `expected_log_fingerprint`. If the log
changed since the caller last read it, or if the fingerprint is missing, the
mutation fails without writing. Use the `log_fingerprint` returned by
validation, summary, dry-run, append, review, or observation responses before
the next mutation.

Multi-objective observation requests use coupled objective values:

```json
{
  "config_path": "configs/10_multi_objective_mixed_constrained_qlogehvi.yaml",
  "log_path": "examples/10_multi_objective_mixed_constrained_campaign_log.csv",
  "row_id": "suggested-row-id",
  "expected_log_fingerprint": "current-log-fingerprint",
  "objective_values": {
    "yield_score": 70.0,
    "waste_score": 15.0
  }
}
```

Partial multi-objective values fail through the existing backend validation and
leave the CSV unchanged.

## Table Payloads

DataFrames are serialized as:

```json
{
  "columns": ["row_id"],
  "records": [{"row_id": "x"}]
}
```

Missing pandas values and NaN values become `null`; other non-JSON-native values
are converted to strings.

## Scope

The API probe does not add BO behavior, schemas, auth, CORS broadening, report
or plot endpoints, a database, remote Streamlit mode, or production deployment
infrastructure.

Streamlit remains the recommended local UI.
