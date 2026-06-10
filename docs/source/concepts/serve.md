# PLAID serve API

`plaid-serve` runs a small HTTP server that exposes a local PLAID dataset to
client tools and the ParaView plugin. It is intended for local or trusted
network use; it does not implement authentication.

## Start the server

Run the server with a default dataset:

```bash
uv run plaid-serve --dataset /path/to/plaid_dataset
```

By default, the server listens on `0.0.0.0:8000`. You can change the bind
address and port:

```bash
uv run plaid-serve \
  --dataset /path/to/plaid_dataset \
  --host 127.0.0.1 \
  --port 9000
```

If no default dataset is provided, each dataset route must include a `dataset`
or `uri` field in the JSON request body.

## Command-line options

| Option | Description |
| --- | --- |
| `--host HOST` | Bind address. Defaults to `0.0.0.0`. |
| `--port PORT` | Bind port. Defaults to `8000`. |
| `--dataset PATH` | Default PLAID dataset path used by dataset endpoints. |
| `--ParaViewRun` | Launch ParaView with the PLAID plugin and stop the server when ParaView exits. The ParaView executable is read from `PARAVIEW_EXEC`. |

## Discovery endpoints

### `GET /health` and `POST /health`

Returns a simple health payload:

```json
{"status": "ok"}
```

### `GET /entry_points`

Returns the capabilities exposed by this server:

```json
{
  "samples_step": true,
  "predict": false,
  "splits": true,
  "timesteps": true,
  "samples": true
}
```

## Dataset endpoints

All dataset endpoints use `POST` with a JSON object request body. The dataset
location can be omitted when the server was started with `--dataset`.

### `POST /infos`

Returns the serialized `infos.yaml` metadata for a dataset.

```bash
curl -X POST http://127.0.0.1:8000/infos \
  -H 'Content-Type: application/json' \
  -d '{"dataset": "/path/to/plaid_dataset"}'
```

### `POST /problem_definition`

Returns a serialized problem definition. Use `problem_definition` or
`problem_definition_name` to request a specific definition.

```bash
curl -X POST http://127.0.0.1:8000/problem_definition \
  -H 'Content-Type: application/json' \
  -d '{
        "dataset": "/path/to/plaid_dataset",
        "problem_definition": "PLAID_benchmark"
      }'
```

If no name is provided, the server returns `PLAID_benchmark` when available,
the only definition when the dataset has one, or the first definition in sorted
name order.

### `POST /samples`

Returns serialized PLAID samples.

```bash
curl -X POST http://127.0.0.1:8000/samples \
  -H 'Content-Type: application/json' \
  -d '{
        "dataset": "/path/to/plaid_dataset",
        "split": "train",
        "sample_ids": [0, 1]
      }'
```

Request fields:

| Field | Required | Description |
| --- | --- | --- |
| `dataset` or `uri` | Required unless `--dataset` was provided | Local PLAID dataset path. |
| `split` | Required when the dataset has multiple splits | Split name such as `train` or `test`. |
| `sample_ids` | Yes | Non-empty list of non-negative sample IDs. |

The response shape is:

```json
{"samples": [{"...": "serialized sample"}]}
```

## Unsupported prediction endpoint

`POST /predict` is intentionally unsupported by `plaid-serve` and returns
HTTP 501:

```json
{"error": "Endpoint /predict is not supported by PLAID serve"}
```

## Error responses

Validation errors return HTTP 400 with an `error` message. Unknown routes return
HTTP 404. Unexpected server errors return HTTP 500 with an `error` message and
are logged by the server.