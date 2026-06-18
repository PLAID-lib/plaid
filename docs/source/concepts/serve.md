# PLAID serve API

`plaid-serve` runs a small HTTP server that exposes a local PLAID dataset to
client tools and the ParaView plugin. The current server is a **read-only data
server**: it serves dataset metadata, problem definitions, and existing samples.
The server is intended for local or trusted-network use. It does not implement
authentication, authorization, or TLS.

## Start the server

Run the server with a default dataset:

```bash
uv run plaid-serve --dataset /path/to/plaid_dataset
```

By default, the server listens on `0.0.0.0:8000`. Use `--host` and `--port` to
change the bind address and port:

```bash
uv run plaid-serve \
  --dataset /path/to/plaid_dataset \
  --host 127.0.0.1 \
  --port 9000
```

If no default dataset is provided, each dataset request must include a dataset
location in the JSON body using either `dataset` or `uri`.

## Command-line options

| Option | Description |
| --- | --- |
| `--host HOST` | Bind address. Defaults to `0.0.0.0`. |
| `--port PORT` | Bind port. Defaults to `8000`. |
| `--dataset PATH` | Default local PLAID dataset directory used by dataset endpoints. |
| `--ParaViewRun` | Launch ParaView with the PLAID plugin, set `PLAID_PORT` to the selected port, and stop the server when ParaView exits. The ParaView executable is read from `PARAVIEW_EXEC`. |

## Request conventions

Dataset endpoints use `POST` with a JSON object body. The server resolves the
dataset path in this order:

1. the request `dataset` field;
2. the request `uri` field;
3. the `--dataset` value configured when the server was started.

The current implementation loads datasets from local directories with
`plaid.storage.init_from_disk`. A dataset path must exist and be a directory.
Loaded datasets are cached by path for subsequent sample requests.

## Discovery endpoints

### `GET /health` and `POST /health`

Returns a simple health payload:

```json
{"status": "ok"}
```

This route is used by `plaid.utils.process_client.PlaidClient.check_connection()`.

### `GET /entry_points`

Returns the capabilities exposed by this server:

```json
{
  "problem_definition": true,
  "process": false,
  "infos": true,
  "samples": true
}
```

`process: false` means that `plaid-serve` does not provide a processing or
prediction backend. It can still be used to retrieve dataset samples.

## Dataset endpoints

### `POST /infos`

Returns the serialized dataset `infos.yaml` metadata.

```bash
curl -X POST http://127.0.0.1:8000/infos \
  -H 'Content-Type: application/json' \
  -d '{"dataset": "/path/to/plaid_dataset"}'
```

When the server was started with `--dataset`, the request body can be empty:

```bash
curl -X POST http://127.0.0.1:8000/infos \
  -H 'Content-Type: application/json' \
  -d '{}'
```

### `POST /problem_definition`

Returns one serialized problem definition from the dataset. Use either
`problem_definition` or `problem_definition_name` to request a specific
definition.

```bash
curl -X POST http://127.0.0.1:8000/problem_definition \
  -H 'Content-Type: application/json' \
  -d '{
        "dataset": "/path/to/plaid_dataset",
        "problem_definition": "PLAID_benchmark"
      }'
```

If no name is provided, the server selects a definition deterministically:

1. `PLAID_benchmark`, when available;
2. the only definition, when the dataset contains exactly one;
3. the first definition in sorted name order.

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
| `dataset` or `uri` | Required unless `--dataset` was provided | Local PLAID dataset directory. Values are stripped of surrounding whitespace. |
| `split` | Required when the dataset has multiple splits | Split name such as `train`, `test`, or `OOD`. Values are stripped of surrounding whitespace. If the dataset has exactly one split, the split can be omitted. |
| `sample_ids` | Yes | Non-empty list of non-negative integer sample IDs. |

The response shape is:

```json
{
  "samples": [
    {"...": "serialized sample"}
  ]
}
```

The sample payloads use the same JSON representation as
`plaid.utils.sample_json.sample_to_json_payload`.

## Python client usage

`plaid.utils.process_client.PlaidClient` can query the read-only endpoints when
the server was started with `--dataset`:

```python
from plaid.utils.process_client import PlaidClient

client = PlaidClient(host="localhost", port=8000)

if client.check_connection():
    infos = client.infos()
    problem_definition = client.problem_definition()
    sample = client.samples(
        sample_ids=[0],
        split=problem_definition["training_split"][0],
    )[0]
```

The same client also has a `process(sample)` method for servers that implement
`POST /process`, but `plaid-serve` itself intentionally does not implement that
operation.

## ParaView usage

`plaid-serve --ParaViewRun` starts ParaView with the PLAID plugin and keeps the
HTTP server alive until ParaView exits. The plugin reads the connection port
from `PLAID_PORT` and can retrieve `/infos`, `/problem_definition`, and
`/samples` from the server.

The plugin also exposes a "Process" toggle for servers that implement
`/process`. Leave this toggle disabled when using the built-in `plaid-serve`
data server.

## Unsupported processing endpoint

`POST /process` is intentionally unsupported by `plaid-serve` and returns HTTP
501:

```json
{"error": "Endpoint /process is not supported by PLAID serve"}
```

## Error responses

Validation errors return HTTP 400 with an `error` message. Typical validation
errors include missing dataset paths, invalid JSON bodies, missing or invalid
`sample_ids`, unknown splits, and non-string problem-definition names.

Unknown `GET` routes return:

```json
{"GET error": "Not Found"}
```

Unknown `POST` routes return:

```json
{"POST error": "Not Found"}
```

Unexpected server errors return HTTP 500 with an `error` message and are logged
by the server.
