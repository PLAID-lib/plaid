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

When `--ParaViewRun` is used, the launcher passes a generated script to
ParaView to create and display a **Plaid Explorer** source. The script is kept
available until the launcher exits.

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

PLAID normally serializes NumPy values with explicit dtype and shape metadata.
For interoperability with other JSON producers, deserialization also accepts a
plain JSON string for CGNS nodes whose declared datatype includes `C1`, such as
`ZoneType_t` and `GridLocation_t`. These values are restored as NumPy `|S1`
character arrays, as required by pyCGNS and mesh readers such as Muscat. Scalar
strings on non-character nodes remain Python strings.

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

The same client also has a `process(...)` method for servers that implement
`POST /process`, but `plaid-serve` itself intentionally does not implement that
operation.

### `process`

For servers that implement `POST /process`, `PlaidClient.process` works on a
single sample at a time and returns a single `Sample`. The client is agnostic
to the server contract: it sends the optional inline `sample` plus any extra
keyword fields verbatim, and decodes the single sample returned by the server.

```python
# Send one sample and get one sample back.
result = client.process(sample)

# Forward extra fields (for example an operation selector defined by the
# server). The client does not interpret these field names.
result = client.process(operation="some_server_operation", sample=sample)
```

Any `Sample` value passed as a keyword field is JSON-encoded automatically, so
a server that expects additional sample-valued fields can be driven without
manual serialization:

```python
result = client.process(
    operation="some_server_operation",
    sample=sample,
    source_sample=other_sample,
)
```

## ParaView usage

The PLAID ParaView plugin converts directly between VTK datasets and CGNS tree
nodes, without writing an intermediate mesh file. Conversion preserves geometry,
cell connectivity, point and cell fields, supported mesh tags, and CGNS base and
zone names when the data is round-tripped through VTK field-data metadata. VTK
character arrays containing only `0` and `1` are interpreted as mesh-tag masks;
other arrays are treated as fields.

Uniform fixed-size unstructured VTK grids use bulk connectivity conversion for
better performance. Mixed cell types and layouts that do not meet the fast-path
requirements use per-cell conversion instead. Supported CGNS cell types are
converted directly; unsupported VTK cell types raise an error rather than being
silently dropped.

CGNS tree JSON serialization preserves NumPy dtype and shape information. For
interoperability with external JSON producers, character-valued CGNS nodes such
as `ZoneType_t` and `GridLocation_t` may also be supplied as plain JSON strings;
they are restored as one-byte character arrays.

`plaid-serve --ParaViewRun` starts ParaView with the PLAID plugin and keeps the
HTTP server alive until ParaView exits. The plugin reads the connection port
from `PLAID_PORT` and can retrieve `/infos`, `/problem_definition`, and
`/samples` from the server.

The launcher bundles the plugin and its CGNS JSON, JSON codec, and VTK conversion
helpers into a single `PlaidParaViewPlugin.py` file, so ParaView can load the
plugin without installing PLAID in its Python environment. To create that
self-contained file in a specific directory instead of a temporary directory,
use the helper from Python:

```python
from pathlib import Path

from plaid.cli.paraview_plugin import get_ParaView_plugin_path_one_file

plugin_directory = get_ParaView_plugin_path_one_file(Path("/tmp/plaid-plugin"))
print(Path(plugin_directory) / "PlaidParaViewPlugin.py")
```

The provided directory must already exist. The helper returns its path as a
string; without a `path` argument, it creates a temporary directory.

The plugin also exposes a "Process" toggle for servers that implement
`/process`. Leave this toggle disabled when using the built-in `plaid-serve`
data server.

### Opening individual Samples

The plugin can read individual Samples directly from disk without starting
`plaid-serve`:

| Reader | Input | PLAID required in ParaView's Python environment |
| --- | --- | --- |
| **Plaid Sample JSON Reader** (`PlaidSampleJSONReader`) | A `.json` file produced by `sample_to_json` | No, when using the self-contained plugin |
| **Plaid Sample Directory Reader** (`PlaidSampleReader`) | A directory written by `Sample.save_to_dir` | Yes |

Export an existing Sample from your PLAID Python environment:

```python
from pathlib import Path

from plaid import Sample
from plaid.utils.sample_json import sample_to_json

sample = Sample.load_from_dir("sample_000000000")
Path("sample.json").write_text(sample_to_json(sample), encoding="utf-8")

# Alternatively, save an in-memory Sample as a directory.
sample.save_to_dir("sample_for_paraview")
```

The JSON reader expects the versioned `plaid-sample-json` format (version 1),
including its timestamped CGNS trees; arbitrary JSON files are not supported.

Generate the self-contained plugin using
`get_ParaView_plugin_path_one_file` as shown above. In ParaView, load the generated
`PlaidParaViewPlugin.py` through **Tools → Manage Plugins → Load New**. Then use
**File → Open** to select `sample.json` or the Sample directory, choose the
corresponding PLAID reader if prompted, and click **Apply**. The directory reader
loads through `Sample.load_from_dir`; it and the existing dataset reader are
registered only when the optional PLAID imports succeed. Missing PLAID does not
disable the JSON reader or the rest of the self-contained plugin.

After loading the plugin, a ParaView Python script can also create either reader:

```python
from paraview.simple import PlaidSampleJSONReader

reader = PlaidSampleJSONReader(FileName="/absolute/path/to/sample.json")
reader.UpdatePipeline()

# Requires PLAID in ParaView's Python environment:
# from paraview.simple import PlaidSampleReader
# reader = PlaidSampleReader(FileName="/absolute/path/to/sample_for_paraview")
```

Both readers expose timestamps in ascending order for ParaView's animation
controls. A requested time selects the nearest available timestamp; without an
explicit request, the earliest timestamp is used. Structured, unstructured, and
multiblock mesh outputs retain their corresponding VTK types. Changing the input
path clears the cached Sample. In client/server sessions, input paths must be
accessible to the ParaView server, and the directory reader requires PLAID in
the server's Python environment.

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
