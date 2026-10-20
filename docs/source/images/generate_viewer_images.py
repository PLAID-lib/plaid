"""Regenerate the viewer documentation images and the embedded skrub report.

The images are committed and refreshed by hand after a UI change::

    uv run python docs/source/images/generate_viewer_images.py

The skrub report embedded in ``concepts/viewer.md`` is regenerated at each
documentation build (``docs/generate_doc.sh``) with::

    python docs/source/images/generate_viewer_images.py --report-only

Both use the same small synthetic dataset, so the documentation never
depends on a local or downloaded dataset. Screenshots drive the viewer
through its trame state (not by clicking in the page, whose markup changes
with Vuetify) and capture it with a headless Firefox through WebDriver BiDi,
which needs neither Playwright nor geckodriver.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import numpy as np

IMAGES_DIR = Path(__file__).resolve().parent
REPORT_PATH = IMAGES_DIR.parent / "concepts" / "viewer_skrub_report.html"
DATASET_ID = "Plate"
TIME_STEP = "last"
REPORT_GLOBALS = ["Global/Efficiency", "Global/Massflow"]
# Efficiency is missing in the test split: plot Globals both splits share.
SCATTER_GLOBALS = ["Global/Compression_ratio", "Global/Massflow", "Global/Omega"]
SPLITS = {"train": 120, "test": 30}
VIEWPORT = {"width": 1330, "height": 1210}


def _sample(index: int):
    """Build one synthetic sample: a bent plate with a pressure field."""
    import Muscat.MeshTools.MeshCreationTools as MCT  # noqa: PLC0415
    from Muscat.Bridges.CGNSBridge import MeshToCGNS  # noqa: PLC0415

    from plaid import Sample  # noqa: PLC0415

    rng = np.random.default_rng(index)
    omega = rng.uniform(1500.0, 1700.0)
    ratio = 1.6 + omega / 5000 + rng.normal(0.0, 0.02)
    massflow = 6.0 + omega / 130 - 2.0 * (ratio - 1.9) + rng.normal(0.0, 0.2)
    efficiency = 0.87 + 0.04 * (ratio - 1.9) - 0.004 * (massflow - 19.5)
    efficiency += rng.normal(0.0, 0.003)

    mesh = MCT.CreateSquare(
        dimensions=[60, 60], origin=[0, 0], spacing=[1 / 59, 1 / 59]
    )
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    mesh.nodes = np.column_stack([x, y, 0.15 * np.sin(np.pi * x) * (1 + y)])
    mesh.nodeFields["Pressure"] = 1.6e5 + 7e4 * ratio * x * (1 + 0.3 * np.sin(4 * y))

    sample = Sample()
    sample.add_tree(MeshToCGNS(mesh))
    sample.add_global("Compression_ratio", ratio)
    sample.add_global("Efficiency", efficiency)
    sample.add_global("Massflow", massflow)
    sample.add_global("Omega", omega)
    return sample


def _test_sample(index: int):
    # Test samples have no Efficiency, as in benchmarks hiding test outputs.
    sample = _sample(1000 + index)
    sample.del_global("Efficiency")
    return sample


def _constructor(key: tuple[str, int]):
    split, index = key
    return _test_sample(index) if split == "test" else _sample(index)


def write_dataset(root: Path) -> Path:
    """Write the synthetic dataset under ``root`` and return its directory."""
    from plaid.storage import save_to_disk  # noqa: PLC0415

    path = root / DATASET_ID
    save_to_disk(
        output_folder=path,
        sample_constructor=_constructor,
        ids={split: [(split, i) for i in range(n)] for split, n in SPLITS.items()},
        backend="zarr",
        overwrite=True,
    )
    return path


def write_report(root: Path, path: Path) -> None:
    """Write the skrub report the viewer builds for ``REPORT_GLOBALS``."""
    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.global_plots import (  # noqa: PLC0415
        globals_dataframe,
        prepare_globals,
    )
    from plaid.viewer.services import PlaidDatasetService  # noqa: PLC0415
    from plaid.viewer.trame_app.server import (  # noqa: PLC0415
        _report_columns,
        _write_globals_report,
    )

    service = PlaidDatasetService(ViewerConfig(datasets_root=root))
    prepared = prepare_globals(service.extract_globals(DATASET_ID, time_step=TIME_STEP))
    columns = _report_columns(REPORT_GLOBALS, prepared.text_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same title as a report generated from the viewer.
    title = f"{DATASET_ID} — {TIME_STEP} timestep"
    _write_globals_report(globals_dataframe(prepared, columns=columns), path, title)


class _Firefox:
    """Minimal WebDriver BiDi client for headless Firefox screenshots."""

    def __init__(self, ws) -> None:
        self._ws = ws
        self._id = 0
        self.context = ""

    async def call(self, method: str, **params):
        self._id += 1
        await self._ws.send_json({"id": self._id, "method": method, "params": params})
        async for message in self._ws:
            data = message.json()
            if data.get("id") == self._id:
                if data["type"] == "error":
                    raise RuntimeError(f"{method}: {data['message']}")
                return data["result"]
        raise RuntimeError("Firefox closed the WebDriver BiDi connection.")

    async def evaluate(self, expression: str):
        result = await self.call(
            "script.evaluate",
            expression=expression,
            target={"context": self.context},
            awaitPromise=True,
        )
        return result["result"].get("value")

    async def wait_for(self, expression: str, timeout: float = 60.0) -> None:
        """Poll ``expression`` in the page until it is truthy."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not await self.evaluate(f"!!({expression})"):
            if loop.time() > deadline:
                raise TimeoutError(f"Page condition not met: {expression}")
            await asyncio.sleep(0.25)

    async def screenshot(self, path: Path) -> None:
        """Save the viewport as a JPEG."""
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        shot = await self.call(
            "browsingContext.captureScreenshot", context=self.context
        )
        with Image.open(io.BytesIO(base64.b64decode(shot["data"]))) as image:
            image.convert("RGB").save(path, quality=90)
        print(f"Wrote {path}")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.asynccontextmanager
async def _firefox(profile: Path):
    import aiohttp  # noqa: PLC0415

    executable = shutil.which("firefox")
    if executable is None:
        raise SystemExit("Firefox is required to capture the viewer screenshots.")
    port = _free_port()
    profile.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [executable, "--headless", "--no-remote", "--profile", str(profile)]
        + ["--remote-debugging-port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        async with aiohttp.ClientSession() as http:
            for _ in range(150):
                try:
                    ws = await http.ws_connect(f"ws://127.0.0.1:{port}/session")
                    break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.2)
            else:
                raise RuntimeError("Firefox did not start its WebDriver BiDi server.")
            browser = _Firefox(ws)
            await browser.call("session.new", capabilities={})
            tree = await browser.call("browsingContext.getTree")
            browser.context = tree["contexts"][0]["context"]
            await browser.call(
                "browsingContext.setViewport",
                context=browser.context,
                viewport=VIEWPORT,
            )
            yield browser
    finally:
        process.terminate()
        process.wait()


async def _settle(server, browser: _Firefox, condition: str, delay: float = 1.0):
    """Push pending state, then wait for the page and its renderers."""
    server.state.flush()
    await browser.wait_for(condition)
    await asyncio.sleep(delay)


async def _capture(root: Path, work: Path) -> None:
    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services import (  # noqa: PLC0415
        ParaviewArtifactService,
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

    service = PlaidDatasetService(
        ViewerConfig(datasets_root=root, initial_dataset_id=DATASET_ID)
    )
    server = build_server(service, ParaviewArtifactService(service, work / "cache"))
    state, ctrl = server.state, server.controller
    port = _free_port()
    server.start(
        port=port,
        host="127.0.0.1",
        open_browser=False,
        show_connection_info=False,
        exec_mode="task",
        timeout=0,
    )
    await server.ready
    try:
        async with _firefox(work / "profile") as browser:
            await browser.call(
                "browsingContext.navigate",
                context=browser.context,
                url=f"http://127.0.0.1:{port}/",
                wait="complete",
            )
            await browser.wait_for("document.querySelector('.v-navigation-drawer')")

            # Sample 3D view: first base, every field, colored by pressure.
            with state:
                # Show a generic root instead of the temporary directory.
                state.datasets_root_text = "/path/to/datasets"
                state.split = "train"
                state.show_globals = True
                state.active_base = state.base_options[0]
            await _settle(server, browser, "true", 2.0)
            ctrl.select_all_features()
            with state:
                state.field = "point:Pressure"
                state.show_edges = True
            ctrl.reset_camera()
            await _settle(server, browser, "document.querySelector('canvas, img')", 3.0)
            await browser.screenshot(IMAGES_DIR / "viewer.jpg")

            # Dataset Globals: extract, select three Globals, draw in 3D.
            with state:
                state.main_drawer = False
                state.viewer_tab = "globals"
                state.explore_time_step = TIME_STEP
            ctrl.extract_globals()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 120
            while state.explore_extracting or not state.explore_names:
                if loop.time() > deadline:
                    raise TimeoutError(state.explore_status)
                await asyncio.sleep(0.25)
            with state:
                state.explore_selected_globals = SCATTER_GLOBALS
            with state:
                state.explore_plot = "3D"
                state.explore_x, state.explore_y, state.explore_z = SCATTER_GLOBALS
            ctrl.draw_globals_plot()
            await _settle(
                server, browser, "document.querySelector('.js-plotly-plot')", 3
            )
            await browser.screenshot(IMAGES_DIR / "viewerglobals.jpg")
    finally:
        await server.stop()


def main() -> None:
    """Write the report, then the screenshots unless ``--report-only``."""
    parser = argparse.ArgumentParser(
        description="Regenerate the viewer documentation images and skrub report."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only write the skrub report embedded in the documentation.",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="plaid-viewer-doc-") as tmp:
        work = Path(tmp)
        # The viewer persists the datasets root it opens: keep the user's own.
        os.environ["PLAID_VIEWER_CONFIG_FILE"] = str(work / "viewer.json")
        root = work / "datasets"
        write_dataset(root)
        write_report(root, REPORT_PATH)
        print(f"Wrote {REPORT_PATH}")
        if args.report_only:
            return
        asyncio.run(_capture(root, work))


if __name__ == "__main__":
    main()
