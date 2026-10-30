"""Check a built viewer image with real CGNS rendering and HTTP startup.

Run from the repository root with Python 3.11+ and a running Docker daemon:
``python docker/smoke_test_viewer_image.py [image-tag]``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

RENDER_SAMPLE = """
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from plaid.viewer.trame_app.server import _VtkPipeline

pipeline = _VtkPipeline()
pipeline.load(next(Path('/datasets/fixture/data').rglob('*.cgns')))
pipeline.update(
    field=None, association='point', cmap='viridis',
    show_edges=True, color_range=(0.0, 1.0),
)
assert pipeline._current_dataset.GetNumberOfPoints() > 0
pipeline.renderer.ResetCamera()
pipeline.render_window.SetSize(320, 240)
pipeline.render_window.Render()
capture = vtk.vtkWindowToImageFilter()
capture.SetInput(pipeline.render_window)
capture.Update()
pixels = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
assert np.ptp(pixels, axis=0).max() > 0, 'Rendered frame is blank'
print('Sample rendered with', pipeline.render_window.GetClassName())
pipeline.render_window.Finalize()
"""


def docker(*args: str) -> str:
    """Run a Docker command and return its stripped standard output.

    Args:
        *args: Arguments passed to the Docker CLI.

    Returns:
        Captured standard output, without surrounding whitespace.
    """
    result = subprocess.run(
        ["docker", *args], check=True, text=True, stdout=subprocess.PIPE
    )
    return result.stdout.strip()


def main() -> None:
    """Check HTTP, non-root execution, read-only data, and sample rendering."""
    image = sys.argv[1] if len(sys.argv) > 1 else "plaid-viewer:local"
    fixture = Path(__file__).resolve().parents[1] / "tests/containers/dataset_cgns"
    if not fixture.is_dir():
        raise FileNotFoundError(f"Missing CGNS fixture: {fixture}")
    container = docker(
        "run",
        "--detach",
        "--init",
        "--publish",
        "127.0.0.1::8080",
        "--mount",
        f"type=bind,source={fixture},target=/datasets/fixture,readonly",
        image,
    )
    try:
        port = docker("port", container, "8080/tcp").rsplit(":", 1)[1]
        for _ in range(90):
            if docker("inspect", "--format", "{{.State.Running}}", container) != "true":
                raise RuntimeError("Viewer container exited before HTTP startup")
            try:
                with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError, ConnectionError):
                # Connection failures are expected while the viewer starts.
                # Retry after the delay below; the loop bounds the wait.
                pass
            time.sleep(1)
        else:
            raise TimeoutError("Viewer did not serve HTTP within 90 attempts")
        assert docker("exec", container, "id", "-u") != "0"
        mounts = docker(
            "inspect", "--format", "{{range .Mounts}}{{.RW}}{{end}}", container
        )
        assert mounts == "false", f"Dataset mount is not read-only: {mounts}"
        print(docker("exec", container, "python", "-c", RENDER_SAMPLE))
        print("Viewer HTTP, non-root user, and read-only dataset checks passed")
    finally:
        try:
            print(docker("logs", container))
        finally:
            docker("rm", "--force", container)
    print("Smoke Test OK")


if __name__ == "__main__":
    main()
