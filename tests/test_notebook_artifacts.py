from io import BytesIO

import pytest
from matplotlib.figure import Figure
from PIL import Image

from notebook_assurance.contracts import _artifact


def image_bytes(extension):
    output = BytesIO()
    if extension == ".png":
        Image.new("RGB", (4, 4), "white").save(output, format="PNG")
    else:
        figure = Figure(figsize=(1, 1))
        figure.subplots().plot([0, 1], [0, 1])
        figure.savefig(output, format="pdf")
    return output.getvalue()


@pytest.mark.parametrize("extension", [".png", ".pdf"])
@pytest.mark.parametrize("damage", ["header_only", "truncated", "corrupt"])
def test_artifact_rejects_undecodable_or_incomplete_figure(tmp_path, extension, damage):
    content = image_bytes(extension)
    if damage == "header_only":
        content = content[:8]
    elif damage == "truncated":
        content = content[:len(content) // 2]
    elif extension == ".png":
        content = content[:35] + b"broken" + content[41:]
    else:
        content = content.replace(b"/Type /Page ", b"/Type /Gone ")
    path = tmp_path / ("figure" + extension)
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Invalid image artifact"):
        _artifact(path)


@pytest.mark.parametrize("extension", [".png", ".pdf"])
def test_artifact_accepts_complete_generated_figure(tmp_path, extension):
    path = tmp_path / ("figure" + extension)
    path.write_bytes(image_bytes(extension))
    _artifact(path)
