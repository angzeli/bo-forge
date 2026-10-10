"""Check non-executing consumers against an installed package, outside the checkout."""

import os
import re
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory


def expected_diagnostics(path: Path) -> Counter:
    return Counter(
        (line, code)
        for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        for code in re.findall(r"# error: ([a-z-]+)", text)
    )


def check_consumers(fixtures: Path) -> None:
    env = dict(os.environ, PYTHONPATH="", MYPYPATH="", PYTHONDONTWRITEBYTECODE="1")
    with TemporaryDirectory(prefix="typing-cache-") as cache:
        for name in ("positive.py", "negative.py"):
            path = fixtures / name
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "--config-file", str(fixtures / "mypy.ini"),
                 "--cache-dir", cache, str(path)],
                env=env, text=True, capture_output=True, timeout=240,
            )
            if name == "positive.py":
                assert result.returncode == 0, result.stdout + result.stderr
            else:
                actual = Counter(
                    (int(line), code) for line, code in re.findall(
                        r"negative\.py:(\d+): error: .*\[([a-z-]+)\]$",
                        result.stdout, re.MULTILINE,
                    )
                )
                assert result.returncode == 1, result.stdout + result.stderr
                assert actual == expected_diagnostics(path), result.stdout + result.stderr
                assert result.stdout.count(": error:") == sum(actual.values()), result.stdout
            assert not result.stderr, result.stderr
            print(f"Installed consumer {name}: passed", flush=True)


def main() -> None:
    import bo_forge

    source = Path(os.environ["SOURCE_ROOT"]).resolve()
    package = Path(bo_forge.__file__).resolve().parent
    assert not Path.cwd().resolve().is_relative_to(source)
    assert not package.is_relative_to(source)
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    assert (package / "py.typed").read_bytes() == b""
    assert not os.environ.get("PYTHONPATH") and not os.environ.get("MYPYPATH")
    print(f"Typing {bo_forge.__version__} from {package}; Mypy {version('mypy')}; "
          f"pandas-stubs {version('pandas-stubs')}", flush=True)
    check_consumers(Path(__file__).resolve().parent / "typing_contract")


if __name__ == "__main__":
    main()
