"""Build and validate clean installations of the wheel and source distribution."""

from __future__ import annotations

import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _artifact_support import (
    clean_subprocess_environment,
    run,
    safe_extract_sdist,
    single_artifact,
    venv_python,
)


def _check_source_distribution_examples(
    *,
    artifact: Path,
    workspace: Path,
    python: Path,
    environment_variables: dict[str, str],
) -> None:
    source = safe_extract_sdist(artifact, workspace / "sdist-source")
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-compile",
            "--disable-pip-version-check",
            "--no-input",
            "--retries",
            "5",
            "--timeout",
            "60",
            f"pipls[examples] @ {artifact.resolve().as_uri()}",
        ],
        cwd=source,
        env=environment_variables,
    )
    example_environment = environment_variables.copy()
    example_environment["MPLBACKEND"] = "Agg"
    examples = (
        (
            ("01_pulp_quick_start.py",),
            Path("pulp_quick_start.pdf"),
        ),
        (
            ("03_pls_path_comparison.py", "--dataset", "pulp"),
            Path("pls_path_comparison") / "pulp_component_path_comparison.pdf",
        ),
    )
    for command, output_name in examples:
        script_name, *arguments = command
        run(
            [str(python), str(source / "examples" / script_name), *arguments],
            cwd=source,
            env=example_environment,
        )
        output = source / "examples" / "results" / output_name
        if not output.is_file():
            raise RuntimeError(
                "Source-distribution example "
                f"{' '.join(command)} did not create {output_name}."
            )

def _check_installation(
    *,
    label: str,
    artifact: Path,
    workspace: Path,
    smoke_test: Path,
    repository: Path,
    environment_variables: dict[str, str],
    check_source_example: bool,
) -> None:
    environment = workspace / f"{label}-venv"
    run_directory = workspace / f"{label}-run"
    run_directory.mkdir()

    run(
        [sys.executable, "-m", "venv", str(environment)],
        env=environment_variables,
    )
    python = venv_python(environment)
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-compile",
            "--disable-pip-version-check",
            "--no-input",
            "--retries",
            "5",
            "--timeout",
            "60",
            str(artifact),
        ],
        cwd=run_directory,
        env=environment_variables,
    )
    run(
        [str(python), str(smoke_test), str(repository), label],
        cwd=run_directory,
        env=environment_variables,
    )
    if check_source_example:
        _check_source_distribution_examples(
            artifact=artifact,
            workspace=workspace,
            python=python,
            environment_variables=environment_variables,
        )

def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment_variables = clean_subprocess_environment()

    with tempfile.TemporaryDirectory(prefix="pipls-dist-check-") as temporary:
        workspace = Path(temporary)
        artifacts = workspace / "artifacts"
        artifacts.mkdir()

        run(
            [sys.executable, "-m", "build", "--quiet", "--outdir", str(artifacts)],
            cwd=repository,
            env=environment_variables,
        )
        wheel = single_artifact(artifacts, "*.whl", "wheel")
        source_distribution = single_artifact(
            artifacts,
            "*.tar.gz",
            "source distribution",
        )

        checks = (
            ("wheel", wheel, False),
            ("sdist", source_distribution, True),
        )
        smoke_test = repository / "tools" / "installed_smoke_test.py"
        with ThreadPoolExecutor(max_workers=len(checks)) as executor:
            futures = [
                executor.submit(
                    _check_installation,
                    label=label,
                    artifact=artifact,
                    workspace=workspace,
                    smoke_test=smoke_test,
                    repository=repository,
                    environment_variables=environment_variables,
                    check_source_example=check_source_example,
                )
                for label, artifact, check_source_example in checks
            ]
            for future in futures:
                future.result()

    print("Wheel and source-distribution installation checks passed.")


if __name__ == "__main__":
    main()
