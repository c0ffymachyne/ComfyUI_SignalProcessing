import nox
from nox import Session

PYTHON_VERSIONS = ["3.10.12"]
REUSE_ENV = True


@nox.session(python=PYTHON_VERSIONS, tags=["style"], reuse_venv=REUSE_ENV)
def lint(session: Session) -> None:

    session.install("black")
    session.install("flake8")
    session.install("mypy")

    session.run("black", ".")
    session.run("flake8", "--max-line-length=100", "--ignore=E501,E203,W503", ".")
    session.run(
        "mypy",
        ".",
        "--ignore-missing-imports",
        "--strict",
        "--show-error-codes",
    )


@nox.session(python=PYTHON_VERSIONS, tags=["tests"], reuse_venv=REUSE_ENV)
def tests(session: Session) -> None:
    """Run pytest tests with Scalene profiling."""

    session.install("scalene", "pytest", "pytest-cov", "pytest-xdist")
    requirements = nox.project.load_toml("pyproject.toml")["project"]["dependencies"]
    for _, v in requirements.items():
        session.install(*v)

    project_name = nox.project.load_toml("pyproject.toml")["project"]["name"]

    pytest_path = session.run("which", "pytest", external=True, silent=True).strip()

    if not session.posargs:
        with session.cd(".."):
            session.run(
                "scalene",
                # "--profile-all",
                pytest_path,
                f"--cov={project_name}",
                "--cov-report=term",
                "--cov-report=html",
                f"{project_name}/tests/",
                external=True,
            )
    else:
        with session.cd(".."):
            session.run(
                "pytest",
                "--rootdir=.",
                f"--cov={project_name}",
                "--cov-report=term",
                "--cov-report=html",
                f"{project_name}/tests/",
                "-k",
                session.posargs[0],
                external=True,
            )
