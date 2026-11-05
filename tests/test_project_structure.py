from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist():
    required_directories = [
        "data/raw",
        "data/processed",
        "sql/schema",
        "sql/staging",
        "sql/transformations",
        "sql/analysis",
        "sql/views",
        "src/data",
        "src/analysis",
        "src/statistics",
        "src/modeling",
        "notebooks",
        "powerbi",
        "powerbi/screenshots",
        "tests",
        "results",
        "scripts",
    ]

    for directory in required_directories:
        assert (PROJECT_ROOT / directory).is_dir(), (
            f"Missing directory: {directory}"
        )


def test_required_files_exist():
    required_files = [
        "requirements.txt",
        ".gitignore",
        "src/data/raw_audit.py",
        "src/data/validator.py",
    ]

    for file in required_files:
        assert (PROJECT_ROOT / file).is_file(), (
            f"Missing file: {file}"
        )