from pathlib import Path

def test_requirements_file_exists():
    assert Path("requirements.txt").read_text(encoding="utf-8").strip()

def test_download_helper_exists():
    assert Path("download_data.py").is_file()