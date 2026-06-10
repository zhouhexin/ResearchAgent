from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from api.services.paper_service import (
    _extract_title_from_text,
    build_or_update_paper_index,
    match_papers_in_answer,
)


class PaperServiceTest(unittest.TestCase):
    def test_extract_title_from_pdf_text_uses_content_title(self) -> None:
        text = """
        DepthDark: Robust Monocular Depth Estimation
        for Low-Light Environments
        Ming Lu, Bolun Zheng, Chenggang Yan
        Abstract
        Low-light depth estimation is challenging.
        """

        self.assertEqual(
            _extract_title_from_text(text),
            "DepthDark: Robust Monocular Depth Estimation for Low-Light Environments",
        )

    def test_index_cache_only_extracts_new_or_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "data"
            docs_dir.mkdir()
            cache_path = Path(tmp) / "paper_index.json"
            first_pdf = docs_dir / "random_name.pdf"
            second_pdf = docs_dir / "another_name.pdf"
            first_pdf.write_bytes(b"first")

            calls: list[Path] = []

            def extractor(path: Path) -> str | None:
                calls.append(path)
                return f"Title for {path.stem}"

            first_index = build_or_update_paper_index(
                docs_dir=docs_dir,
                cache_path=cache_path,
                title_extractor=extractor,
            )
            second_index = build_or_update_paper_index(
                docs_dir=docs_dir,
                cache_path=cache_path,
                title_extractor=extractor,
            )
            second_pdf.write_bytes(b"second")
            third_index = build_or_update_paper_index(
                docs_dir=docs_dir,
                cache_path=cache_path,
                title_extractor=extractor,
            )

        self.assertEqual([entry["title"] for entry in first_index], ["Title for random_name"])
        self.assertEqual([entry["title"] for entry in second_index], ["Title for random_name"])
        self.assertEqual(
            sorted(entry["title"] for entry in third_index),
            ["Title for another_name", "Title for random_name"],
        )
        self.assertEqual([path.name for path in calls], ["random_name.pdf", "another_name.pdf"])

    def test_match_papers_in_answer_uses_title_not_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp) / "data"
            docs_dir.mkdir()
            cache_path = Path(tmp) / "paper_index.json"
            pdf_path = docs_dir / "short.pdf"
            pdf_path.write_bytes(b"pdf")
            cache_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "paper-id",
                            "title": "Always Clear Depth: Robust Monocular Depth Estimation Under Adverse Weather",
                            "path": str(pdf_path),
                            "size": pdf_path.stat().st_size,
                            "mtime_ns": pdf_path.stat().st_mtime_ns,
                            "extractor_version": 2,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            matches = match_papers_in_answer(
                "推荐 Always Clear Depth: Robust Monocular Depth Estimation Under Adverse Weather。",
                docs_dir=docs_dir,
                cache_path=cache_path,
            )

        self.assertEqual(len(matches), 1)
        self.assertEqual(
            matches[0]["title"],
            "Always Clear Depth: Robust Monocular Depth Estimation Under Adverse Weather",
        )
        self.assertEqual(matches[0]["preview_url"], "/papers/file/paper-id")
        self.assertEqual(matches[0]["download_url"], "/papers/file/paper-id?download=1")


if __name__ == "__main__":
    unittest.main()
