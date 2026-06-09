from __future__ import annotations

import sys
import types
import unittest

from api.services import qa_service


class QaApiTemperatureTest(unittest.TestCase):
    def test_public_question_passes_api_temperature_to_answer_query(self) -> None:
        captured: dict = {}

        def fake_answer_query(**kwargs):
            captured.update(kwargs)
            return '{"answer": "ok"}'

        fake_app = types.SimpleNamespace(answer_query=fake_answer_query)
        original_app = sys.modules.get("app")
        original_find_run_id = qa_service._find_run_id
        original_api_temperature = getattr(qa_service.settings, "API_TEMPERATURE", None)

        sys.modules["app"] = fake_app
        qa_service._find_run_id = lambda run_label: "run-id"
        qa_service.settings.API_TEMPERATURE = 0.0
        try:
            result = qa_service.ask_public_question("same question")
        finally:
            qa_service._find_run_id = original_find_run_id
            if original_api_temperature is None:
                delattr(qa_service.settings, "API_TEMPERATURE")
            else:
                qa_service.settings.API_TEMPERATURE = original_api_temperature
            if original_app is None:
                sys.modules.pop("app", None)
            else:
                sys.modules["app"] = original_app

        self.assertEqual(result["answer"], "ok")
        self.assertEqual(captured.get("temperature"), 0.0)


if __name__ == "__main__":
    unittest.main()
