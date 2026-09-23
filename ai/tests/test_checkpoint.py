import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
from checkpoint import process_with_checkpoint, FIELDS

def success(item):
    return {**item, "AI": {k: "valid" for k in FIELDS}}

class RecoveryTests(unittest.TestCase):
    def test_partial_failure_retry_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def processor(item):
                calls.append(item["id"])
                if item["id"] == "bad":
                    raise ValueError("malformed")
                return success(item)
            items = [{"id": "ok", "summary": "a"}, {"id": "bad", "summary": "b"}]
            result = process_with_checkpoint(items, processor, directory, "sig", sleep=lambda _: None)
            self.assertEqual(calls, ["ok", "bad", "bad", "bad"])
            self.assertEqual(result[1]["AI_status"], "failed")
            calls.clear()
            def recovered(item):
                calls.append(item["id"])
                return success(item)
            # Failed paper is retried even if absent from the next feed.
            result = process_with_checkpoint(items[:1], recovered, directory, "sig")
            self.assertEqual(calls, ["bad"])
            self.assertTrue(all(r["AI_status"] == "success" for r in result))
            calls.clear()
            process_with_checkpoint(items, recovered, directory, "sig")
            self.assertEqual(calls, [])
            process_with_checkpoint([{"id": "ok", "summary": "changed"}], recovered, directory, "sig")
            self.assertEqual(calls, ["ok"])

    def test_interruption_preserves_completed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            def interrupted(item):
                if item["id"] == "two":
                    raise KeyboardInterrupt()
                return success(item)
            items = [{"id": "one"}, {"id": "two"}]
            with self.assertRaises(KeyboardInterrupt):
                process_with_checkpoint(items, interrupted, directory, "sig")
            calls = []
            def resume(item):
                calls.append(item["id"])
                return success(item)
            process_with_checkpoint(items, resume, directory, "sig")
            self.assertEqual(calls, ["two"])

    def test_missing_fields_and_signature_change(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def malformed(item):
                calls.append(1)
                if len(calls) == 1:
                    return {**item, "AI": {"tldr": "partial"}}
                return success(item)
            process_with_checkpoint([{"id": "one"}], malformed, directory, "sig", sleep=lambda _: None)
            self.assertEqual(len(calls), 2)
            process_with_checkpoint([{"id": "one"}], malformed, directory, "new-sig")
            self.assertEqual(len(calls), 3)

if __name__ == "__main__":
    unittest.main()

