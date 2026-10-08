"""Concurrent runners must not share or clear each other's error reports."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from runner.inference import InferenceRunner


class ErrorDirectoryTest(unittest.TestCase):
    def make_runner(self, output):
        runner = object.__new__(InferenceRunner)
        runner.configs = SimpleNamespace(dump_dir=str(output))
        runner.init_basics()
        return runner

    def test_start_preserves_previous_and_active_error_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            old = output / "ERR" / "previous.txt"
            old.parent.mkdir()
            old.write_text("previous failure")
            first = self.make_runner(output)
            first_report = Path(first.error_dir) / "target.txt"
            first_report.write_text("first failure")
            second = self.make_runner(output)

            self.assertTrue(old.is_file())
            self.assertTrue(first_report.is_file())
            self.assertEqual(old.read_text(), "previous failure")
            self.assertEqual(first_report.read_text(), "first failure")
            self.assertNotEqual(first.error_dir, second.error_dir)
            Path(second.error_dir).rmdir()
            self.assertEqual(first_report.read_text(), "first failure")
            self.assertEqual(first.dump_dir, str(output))

    def test_five_concurrent_runners_write_the_same_error_filename_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def write_error(index):
                runner = self.make_runner(output)
                report = Path(runner.error_dir) / "target.txt"
                report.write_text(str(index))
                return report

            with ThreadPoolExecutor(max_workers=5) as pool:
                reports = list(pool.map(write_error, range(5)))

            self.assertEqual(len({report.parent for report in reports}), 5)
            for index, report in enumerate(reports):
                self.assertEqual(report.read_text(), str(index))
                self.assertEqual(report.parent.parent, output / "ERR")


if __name__ == "__main__":
    unittest.main()
