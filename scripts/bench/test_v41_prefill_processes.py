import unittest
from pathlib import Path

from v41_prefill_packet import resident_models


class ResidentModels(unittest.TestCase):
    def test_expected_ghost_memory_is_recorded_but_does_not_block_models(self):
        ghost=Path.home()/"ghost/host-cache/44d4fe0f/headless/Headless.app/Contents/MacOS/shadps4"
        source=f"PID RSS COMM\n53007 28692432 {ghost}\n"
        self.assertEqual(resident_models(source),[])

    def test_real_models_and_large_unknown_processes_still_block(self):
        rows=["1 170000000 /model/target/release/qwen", "2 2000000 mlx-serve", "3 24000000 python3"]
        self.assertEqual(resident_models("PID RSS COMM\n"+"\n".join(rows)),rows)

    def test_lookalikes_are_not_exempt_and_only_exact_excluded_pid_is_ignored(self):
        rows=["4 28692432 /elsewhere/headless/Headless.app/Contents/MacOS/shadps4",
              f"5 28692432 {Path.home()}/ghost/host-cache/hash/model"]
        source="PID RSS COMM\n"+"\n".join(rows)
        self.assertEqual(resident_models(source,exclude_pid=4),rows[1:])


if __name__=="__main__":unittest.main()
