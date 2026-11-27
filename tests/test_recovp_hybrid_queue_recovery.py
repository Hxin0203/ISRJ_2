import unittest

from tools.recover_recovp_hybrid_queue import admission_failure


class AdmissionFailureTests(unittest.TestCase):
    def test_resource_wait_does_not_become_training_error(self):
        self.assertTrue(admission_failure(
            "Traceback (most recent call last):\n"
            "tools.server_gpu_admission.NoGpuAvailable: GPU-0 at/above 85%\n"))

    def test_other_terminal_errors_are_not_hidden(self):
        self.assertFalse(admission_failure(
            "Earlier NoGpuAvailable: resource wait\n"
            "RuntimeError: frozen run config changed\n"))
        self.assertFalse(admission_failure(""))


if __name__ == "__main__":
    unittest.main()
