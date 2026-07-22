import unittest

import torch

from qwen_kernel_lab.ops import add_rms_norm, add_rms_norm_residual, silu_mul


class OperatorCorrectnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_add_rms_norm_matches_oracle(self):
        for shape in ((1, 1, 64), (2, 17, 128), (3, 256)):
            with self.subTest(shape=shape):
                x = torch.randn(shape)
                residual = torch.randn(shape)
                weight = torch.randn(shape[-1])
                actual = add_rms_norm(x, residual, weight, use_native=False)

                summed = x + residual
                expected = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
                expected = expected * weight
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_add_rms_norm_residual_returns_both_outputs(self):
        x = torch.randn(2, 17, 128)
        residual = torch.randn_like(x)
        weight = torch.randn(128)
        normalized, summed = add_rms_norm_residual(
            x, residual, weight, use_native=False
        )
        torch.testing.assert_close(summed, x + residual)
        expected = add_rms_norm(x, residual, weight, use_native=False)
        torch.testing.assert_close(normalized, expected)

    def test_silu_mul_matches_oracle(self):
        gate = torch.randn(4, 32, 256)
        up = torch.randn_like(gate)
        actual = silu_mul(gate, up, use_native=False)
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(actual, expected)

    def test_reference_keeps_autograd(self):
        x = torch.randn(2, 8, requires_grad=True)
        residual = torch.randn(2, 8, requires_grad=True)
        weight = torch.ones(8, requires_grad=True)
        add_rms_norm(x, residual, weight).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(residual.grad)
        self.assertIsNotNone(weight.grad)

    def test_invalid_shapes_raise(self):
        with self.assertRaises(ValueError):
            add_rms_norm(torch.randn(2, 8), torch.randn(2, 7), torch.ones(8))
        with self.assertRaises(ValueError):
            silu_mul(torch.randn(2, 8), torch.randn(2, 7))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class CudaNativeTests(unittest.TestCase):
    def test_native_fp16_matches_reference(self):
        from qwen_kernel_lab import native_extension_loaded

        if not native_extension_loaded():
            self.skipTest("native extension is not installed")
        x = torch.randn(2, 128, 896, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)
        weight = torch.randn(896, device="cuda", dtype=torch.float16)
        expected = add_rms_norm(x, residual, weight, use_native=False)
        actual = add_rms_norm(x, residual, weight, use_native=True)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_native_add_rms_norm_residual_fp16_matches_reference(self):
        from qwen_kernel_lab import native_extension_loaded

        if not native_extension_loaded():
            self.skipTest("native extension is not installed")
        x = torch.randn(2, 128, 896, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)
        weight = torch.randn(896, device="cuda", dtype=torch.float16)
        expected_norm, expected_sum = add_rms_norm_residual(
            x, residual, weight, use_native=False
        )
        actual_norm, actual_sum = add_rms_norm_residual(
            x, residual, weight, use_native=True
        )
        torch.testing.assert_close(actual_norm, expected_norm, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(actual_sum, expected_sum, rtol=0, atol=0)

    def test_native_silu_mul_fp16_matches_reference(self):
        from qwen_kernel_lab import native_extension_loaded

        if not native_extension_loaded():
            self.skipTest("native extension is not installed")
        for shape in ((1, 1, 4864), (3, 17)):
            with self.subTest(shape=shape):
                gate = torch.randn(shape, device="cuda", dtype=torch.float16)
                up = torch.randn_like(gate)
                expected = silu_mul(gate, up, use_native=False)
                actual = silu_mul(gate, up, use_native=True)
                torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()

