"""Exercise repeated lifts and cleanup using the real ComfyUI loader on CPU."""
import unittest
from unittest.mock import patch

import test_avatar
import torch
import comfy.model_management as mm
import comfy.model_patcher
from selflift_avatar_test import h3_upscaler


class UpscalerMemoryTests(unittest.TestCase):
    def setUp(self):
        self.model = h3_upscaler.LatentResizer3D(in_blocks=1, out_blocks=1, channels=32)
        self.model.eval().requires_grad_(False)
        self.patcher = comfy.model_patcher.ModelPatcher(
            self.model, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
        self.input = torch.randn(1, 24, 3, 4, 4)

    def test_repeated_lifts_release_model_and_preserve_results(self):
        with patch.object(h3_upscaler, '_load_model', return_value=self.patcher):
            first = h3_upscaler.learned_latent_lift(self.input, (8, 8), 'test', device='cpu')
            self.assertNotIn(self.patcher, mm.loaded_models())
            second = h3_upscaler.learned_latent_lift(self.input, (8, 8), 'test', device='cpu')
        self.assertNotIn(self.patcher, mm.loaded_models())
        self.assertEqual(first.shape, (1, 24, 3, 8, 8))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_failed_lift_also_releases_model(self):
        with patch.object(h3_upscaler, '_load_model', return_value=self.patcher), \
             patch.object(self.model, 'forward', side_effect=RuntimeError('test interruption')):
            with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                h3_upscaler.learned_latent_lift(self.input, (8, 8), 'test')
        self.assertNotIn(self.patcher, mm.loaded_models())


if __name__ == '__main__':
    unittest.main(argv=[__file__], verbosity=2)
