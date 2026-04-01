import numpy as np
import pytest
import cv2
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.stereo_generator import StereoGenerator
from src.inpainter import Inpainter


def make_test_image(width=200, height=150):
    """Create a synthetic RGB gradient image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)  # R gradient
    img[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]  # G gradient
    img[:, :, 2] = 128
    return img


def make_test_depth(width=200, height=150):
    """Create a synthetic depth map (horizontal gradient, 0 to 1)."""
    depth = np.linspace(0, 1, width, dtype=np.float32)
    return np.tile(depth, (height, 1))


# --- StereoGenerator tests ---

class TestStereoGenerator:

    def test_generate_stereo_pair_shapes(self):
        image = make_test_image()
        depth = make_test_depth()
        gen = StereoGenerator(ipd_mm=12.0)

        left, right, left_mask, right_mask = gen.generate_stereo_pair(image, depth)

        assert left.shape == image.shape
        assert right.shape == image.shape
        assert left_mask.shape == (150, 200)
        assert right_mask.shape == (150, 200)

    def test_generate_stereo_pair_types(self):
        image = make_test_image()
        depth = make_test_depth()
        gen = StereoGenerator(ipd_mm=12.0)

        left, right, left_mask, right_mask = gen.generate_stereo_pair(image, depth)

        assert left.dtype == np.uint8
        assert right.dtype == np.uint8
        assert left_mask.dtype == np.uint8
        assert right_mask.dtype == np.uint8

    def test_left_view_is_original(self):
        image = make_test_image()
        depth = make_test_depth()
        gen = StereoGenerator(ipd_mm=12.0)

        left, _, _, _ = gen.generate_stereo_pair(image, depth)

        np.testing.assert_array_equal(left, image)

    def test_left_mask_is_zero(self):
        image = make_test_image()
        depth = make_test_depth()
        gen = StereoGenerator(ipd_mm=12.0)

        _, _, left_mask, _ = gen.generate_stereo_pair(image, depth)

        assert np.all(left_mask == 0)

    def test_right_view_is_shifted(self):
        image = make_test_image()
        depth = make_test_depth()
        gen = StereoGenerator(ipd_mm=12.0)

        left, right, _, _ = gen.generate_stereo_pair(image, depth)

        # Right view should differ from left (pixels are shifted)
        assert not np.array_equal(left, right)

    def test_zero_depth_no_shift(self):
        image = make_test_image()
        depth = np.zeros((150, 200), dtype=np.float32)
        gen = StereoGenerator(ipd_mm=12.0)

        left, right, _, right_mask = gen.generate_stereo_pair(image, depth)

        # With zero depth, disparity is zero, so right should equal left
        np.testing.assert_array_equal(left, right)
        assert np.all(right_mask == 0)

    def test_shape_mismatch_raises(self):
        image = make_test_image(200, 150)
        depth = np.zeros((100, 200), dtype=np.float32)
        gen = StereoGenerator(ipd_mm=12.0)

        with pytest.raises(ValueError, match="Shape mismatch"):
            gen.generate_stereo_pair(image, depth)

    def test_disparity_scales_with_ipd(self):
        depth = make_test_depth()
        gen_small = StereoGenerator(ipd_mm=6.0)
        gen_large = StereoGenerator(ipd_mm=24.0)

        disp_small = gen_small.calculate_disparity(depth, 200)
        disp_large = gen_large.calculate_disparity(depth, 200)

        # Larger IPD should produce larger disparity
        assert disp_large.max() > disp_small.max()
        # Should scale linearly with IPD
        np.testing.assert_allclose(disp_large, disp_small * 4.0, atol=1e-5)


# --- Output format tests ---

class TestOutputFormats:

    def test_side_by_side_dimensions(self):
        left = make_test_image(200, 150)
        right = make_test_image(200, 150)
        gen = StereoGenerator()

        sbs = gen.create_side_by_side(left, right)

        assert sbs.shape == (150, 400, 3)

    def test_anaglyph_dimensions(self):
        left = make_test_image(200, 150)
        right = make_test_image(200, 150)
        gen = StereoGenerator()

        anaglyph = gen.create_anaglyph(left, right)

        assert anaglyph.shape == (150, 200, 3)

    def test_anaglyph_channels(self):
        left = make_test_image(200, 150)
        right = make_test_image(200, 150)
        gen = StereoGenerator()

        anaglyph = gen.create_anaglyph(left, right)

        # Red channel from left, green/blue from right
        np.testing.assert_array_equal(anaglyph[:, :, 0], left[:, :, 0])
        np.testing.assert_array_equal(anaglyph[:, :, 1], right[:, :, 1])
        np.testing.assert_array_equal(anaglyph[:, :, 2], right[:, :, 2])


# --- Inpainter tests ---

class TestInpainter:

    def test_inpaint_fills_holes(self):
        image = make_test_image()
        # Create a mask with a hole in the middle
        mask = np.zeros((150, 200), dtype=np.uint8)
        mask[50:100, 80:120] = 255

        inpainter = Inpainter(method='telea', radius=3)
        result = inpainter.inpaint_view(image, mask)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_inpaint_no_mask_returns_copy(self):
        image = make_test_image()
        mask = np.zeros((150, 200), dtype=np.uint8)

        inpainter = Inpainter(method='telea', radius=3)
        result = inpainter.inpaint_view(image, mask)

        np.testing.assert_array_equal(result, image)

    def test_inpaint_stereo_pair(self):
        left = make_test_image()
        right = make_test_image()
        left_mask = np.zeros((150, 200), dtype=np.uint8)
        right_mask = np.zeros((150, 200), dtype=np.uint8)
        right_mask[10:20, 10:20] = 255

        inpainter = Inpainter(method='telea', radius=3)
        left_out, right_out = inpainter.inpaint_stereo_pair(left, right, left_mask, right_mask)

        assert left_out.shape == left.shape
        assert right_out.shape == right.shape

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            Inpainter(method='invalid')


# --- Full pipeline integration test (mocked depth) ---

class TestFullPipeline:

    def test_image_pipeline_end_to_end(self):
        """Test the full pipeline: image -> depth -> stereo -> inpaint."""
        image = make_test_image(320, 240)
        depth = make_test_depth(320, 240)

        gen = StereoGenerator(ipd_mm=12.0)
        inpainter = Inpainter(method='telea', radius=3)

        left, right, left_mask, right_mask = gen.generate_stereo_pair(image, depth)
        left_final, right_final = inpainter.inpaint_stereo_pair(left, right, left_mask, right_mask)

        assert left_final.shape == (240, 320, 3)
        assert right_final.shape == (240, 320, 3)
        assert left_final.dtype == np.uint8
        assert right_final.dtype == np.uint8

    def test_video_pipeline_synthetic(self):
        """Test video processing with synthetic frames and mocked depth."""
        width, height, num_frames = 160, 120, 5
        gen = StereoGenerator(ipd_mm=12.0)
        inpainter = Inpainter(method='telea', radius=3)

        # Write synthetic video
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            output_path = Path(tmpdir) / "test_right.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
            for i in range(num_frames):
                frame = np.full((height, width, 3), i * 50, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            # Process each frame manually (mocking depth estimation)
            cap = cv2.VideoCapture(str(video_path))
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                depth = make_test_depth(width, height)
                left, right, lm, rm = gen.generate_stereo_pair(frame_rgb, depth)
                _, right_final = inpainter.inpaint_stereo_pair(left, right, lm, rm)
                out.write(cv2.cvtColor(right_final, cv2.COLOR_RGB2BGR))

            cap.release()
            out.release()

            # Verify output video
            cap_out = cv2.VideoCapture(str(output_path))
            out_frames = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
            out_w = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_h = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_out.release()

            assert out_frames == num_frames
            assert out_w == width
            assert out_h == height

    def test_sbs_video_dimensions(self):
        """Verify SBS output produces 2x width video."""
        width, height = 160, 120
        gen = StereoGenerator(ipd_mm=12.0)
        inpainter = Inpainter(method='telea', radius=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test.mp4"
            output_path = Path(tmpdir) / "test_sbs.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
            for _ in range(3):
                writer.write(np.zeros((height, width, 3), dtype=np.uint8))
            writer.release()

            # Process with SBS output
            cap = cv2.VideoCapture(str(video_path))
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width * 2, height))

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                depth = make_test_depth(width, height)
                left, right, lm, rm = gen.generate_stereo_pair(frame_rgb, depth)
                _, right_final = inpainter.inpaint_stereo_pair(left, right, lm, rm)
                sbs = gen.create_side_by_side(left, right_final)
                out.write(cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))

            cap.release()
            out.release()

            cap_out = cv2.VideoCapture(str(output_path))
            out_w = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap_out.release()

            assert out_w == width * 2


# --- DepthEstimator config test (mocked model loading) ---

class TestDepthEstimatorConfig:

    @patch('src.depth_estimator.pipeline')
    @patch('src.depth_estimator.torch')
    def test_model_size_parameter(self, mock_torch, mock_pipeline):
        """Verify the correct model ID is constructed from model_size."""
        mock_torch.cuda.is_available.return_value = False
        mock_pipeline.return_value = MagicMock()

        from src.depth_estimator import DepthEstimator

        for size in ['small', 'base', 'large']:
            DepthEstimator(model_size=size, device=-1)
            call_kwargs = mock_pipeline.call_args
            assert f"Depth-Anything-V2-{size}-hf" in call_kwargs[1]['model']

    @patch('src.depth_estimator.pipeline')
    @patch('src.depth_estimator.torch')
    def test_cpu_device(self, mock_torch, mock_pipeline):
        """Verify device=-1 forces CPU."""
        mock_torch.cuda.is_available.return_value = True
        mock_pipeline.return_value = MagicMock()

        from src.depth_estimator import DepthEstimator
        DepthEstimator(model_size='small', device=-1)

        call_kwargs = mock_pipeline.call_args
        assert call_kwargs[1]['device'] == -1
