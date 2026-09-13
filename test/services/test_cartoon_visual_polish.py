import unittest

from app.models.schema import VideoAspect, VideoParams
from app.services.video import _subtitle_y_for_layout


class TestCartoonSubtitleSafeZones(unittest.TestCase):
    def test_cartoon_bottom_captions_stay_above_character_band_for_all_aspects(self):
        for aspect, size in (
            (VideoAspect.portrait, (1080, 1920)),
            (VideoAspect.square, (1080, 1080)),
            (VideoAspect.landscape, (1920, 1080)),
        ):
            params = VideoParams(
                video_subject="cartoon",
                video_source="ai_cartoon",
                video_aspect=aspect,
                subtitle_position="bottom",
            )
            width, height = size
            y = _subtitle_y_for_layout(params, width, height, 240)
            self.assertGreaterEqual(y, height * 0.025)
            self.assertLessEqual(y + 240, height * 0.76)

    def test_long_cartoon_caption_is_clamped_without_negative_offset(self):
        params = VideoParams(
            video_subject="cartoon",
            video_source="ai_cartoon",
            video_aspect=VideoAspect.portrait,
            subtitle_position="bottom",
        )
        y = _subtitle_y_for_layout(params, 1080, 1920, 1000)
        self.assertGreaterEqual(y, 48)
        self.assertLessEqual(y, 1920 - 1000)

    def test_non_cartoon_bottom_position_keeps_legacy_anchor(self):
        params = VideoParams(
            video_subject="stock",
            video_source="pexels",
            video_aspect=VideoAspect.portrait,
            subtitle_position="bottom",
        )
        self.assertEqual(_subtitle_y_for_layout(params, 1080, 1920, 240), 1584.0)

    def test_explicit_top_and_center_positions_are_not_overridden(self):
        for position in ("top", "center"):
            params = VideoParams(
                video_subject="cartoon",
                video_source="ai_cartoon",
                video_aspect=VideoAspect.portrait,
                subtitle_position=position,
            )
            resolved = _subtitle_y_for_layout(params, 1080, 1920, 240)
            self.assertEqual(resolved, "center" if position == "center" else position)

    def test_cartoon_custom_and_two_thirds_positions_use_the_safe_band(self):
        for position in ("two_thirds_bottom", "custom"):
            for display_mode in ("sentence", "word_by_word"):
                params = VideoParams(
                    video_subject="cartoon",
                    video_source="ai_cartoon",
                    video_aspect=VideoAspect.portrait,
                    subtitle_position=position,
                    subtitle_display_mode=display_mode,
                    custom_position=63.0,
                )
                y = _subtitle_y_for_layout(params, 1080, 1920, 240)
                self.assertGreaterEqual(y, 48)
                self.assertLessEqual(y + 240, 1920 * 0.49)
