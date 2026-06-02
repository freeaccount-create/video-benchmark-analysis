"""
Tests for metadata schemas and loading utilities.
"""

import pytest
from directorbench.metadata.schema import (
    CameraType,
    DiagnosisReport,
    DimensionScore,
    EvalDimension,
    EvalResult,
    ShotDescription,
    VideoMetadata,
)


class TestShotDescription:
    def test_valid_shot(self):
        # TODO: Assert ShotDescription is created with valid fields
        pass

    def test_end_before_start_raises(self):
        # TODO: Assert ValueError when end_time <= start_time
        pass

    def test_duration_property(self):
        # TODO: Assert shot.duration == end_time - start_time
        pass


class TestVideoMetadata:
    def test_valid_metadata(self):
        # TODO: Assert VideoMetadata created with minimum required fields
        pass

    def test_shot_beyond_duration_raises(self):
        # TODO: Assert ValueError when a shot's end_time > duration_seconds
        pass


class TestDimensionScore:
    def test_score_bounds(self):
        # TODO: Assert score is clamped to [0, 10]
        pass

    def test_grade_label(self):
        # TODO: Assert grade_label returns correct letter for various scores
        pass


class TestVideoLoader:
    def test_load_metadata_from_json(self, tmp_path):
        # TODO: Write sample JSON, load it, assert fields match
        pass

    def test_missing_file_raises(self):
        # TODO: Assert FileNotFoundError for non-existent path
        pass
