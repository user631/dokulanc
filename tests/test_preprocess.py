import random

import cv2
import numpy as np
import pytest

from dokulanc.preprocess import (apply_profile, classifier_image, estimate_skew, find_document_quad,
                                 normalize_document, rotate)
from dokulanc.synth import Generator, photo_effect


@pytest.fixture(scope="module")
def letter():
    img = Generator(5).generate("level").image
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


@pytest.mark.parametrize("angle", [-3.0, 2.0, 4.5])
def test_deskew_estimates_rotation(letter, angle):
    rotated = rotate(letter, angle)
    assert estimate_skew(rotated) == pytest.approx(-angle, abs=0.5)


def test_document_detected_on_photo():
    gen = Generator(9)
    found = 0
    for i in range(6):
        photo = photo_effect(gen.generate("szamla").image, random.Random(i))
        img = cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2BGR)
        found += find_document_quad(img) is not None
    assert found >= 5


def test_normalize_on_clean_page_keeps_size(letter):
    out, info = normalize_document(letter)
    assert not info["perspektiva_javitva"]
    assert abs(info["dolesszog"]) < 0.5
    assert out.shape[:2] == letter.shape[:2]


def test_profiles_and_classifier_input(letter):
    for profile in ("alap", "tablazat", "nyugta", "urlap"):
        gray, scale = apply_profile(letter, profile)
        assert gray.ndim == 2 and scale >= 1.0
    small = classifier_image(letter, 256)
    assert small.size == (256, 256) and small.mode == "L"


def test_quad_touching_image_border_is_rejected():
    img = np.full((400, 300, 3), 170, np.uint8)
    img[60:380, 40:300] = 235
    quad = find_document_quad(img)
    assert quad is None or sum(1 for x, y in quad if x > 0.98 * 300 or y < 0.02 * 400) <= 1
