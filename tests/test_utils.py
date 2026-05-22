"""Tests for ``preframr_tokens.utils``.

``get_logger`` has no callers inside ``preframr_tokens`` itself, but the
sibling ``preframr`` repo uses it from ~9 modules (parse.py,
stftokenize.py, inference/predict.py, train/trainer.py,
inference/render_play.py, integration_tests/*). It is part of the public
surface and must not be removed without coordinating with that repo.
"""

import logging
import unittest

from preframr_tokens.utils import get_logger


class TestGetLogger(unittest.TestCase):
    def test_returns_logger(self):
        logger = get_logger()
        self.assertIsInstance(logger, logging.Logger)

    def test_idempotent_handler_install(self):
        # Repeated calls must not stack handlers on the module logger.
        # (Under pytest the root logger has a handler, so
        # ``hasHandlers()`` short-circuits and no install ever happens —
        # that's still the "no growth" contract we care about.)
        logger = get_logger()
        before = len(logger.handlers)
        get_logger()
        get_logger(level="INFO")
        self.assertEqual(len(logger.handlers), before)

    def test_level_argument_applied(self):
        logger = get_logger(level="DEBUG")
        self.assertEqual(logger.level, logging.DEBUG)
        logger = get_logger(level="warning")
        self.assertEqual(logger.level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
