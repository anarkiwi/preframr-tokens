"""Tests for ``preframr_tokens.utils``. ``get_logger`` has no in-package callers but is consumed by ~9 modules in the sibling ``preframr`` repo; it is part of the public surface."""

import logging
import unittest

from preframr_tokens.utils import get_logger


class TestGetLogger(unittest.TestCase):
    def test_returns_logger(self):
        logger = get_logger()
        self.assertIsInstance(logger, logging.Logger)

    def test_idempotent_handler_install(self):
        """Repeated get_logger calls must not stack handlers on the module logger."""
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
