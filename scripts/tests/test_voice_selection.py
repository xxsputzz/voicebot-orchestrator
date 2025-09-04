#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_interactive_pipeline import test_tortoise_tts

if __name__ == "__main__":
    test_tortoise_tts()
