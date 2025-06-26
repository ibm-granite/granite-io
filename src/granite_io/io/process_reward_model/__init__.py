# SPDX-License-Identifier: Apache-2.0

# Local
from .best_of_n import PRMBestOfNCompositeIOProcessor, ProcessRewardModelIOProcessor

# Expose public symbols at `granite_io.io.process_reward_model`
# to save users from typing

__all__ = ["ProcessRewardModelIOProcessor", "PRMBestOfNCompositeIOProcessor"]
