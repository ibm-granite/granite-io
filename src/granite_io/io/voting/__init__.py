"""
The voting module holds I/O processors that perform various types of majority voting.
"""

# Local
from granite_io.io.voting.mbrd import MBRDMajorityVotingProcessor  # noqa: F401
from granite_io.io.voting.simple import (  # noqa: F401
    MajorityVotingProcessor,
    integer_normalizer,
)
