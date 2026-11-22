"""Logger shim: uses loguru (pinned in requirements.txt) when installed, stdlib logging otherwise."""
try:
    from loguru import logger  # noqa: F401
except ImportError:  # pragma: no cover - only hit when loguru is absent
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
    )
    logger = logging.getLogger("sentinel")
