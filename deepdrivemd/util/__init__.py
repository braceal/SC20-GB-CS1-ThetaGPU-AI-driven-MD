import logging, logging.handlers
import time

from .filelock import FileLock
from .copy_sender import RemoteCopySender, LocalCopySender


class PeriodicMemoryHandler(logging.handlers.MemoryHandler):
    def __init__(
        self,
        capacity,
        flushLevel=logging.ERROR,
        target=None,
        flushOnClose=True,
        flush_period=30,
    ):
        super().__init__(
            capacity,
            flushLevel=flushLevel,
            target=target,
            flushOnClose=flushOnClose,
        )
        self.flush_period = flush_period
        self.last_flush = 0
        self.flushLevel = flushLevel
        self.target = target
        self.capacity = capacity
        self.flushOnClose = flushOnClose

    def flush(self):
        super().flush()
        self.last_flush = time.time()

    def shouldFlush(self, record):
        """
        Check for buffer full or a record at the flushLevel or higher.
        """
        return (
            (len(self.buffer) >= self.capacity)
            or (record.levelno >= self.flushLevel)
            or (time.time() - self.last_flush > self.flush_period)
        )


def config_logging(filename, level, format, datefmt, buffer_num_records, flush_period):
    level = getattr(logging, level, logging.DEBUG)

    file_handler = logging.FileHandler(filename=filename)
    mem_handler = PeriodicMemoryHandler(
        capacity=buffer_num_records,
        flushLevel=level,
        target=file_handler,
        flush_period=flush_period,
    )

    logging.basicConfig(
        level=level,
        format=format,
        datefmt=datefmt,
        handlers=[mem_handler],
    )


__all__ = ["FileLock", "LocalCopySender", "RemoteCopySender", "config_logging"]