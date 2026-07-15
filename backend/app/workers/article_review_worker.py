from __future__ import annotations

import asyncio
import logging
import os
import signal
from uuid import uuid4

from ..services import article_review_service, contributor_store, product_master


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


async def run() -> None:
    product_master.ensure_geo_schema()
    worker_id = f"worker_{uuid4().hex[:12]}"
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            try:
                loop.add_signal_handler(getattr(signal, name), stop.set)
            except NotImplementedError:
                pass
    running: set[asyncio.Task] = set()
    logger.info("article review worker started: %s", worker_id)
    while not stop.is_set():
        running = {task for task in running if not task.done()}
        settings = contributor_store.get_review_settings()
        capacity = max(0, settings["effective_concurrency"] - len(running))
        if capacity:
            for job in article_review_service.claim_jobs(worker_id, capacity):
                task = asyncio.create_task(article_review_service.process_job(job))
                running.add(task)
        article_review_service.update_worker_heartbeat(worker_id, len(running))
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
    if running:
        await asyncio.gather(*running, return_exceptions=True)
    article_review_service.update_worker_heartbeat(worker_id, 0)


if __name__ == "__main__":
    asyncio.run(run())
