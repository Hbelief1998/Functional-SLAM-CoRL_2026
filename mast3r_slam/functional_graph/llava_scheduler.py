from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional


class LlavaScheduler:
    def __init__(self, worker: Optional[Callable[[Dict[str, Any]], Any]] = None, max_workers: int = 1) -> None:
        self.worker = worker
        self.executor = ThreadPoolExecutor(max_workers=max_workers) if worker is not None else None
        self.submitted_tasks: list[dict] = []
        self._submitted_keys: set[str] = set()
        self._futures: dict[str, Future] = {}

    def submit(self, task_type: str, dedupe_key: str, payload: Dict[str, Any]) -> Optional[Future]:
        if dedupe_key in self._submitted_keys:
            return self._futures.get(dedupe_key)
        task = {"task_type": task_type, "dedupe_key": dedupe_key, **payload}
        self.submitted_tasks.append(task)
        self._submitted_keys.add(dedupe_key)
        if self.executor is None or self.worker is None:
            return None
        future = self.executor.submit(self.worker, task)
        self._futures[dedupe_key] = future
        return future

    def submit_local(self, child_node_id: str, payload: Dict[str, Any]) -> Optional[Future]:
        return self.submit("local", f"local:{child_node_id}", payload)

    def submit_remote(self, relation_key: str, payload: Dict[str, Any]) -> Optional[Future]:
        return self.submit("remote", f"remote:{relation_key}", payload)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False)
