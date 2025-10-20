import time
from concurrent.futures import Future
import queue
import threading
import time
import torch
import numpy as np
import logging
import os
import asyncio

logger = logging.getLogger("root_logger")

class RequestBatcher(threading.Thread):
    def __init__(
        self,
        batch_fn,
        max_batch_size=20,
    ):
        threading.Thread.__init__(self)
        self.global_queue = queue.Queue()
        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size

    def run(self,*args,**kwargs):
        while True:
            try:
                data = []
                while len(data) < self.max_batch_size:
                    try:
                        q_res = self.global_queue.get(len(data) == 0)
                        data.append(q_res)
                    except queue.Empty:
                        break

                batch_res = self.batch_fn([d["data"] for d, _ in data])

                for i, o in enumerate(batch_res):
                    _, fut = data[i]

                    if not fut.cancelled():
                        fut.set_result(batch_res[i])

            except:
                logger.exception("Error synthesizing data")
                for i, o in enumerate(data):
                    _, fut = data[i]

                    try:
                        fut.set_result(None)
                    except:
                        logger.exception(f"Error setting result for request")


                continue


    def append_queue(self, data):
        fut = Future()
        self.global_queue.put((data, fut))

        return fut

    async def infer(self, data):
        fut = self.append_queue({
            "time": time.time(),
            "data": data,
        })

        return await asyncio.wrap_future(fut)