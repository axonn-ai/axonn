import torch
from collections import defaultdict
from collections import deque
import axonn


class Timers:
    def __init__(self):
        self.timers = defaultdict(list)
        self.curr_index = defaultdict(int)
        self.stack = deque()

    def start(self, key):
        if not axonn.axonn.enable_timers:
            return
        self.stack.append(key)
        key = tuple(self.stack)
        index = self.curr_index[key]
        timers = self.timers[key]
        assert index == len(timers) or index < len(timers)
        if index == len(timers):
            self.timers[key].append(
                [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            )
        self.timers[key][index][0].record()

    def stop(self, key):
        if not axonn.axonn.enable_timers:
            return
        key = tuple(self.stack)
        index = self.curr_index[key]
        self.timers[key][index][1].record()
        self.curr_index[key] += 1
        self.stack.pop()

    def get_times(self):
        torch.cuda.synchronize()
        total_times = defaultdict(float)
        total_events = defaultdict(int)
        for key in self.timers:
            for events in self.timers[key]:
                start_event, end_event = events
                total_times[key] += start_event.elapsed_time(end_event)
            total_events[key] = self.curr_index[key]
            self.curr_index[key] = 0
        return total_times, total_events

    def create_hatchet_literal_cct(self):
        def insert_node(key, time, root):
            if len(key) == 1:
                root[key] = {
                    "frame": {"name": key[0]},
                    "metrics": {"time (inc)": time},
                    "children": [],
                }
                return root[key[0]]
            else:
                top_node = key[0]
                child = insert_node(
                    key[1:],
                    time,
                    {
                        child["frame"]["name"]: child
                        for child in root[top_node]["children"]
                    },
                )
                if child not in root[top_node]["children"]:
                    root[top_node]["children"].append(child)
                return root[top_node]

        sorted_timers = dict(sorted(self.timers.items(), key=lambda item: len(item[0])))
        root = {}
        for key, time in sorted_timers.items():
            insert_node(key, time, root)
        hatchet_format = list(root.values())
        return hatchet_format
