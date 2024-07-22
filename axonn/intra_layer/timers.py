import torch
from collections import defaultdict

class Timers():
    def __init__(self):
        self.timers = defaultdict(list)
        self.curr_index = defaultdict(int)

    def start(self, key):
        index = self.curr_index[key]
        timers = self.timers[key]
        assert index == len(timers) or index < len(timers)
        if index == len(timers):
            self.timers[key].append([torch.cuda.Event(enable_timing=True) for _ in range(2)])
        self.timers[key][index][0].record()


    def stop(self, key):
        index = self.curr_index[key]
        self.timers[key][index][1].record()
        self.curr_index[key] += 1

    def get_times(self):
        torch.cuda.synchronize()
        total_times = defaultdict(float)
        total_events = defaultdict(int)
        for key in self.timers:
            for events in self.timers[key]:
                start_event, end_event = events
                total_times[key] += start_event.elapsed_time(end_event) / 1000
                total_events[key] += 1
            self.curr_index[key] = 0
        return total_times, total_events

