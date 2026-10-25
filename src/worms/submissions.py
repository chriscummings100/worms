"""Bound transient command resources while recording ahead of the GPU."""

from collections import deque
from time import perf_counter


def chunk_timings():
    """Intermediate submission costs; final submission remains caller-owned."""
    return {
        "chunks": 0,
        "submissions": 0,
        "waits": 0,
        "cpu_finish_ms": 0.0,
        "cpu_submit_ms": 0.0,
        "wait_ms": 0.0,
    }


class BoundedSubmissions:
    """Retain command buffers until their same-queue submission has completed.

    A borrowed final encoder is submitted by the caller. A window refresh can
    reserve a slot for the previous SDK-owned final submission; the first new
    submission's fence also covers it because both use the graphics queue.
    SlangPy continues to own and bound swapchain presentation.
    """

    def __init__(self, device):
        self.device = device
        self.pending = deque()
        self._previous_final = False

    def _retire(self):
        _, _, covers_previous = self.pending.popleft()
        if covers_previous:
            self._previous_final = False

    def _wait_oldest(self, stats, measure):
        started = perf_counter() if measure else 0.0
        self.device.wait_for_submit(self.pending[0][0])
        if measure:
            stats["wait_ms"] += (perf_counter() - started) * 1000
        stats["waits"] += 1
        self._retire()

    def retire_finished(self):
        while self.pending and self.device.is_submit_finished(self.pending[0][0]):
            self._retire()

    def begin_frame(self, stats, *, previous_final=False, measure=False):
        # A prior frame's last simulation chunk precedes its SDK-owned final
        # draw. Retire our references before charging one slot for that draw.
        self.retire_finished()
        while self.pending:
            self._wait_oldest(stats, measure)
        self._previous_final = previous_final

    def make_room(self, max_in_flight, stats, *, measure=False):
        """Reserve one submission slot, after recording the next chunk."""
        self.retire_finished()
        while len(self.pending) + int(self._previous_final) >= max_in_flight:
            self._wait_oldest(stats, measure)

    def submit(self, encoder, max_in_flight, stats, *, measure=False):
        started = perf_counter() if measure else 0.0
        command_buffer = encoder.finish()
        if measure:
            stats["cpu_finish_ms"] += (perf_counter() - started) * 1000
        self.make_room(max_in_flight, stats, measure=measure)
        started = perf_counter() if measure else 0.0
        submission = self.device.submit_command_buffer(command_buffer)
        if measure:
            stats["cpu_submit_ms"] += (perf_counter() - started) * 1000
        self.pending.append((submission, command_buffer, self._previous_final))
        stats["submissions"] += 1
        return submission
