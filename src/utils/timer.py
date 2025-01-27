import time
from typing import Tuple, Optional


class Timer:
    def __init__(self):
        """
        A simple Timer class to measure execution time.
        """
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.laps = []

    def start(self) -> None:
        """Starts the timer."""
        self.start_time = time.time()
        self.laps = []
        print("Timer started.")

    def stop(self) -> Tuple[int, int, float]:
        """Stops the timer and returns the elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        self.end_time = time.time()
        elapsed_time = self._compute_elapsed_time(self.start_time, self.end_time)
        return elapsed_time

    def lap(self) -> Tuple[int, int, float]:
        """
        Records a lap time since the timer started.
        Returns the time since the last lap or start.
        """
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        lap_time = time.time()
        last_time = self.laps[-1] if self.laps else self.start_time
        elapsed = self._compute_elapsed_time(last_time, lap_time)
        self.laps.append(lap_time)
        print(f"Lap {len(self.laps)}: {self._format_time(*elapsed)}")
        return elapsed

    def reset(self) -> None:
        """Resets the timer to initial state."""
        self.start_time = None
        self.end_time = None
        self.laps = []
        print("Timer reset.")

    def elapsed(self) -> Tuple[int, int, float]:
        """Returns the elapsed time without stopping the timer."""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")
        end_time = time.time()
        return self._compute_elapsed_time(self.start_time, end_time)

    def format_elapsed_time(self) -> str:
        """Returns the elapsed time as a formatted string."""
        if self.start_time is None:
            return "Timer has not started."
        end_time = self.end_time if self.end_time else time.time()
        return self._format_time(*self._compute_elapsed_time(self.start_time, end_time))

    def print_elapsed_time(self) -> str:
        print(f"Processing completed in {self.format_elapsed_time()}")

    @staticmethod
    def _compute_elapsed_time(start: float, end: float) -> Tuple[int, int, float]:
        """Computes elapsed time in hours, minutes, and seconds."""
        elapsed_time = end - start
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        return hours, minutes, seconds

    @staticmethod
    def _format_time(hours: int, minutes: int, seconds: float) -> str:
        """Formats elapsed time into a readable string."""
        return f"{hours} hours, {minutes} minutes, and {seconds:.2f} seconds"
