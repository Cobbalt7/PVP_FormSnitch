import time
import queue
import threading

class SyncManagerThread(threading.Thread):
    def __init__(self, input_q1, input_q2, output_q1, output_q2, running_event, max_tolerance=0.005):
        super().__init__()
        self.input_q1 = input_q1
        self.input_q2 = input_q2
        self.output_q1 = output_q1
        self.output_q2 = output_q2
        self.running_event = running_event
        self.max_tolerance = max_tolerance
        self.daemon=True

    def run(self):
        while self.running_event.is_set():
            # Wait until both source buffers have data to compare
            if not self.input_q1.queue or not self.input_q2.queue:
                time.sleep(0.001)
                continue

            try:
                # Peek at the oldest items in both queues without extracting them yet
                item1 = self.input_q1.queue[0]
                item2 = self.input_q2.queue[0]
                
                t1 = item1["timestamp"]
                t2 = item2["timestamp"]
                time_delta = t1 - t2

                # --- CASE 1: MATCH FOUND ---
                if abs(time_delta) <= self.max_tolerance:
                    # Safely remove matching items from input queues
                    matched_item1 = self.input_q1.get()["data"]
                    matched_item2 = self.input_q2.get()["data"]
                    
                    # Dispatch to output queues
                    self._push_to_output(self.output_q1, matched_item1)
                    self._push_to_output(self.output_q2, matched_item2)

                # --- CASE 2: ITEM 1 IS STALE ---
                elif time_delta < -self.max_tolerance:
                    # Drop the older item from input 1 to catch up with input 2
                    self.input_q1.get()

                # --- CASE 3: ITEM 2 IS STALE ---
                else:
                    # Drop the older item from input 2 to catch up with input 1
                    self.input_q2.get()

            except (IndexError, queue.Empty):
                # Handle race conditions gracefully if items are pulled elsewhere
                time.sleep(0.001)

    def _push_to_output(self, dest_queue, payload):
        """
        Helper method to insert payloads into bounded output queues. 
        If the destination queue is full, it clears the oldest item to prevent lag.
        """
        if dest_queue.full():
            try:
                dest_queue.get_nowait()
            except queue.Empty:
                pass
        dest_queue.put(payload)
