from src.data_recorder import DataRecorder
from config.config import Config
import time

Config.create_output_directories()

class MockLandmark:
    def __init__(self, x, y, z, v):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = v

class MockLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark(i*0.1, i*0.1, i*0.1, 0.9) for i in range(33)]

recorder = DataRecorder("test_session")
recorder.start_recording()

for i in range(10):
    recorder.record_frame(MockLandmarks())
    time.sleep(0.1)

recorder.save_session(fps=10.0)
print("\nData recorder test passed")
