"""
Check system specifications for performance reporting
"""
import platform
import psutil
import cv2
import sys

if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

def check_system_specs():
    """Display system specifications"""
    
    print("\n" + "="*60)
    print("SYSTEM SPECIFICATIONS")
    print("="*60 + "\n")
    
    print("Operating System:")
    print(f"   System: {platform.system()}")
    print(f"   Release: {platform.release()}")
    print(f"   Version: {platform.version()}")
    print(f"   Machine: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")
    
    print("\nCPU:")
    print(f"   Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"   Logical cores: {psutil.cpu_count(logical=True)}")
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        print(f"   CPU frequency: {cpu_freq.current:.2f} MHz")
    else:
        print(f"   CPU frequency: N/A")
    
    memory = psutil.virtual_memory()
    print("\nMemory:")
    print(f"   Total: {memory.total / (1024**3):.2f} GB")
    print(f"   Available: {memory.available / (1024**3):.2f} GB")
    print(f"   Used: {memory.percent}%")
    
    print("\nCamera:")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"   Resolution: {int(width)}x{int(height)}")
        print(f"   FPS: {fps}")
        cap.release()
    else:
        print("   Camera not accessible")
    
    print("\nPython:")
    print(f"   Version: {platform.python_version()}")
    print(f"   Implementation: {platform.python_implementation()}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    check_system_specs()
