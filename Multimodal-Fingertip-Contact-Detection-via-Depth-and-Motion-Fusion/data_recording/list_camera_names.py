
try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    print("pygrabber not installed. Run: pip install pygrabber")
    raise SystemExit(1)

graph = FilterGraph()
devices = graph.get_input_devices()

if not devices:
    print("No DirectShow video devices found.")
else:
    print("DirectShow devices, in index order (this order should match")
    print("cv2.VideoCapture(i, cv2.CAP_DSHOW)):\n")
    for i, name in enumerate(devices):
        print(f"  {i}: {name}")