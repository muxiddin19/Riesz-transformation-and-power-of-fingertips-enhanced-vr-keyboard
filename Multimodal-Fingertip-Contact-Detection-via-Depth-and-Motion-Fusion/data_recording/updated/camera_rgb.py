"""
camera_rgb.py
==============
Thin wrapper around a plain RGB source: a USB webcam (int index), an
IP-camera / DroidCam-over-URL ("http://192.168.x.x:4747/video"), or a
DroidCam-over-USB virtual webcam identified by a DEVICE NAME substring
(e.g. "DroidCam").

Runs its own grab thread, same shape as camera_depth.py, and always
exposes the LATEST (frame, timestamp) pair.

Why this file exists / what it fixes
-------------------------------------
DirectShow + DroidCam on Windows is flaky in three specific ways this
wrapper works around:

  1. DirectShow indices SHIFT once the RealSense D405 is opened. So
     resolving "DroidCam" -> index must happen at start() time, not at
     config-write time or import time.
  2. cv2.VideoCapture(index, cv2.CAP_DSHOW).open() sometimes throws
     "raised unknown C++ exception!" on the FIRST attempt even when the
     device is fine — it's a known DirectShow/DroidCam race condition.
     Retrying 2-3 times with a short delay usually succeeds.
  3. If the DroidCam Client desktop app isn't running/connected yet,
     DSHOW can't see the virtual camera at all — no amount of retrying
     fixes that, so we give a clear, actionable error instead of a
     generic traceback.

PATCH: with 3+ devices (matching the 30/45/60/90-degree multi-angle
setup), a 4th failure mode showed up that this wrapper didn't handle:

  4. Multiple DroidCam devices all enumerate under DirectShow with
     similar/identical names (e.g. "DroidCam Source 3", "DroidCam Source
     4"). _resolve_name_to_index() used to return the FIRST device whose
     name contained the configured substring -- so every RGBCamera
     configured with source="DroidCam" silently resolved to the SAME
     physical phone, and any additional phones were never actually
     opened (their frames would just be duplicates of the first one).

     Fixed with a process-wide "claimed indices" set: each successful
     name resolution claims its index, and subsequent cameras (even with
     the same name substring) skip already-claimed indices and take the
     next matching one instead. This guarantees each configured camera
     gets a DISTINCT physical device.

     Caveat this does NOT solve: if your phones are truly
     indistinguishable by name (identical DroidCam Client version, no
     custom device name), this fix guarantees "each gets a different
     index," not "config entry N always maps to physical phone N" across
     runs -- Windows' DirectShow enumeration order for identically-named
     devices isn't guaranteed stable across reboots/reconnects. If you
     need guaranteed per-phone identity, either give each DroidCam
     Client instance a distinct name (check its settings), or pin down
     indices manually with list_camera_names.py at the start of each
     session and pass explicit int sources in config.py instead of the
     "DroidCam" substring.
"""

import threading
import time

import cv2


class RGBCamera:
    # PATCH: shared across ALL RGBCamera instances in this process, so
    # name-substring resolution never double-assigns the same physical
    # device to two different camera configs. Reset with
    # RGBCamera.reset_claimed_indices() if you tear down and restart a
    # recording session's cameras without restarting the process.
    _claimed_indices = set()

    def __init__(self, name: str, source, width: int = 640, height: int = 480,
                 fps: int = 30, backend: str = "auto", fourcc: str = None):
        self.name = name
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        # PATCH: per-camera FourCC override for USB bandwidth reduction.
        # Not applied globally -- WCAM100 (cam1) produces corrupted/solid-
        # color frames when forced to MJPEG, so this is opt-in per camera
        # via config.py rather than a blanket default.
        self.fourcc = fourcc

        self.cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest = None  # (frame, timestamp)

    @classmethod
    def reset_claimed_indices(cls):
        """Call this if you stop() all cameras and start() a fresh set
        within the same process (e.g. between sessions) and want name
        resolution to be free to reuse indices from scratch."""
        cls._claimed_indices = set()

    def _safe_set(self, prop, value, prop_name, attempts=3, delay_sec=0.2):
        """cap.set() can throw a raw cv2/COM exception (not just return
        False) when the DirectShow graph isn't fully settled yet,
        particularly under multi-camera USB contention. Never let a
        property-set glitch take down the whole camera/session -- retry a
        few times, and if it still won't take, log it and carry on with
        whatever the device's current setting is rather than crashing."""
        for attempt in range(1, attempts + 1):
            try:
                self.cap.set(prop, value)
                return True
            except cv2.error as e:
                if attempt == attempts:
                    print(f"[{self.name}] could not set {prop_name}={value} "
                          f"after {attempts} attempts ({e}) -- continuing with "
                          f"the device's current setting")
                    return False
                time.sleep(delay_sec)
        return False

    # ------------------------------------------------------------
    def start(self):
        self.cap = self._open_with_retries()
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                f"[{self.name}] cannot open source: {self.source} "
                f"(backend={self.backend})"
            )

        # PATCH: DirectShow's graph is sometimes still settling right after
        # open() reports success, especially under multi-camera USB
        # contention -- calling .set() immediately can throw a raw,
        # uncatchable-looking "Unknown C++ exception" from the underlying
        # COM layer. A brief pause here lets the graph stabilize before we
        # touch any properties.
        time.sleep(0.2)

        if self.fourcc:
            self._safe_set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc), "fourcc")
        self._safe_set(cv2.CAP_PROP_FRAME_WIDTH, self.width, "width")
        self._safe_set(cv2.CAP_PROP_FRAME_HEIGHT, self.height, "height")
        self._safe_set(cv2.CAP_PROP_FPS, self.fps, "fps")

        # confirm we can actually pull a frame, not just that .open()
        # returned true (DSHOW/MSMF sometimes report opened=True but then
        # never yield a frame). Retry a few times with a short delay
        # first -- some drivers need a brief warm-up after opening before
        # the first real frame is available, and a single immediate read
        # can fail even though the camera is genuinely fine.
        # PATCH: some UVC devices (e.g. the "USB2.0 UVC PC Camera" / cam2)
        # intermittently need noticeably longer than ~1.5s to settle after
        # start()'s property-set calls before the very first read()
        # succeeds -- this isn't a consistent failure, it comes and goes
        # run to run, so widen the retry budget rather than chase a single
        # deterministic cause.
        ok = False
        for attempt in range(10):
            ok, _ = self.cap.read()
            if ok:
                break
            time.sleep(0.3)
        if not ok:
            self.cap.release()
            raise RuntimeError(
                f"[{self.name}] source opened but produced no frame: "
                f"{self.source} (backend={self.backend})"
            )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[{self.name}] started ({self.width}x{self.height} @ {self.fps}fps)")

    # ------------------------------------------------------------
    def _open_with_retries(self, attempts: int = 4, delay_sec: float = 1.0):
        resolved_source = self._resolve_source()

        backends_to_try = self._backend_candidates()

        last_exc = None
        for backend_name, backend_flag in backends_to_try:
            for attempt in range(1, attempts + 1):
                try:
                    if backend_flag is None:
                        cap = cv2.VideoCapture(resolved_source)
                    else:
                        cap = cv2.VideoCapture(resolved_source, backend_flag)
                except Exception as e:  # the "unknown C++ exception" case
                    last_exc = e
                    print(f"[{self.name}] open attempt {attempt}/{attempts} "
                          f"via {backend_name} raised: {e} — retrying...")
                    time.sleep(delay_sec)
                    continue

                if cap.isOpened():
                    print(f"[{self.name}] opened via {backend_name} "
                          f"(attempt {attempt}, source={resolved_source})")
                    return cap

                cap.release()
                print(f"[{self.name}] open attempt {attempt}/{attempts} "
                      f"via {backend_name} did not open — retrying...")
                time.sleep(delay_sec)

        if last_exc is not None:
            print(f"[{self.name}] all backends failed, last exception: {last_exc}")
        return None

    def _backend_candidates(self):
        """Order of backends to try. String URL sources (IP cam / DroidCam
        over wifi) should NOT be forced through DSHOW."""
        if isinstance(self.source, str) and self.source.lower().startswith("http"):
            return [("ANY", None), ("FFMPEG", cv2.CAP_FFMPEG)]

        if self.backend == "dshow":
            return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", None)]
        if self.backend == "msmf":
            return [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", None)]
        return [("ANY", None), ("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]

    def _resolve_source(self):
        """If self.source is a plain int or a URL string, use as-is.
        If it's a name-like string (e.g. "DroidCam"), look up the CURRENT
        DirectShow index for a device whose name contains it."""
        if isinstance(self.source, int):
            return self.source

        if isinstance(self.source, str) and self.source.lower().startswith("http"):
            return self.source  # URL — pass straight through

        # name substring -> resolve to a live index right now
        index = self._resolve_name_to_index(self.source)
        if index is None:
            raise RuntimeError(
                f"[{self.name}] could not find an UNCLAIMED DirectShow "
                f"device matching '{self.source}'. Run list_camera_names.py "
                f"to see what's currently enumerated -- if '{self.source}' "
                f"isn't listed at all, the physical device itself isn't "
                f"connected/powered right now (check cable/port; for a "
                f"phone camera, make sure its DroidCam Client app is open "
                f"and connected). If it IS listed, another camera config "
                f"already claimed every matching device this run (see "
                f"RGBCamera._claimed_indices)."
            )
        print(f"[{self.name}] resolved device name '{self.source}' -> index {index}")
        return index

    def _resolve_name_to_index(self, name_substring):
        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError:
            print(f"[{self.name}] pygrabber not installed, cannot resolve "
                  f"'{name_substring}' by name — falling back to index 0. "
                  f"Run: pip install pygrabber")
            return 0

        graph = FilterGraph()
        devices = graph.get_input_devices()

        # PATCH: skip indices already claimed by another RGBCamera in this
        # process, so N cameras configured with the same name substring
        # (e.g. several phones all named "DroidCam Source N") each get a
        # DISTINCT physical device instead of all opening the first match.
        for i, dev_name in enumerate(devices):
            if i in RGBCamera._claimed_indices:
                continue
            if name_substring.lower() in dev_name.lower():
                RGBCamera._claimed_indices.add(i)
                print(f"[{self.name}] claimed index {i} ('{dev_name}'); "
                      f"already-claimed indices this run: "
                      f"{sorted(RGBCamera._claimed_indices)}")
                return i
        return None

    # ------------------------------------------------------------
    def _capture_loop(self):
        consecutive_failures = 0
        warned_at_failure_count = 0
        last_frame_sample = None
        stale_count = 0
        stale_limit = self.fps * 2  # ~2s of an unchanged frame means the stream stalled

        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures in (5, 20, 100) and consecutive_failures != warned_at_failure_count:
                    warned_at_failure_count = consecutive_failures
                    print(f"[{self.name}] {consecutive_failures} consecutive "
                          f"read failures — camera may have disconnected.")

                # PATCH: past a sustained-failure point, retrying read() on
                # the same handle forever was a dead end -- some backends
                # (MSMF in particular) drop the stream mid-session and
                # never recover on their own, leaving the last-good frame
                # frozen in the preview indefinitely until the process is
                # killed by hand. Force an actual reconnect instead, and
                # keep retrying every 100 failures if it doesn't take.
                if consecutive_failures % 100 == 0:
                    print(f"[{self.name}] reconnecting after sustained read failures...")
                    old_cap = self.cap
                    new_cap = self._open_with_retries()
                    if new_cap is not None and new_cap.isOpened():
                        old_cap.release()
                        self.cap = new_cap
                    else:
                        # keep the old (dead) handle rather than leaving
                        # self.cap as None -- its .read() calls stay safe
                        # (just keep failing) until the next retry point
                        print(f"[{self.name}] reconnect attempt failed, will keep retrying")

                time.sleep(min(1.0, 0.1 * consecutive_failures))
                continue

            consecutive_failures = 0
            warned_at_failure_count = 0

            # PATCH: some MSMF-backed devices keep returning ok=True with
            # the SAME last frame forever once the stream stalls, instead
            # of ever failing outright -- invisible to the check above.
            # Catch it by fingerprinting a cheap downsample of each frame;
            # if it hasn't changed for ~2s worth of reads, force a
            # reconnect instead of silently recording a frozen feed.
            sample = frame[::20, ::20].tobytes()
            if sample == last_frame_sample:
                stale_count += 1
                if stale_count >= stale_limit:
                    print(f"[{self.name}] frame unchanged for {stale_count} reads — "
                          f"stream appears stalled, reconnecting...")
                    self.cap.release()
                    self.cap = self._open_with_retries()
                    stale_count = 0
                    last_frame_sample = None
                    if self.cap is None or not self.cap.isOpened():
                        print(f"[{self.name}] reconnect attempt failed, will keep retrying")
                        time.sleep(1.0)
                    continue
            else:
                stale_count = 0
            last_frame_sample = sample

            ts = time.time()
            with self._lock:
                self._latest = (frame, ts)

    def get_latest(self):
        """Returns (frame, timestamp) or None if nothing yet."""
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        print(f"[{self.name}] stopped")