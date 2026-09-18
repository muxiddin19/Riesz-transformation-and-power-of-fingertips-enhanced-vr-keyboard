import sys

# PATCH: on Windows consoles using a non-UTF-8 codepage (e.g. cp949 for
# Korean locales), any print() of the em-dashes ("--") used throughout
# camera_rgb.py/recorder.py's log and error messages raises
# UnicodeEncodeError and crashes the process -- including from inside
# camera retry/error-logging code, which can mask a real camera failure
# behind an unrelated encoding crash. Force UTF-8 stdout/stderr up front
# so logging can never take down the recorder.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from recorder import Recorder


def main():
    recorder = Recorder()
    recorder.run()


if __name__ == "__main__":
    main()