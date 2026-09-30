#!/usr/bin/env python3
"""Serve a directory over HTTP with single-range request support."""

from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeRequestHandler(SimpleHTTPRequestHandler):
    range_pattern = re.compile(r"bytes=(\d*)-(\d*)$")

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return None

        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(source.fileno()).st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        status = 200

        if range_header:
            match = self.range_pattern.fullmatch(range_header.strip())
            if match is None or not match.group(1):
                source.close()
                self.send_error(416, "Invalid byte range")
                return None
            start = int(match.group(1))
            if match.group(2):
                end = min(int(match.group(2)), size - 1)
            if start >= size or start > end:
                source.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            status = 206

        self.send_response(status)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(os.path.getmtime(path)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        self._requested_range = (start, end)
        return source

    def copyfile(self, source, outputfile):
        start, end = self._requested_range
        source.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    handler = partial(RangeRequestHandler, directory=args.directory)
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Serving {args.directory} on {args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
