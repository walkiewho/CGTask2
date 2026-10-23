#!/bin/python3
import argparse
import cv2
import numpy as np
from pathlib import Path
import time
import os
from ASCIIConverter import ASCIIConverter

class ASCIIVideoConverter(ASCIIConverter):
    def __init__(self, video_path: Path, width=150, area_interpolation=False, target_fps=24):
        self._video_path = self._validate_path(video_path)
        self._cap = cv2.VideoCapture(str(self._video_path))

        if not self._cap.isOpened():
            raise IOError(f"Cannot onen file")

        self._char_width = width
        self._interpolation = cv2.INTER_AREA if area_interpolation else cv2.INTER_LINEAR

        ret, frame = self._cap.read()
        if not ret:
            raise IOError("Cannot read first frame")

        self._default_image = frame
        self._changeable_image = np.copy(frame)
        self._char_height = self._get_height()
        self._char_buffer = None

        self._video_buffers = []
        self._target_fps = target_fps

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def set_path(self, path: Path):
        self._video_path = self._validate_path(path)
        if hasattr(self, '_cap') and self._cap.isOpened():
            self._cap.release()

        self._cap = cv2.VideoCapture(str(self._video_path))
        if not self._cap.isOpened():
            raise IOError("Read error")

        ret, frame = self._cap.read()
        if not ret:
            raise IOError("Cannot read first frame")

        self._default_image = frame
        self._changeable_image = np.copy(frame)
        self._char_height = self._get_height()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def create_ascii(self) -> None:
        self._video_buffers = []
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("ASCII-generating started...")
        frame_count = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            self._changeable_image = frame
            self._proceed_image()
            self._resize_image()
            luma_list = self._to_luma_list()

            self._char_buffer = np.array(ASCIIConverter.CHAR_SET)[luma_list]
            self._video_buffers.append(self._char_buffer)

            frame_count += 1
            print(f"Frames proceed: {frame_count}", end='\r')

        print(f"\nGeneration ended up. Total frames: {frame_count}")

    def print_ascii(self) -> None:
        if not self._video_buffers:
            print('Video not proceed yet')
            return

        frame_delay = 1.0 / self._target_fps
        try:
            while True:
                for frame_buffer in self._video_buffers:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    string = '\n'.join(''.join(row) for row in frame_buffer)
                    print(string, flush=True)
                    time.sleep(frame_delay)
        except KeyboardInterrupt:
            pass

    def __del__(self):
        if self._cap.isOpened():
            self._cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("files/tux.mp4"))
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--interpolate", action="store_true")
    args = parser.parse_args()

    if args.path.exists():
        video_converter = ASCIIVideoConverter(args.path, width=args.width, area_interpolation=args.interpolate, target_fps=args.fps)
        video_converter.create_ascii()
        input("Generation ended...")
        video_converter.print_ascii()
    else:
        print(f"Файл {args.path} не найден.")