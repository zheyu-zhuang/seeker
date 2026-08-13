from __future__ import annotations

import math

import av
import gym
import numpy as np


class VideoRecorder:
    def __init__(
        self,
        fps,
        codec,
        input_pix_fmt,
        **kwargs,
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()
    
    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0
    
    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                "crf": str(crf),
                "profile": profile,
            },
            **kwargs,
        )
        return obj

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time
    
    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")
        
        n_repeats = 1
        if self.start_time is not None:
            global_idx = math.floor((frame_time - self.start_time) * self.fps + 1e-5)
            n_repeats = max(0, global_idx - self.next_global_idx + 1)
            self.next_global_idx += n_repeats
        
        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h, w, _ = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for _ in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_recoder: VideoRecorder,
        mode="rgb_array",
        file_path=None,
        steps_per_render=1,
        **kwargs,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        return obs
    
    def step(self, action):
        result = super().step(action)
        self.step_count += 1
        if self.file_path is not None and self.step_count % self.steps_per_render == 0:
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            frame = self.env.render(mode=self.mode, **self.render_kwargs)
            assert frame.dtype == np.uint8
            self.video_recoder.write_frame(frame)
        return result
    
    def render(self, mode="rgb_array", **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path

    def set_prediction_video_boxes(self, boxes):
        """Forward rollout prediction boxes to the render env when available."""
        if hasattr(self.env, "set_prediction_video_boxes"):
            return self.env.set_prediction_video_boxes(boxes)
        return None
