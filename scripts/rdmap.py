"""Range-Doppler Map 실시간 시각화.

cfg: xwr68xx_AOP_profile_rdmap.cfg (guiMonitor rangeDopplerHeatMap=1)

축:
  X축 — Doppler velocity (m/s), 음수=접근, 양수=멀어짐
  Y축 — Range (m), 센서로부터의 거리
  색상 — 신호 세기 (log magnitude, 높을수록 밝음)
  빨간 X — CFAR로 감지된 포인트
"""

import argparse
import queue
import sys
import threading
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from arda.radar import IWR6843Sensor
from arda.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "config/profiles/xwr68xx_AOP_profile_rdmap.cfg"

NUM_RANGE_BINS = 256
NUM_DOPPLER_BINS = 16
RANGE_RES = 0.044   # m/bin
VEL_RES = 0.13      # m/s/bin
MAX_RANGE_DISPLAY = 3.0


def _doppler_axis() -> np.ndarray:
    return (np.arange(NUM_DOPPLER_BINS) - NUM_DOPPLER_BINS // 2) * VEL_RES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Range-Doppler Map 시각화")
    p.add_argument("--cli-port", default="/dev/ttyUSB0")
    p.add_argument("--data-port", default="/dev/ttyUSB1")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--max-range", type=float, default=MAX_RANGE_DISPLAY)
    p.add_argument("--clim", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    return p.parse_args()


# ── 센서 읽기 스레드 ────────────────────────────────────────────────────────────

def sensor_thread(cli_port: str, data_port: str, config: str,
                  frame_q: queue.Queue, stop_evt: threading.Event) -> None:
    try:
        sensor = IWR6843Sensor(cli_port, data_port)
        sensor.configure(config)
        with sensor:
            logger.info("센서 연결 완료, 프레임 수신 중")
            read_count = 0
            frame_count = 0
            while not stop_evt.is_set():
                frame = sensor.read_frame()
                read_count += 1

                # 처음 20번 read 결과를 터미널에 출력 (진단용)
                if read_count <= 20:
                    raw_bytes = sensor._data_serial.in_waiting if sensor._data_serial else 0
                    print(f"[DIAG] read#{read_count}  frame={'OK' if frame else 'None'}"
                          f"  in_waiting={raw_bytes}")

                if frame is not None:
                    frame_count += 1
                    if frame_count <= 3:
                        print(f"[DIAG] first frame #{frame['frame_number']}"
                              f"  keys={list(frame.keys())}"
                              f"  points={len(frame.get('points', []))}"
                              f"  has_heatmap={'rd_heatmap' in frame}")
                    if frame_q.qsize() > 2:
                        try:
                            frame_q.get_nowait()
                        except queue.Empty:
                            pass
                    frame_q.put(frame)
    except Exception as e:
        logger.error("센서 오류: %s", e)
        frame_q.put({"error": str(e)})


# ── 시각화 ──────────────────────────────────────────────────────────────────────

class RDMapPlotter:
    def __init__(self, max_range: float, clim: tuple | None):
        self._r_end = min(int(max_range / RANGE_RES) + 1, NUM_RANGE_BINS)
        self._clim = clim
        self._doppler_axis = _doppler_axis()
        self._range_max = (self._r_end - 1) * RANGE_RES

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.suptitle("Range-Doppler Map  —  IWR6843AOPEVM", fontsize=12)

        blank = np.zeros((self._r_end, NUM_DOPPLER_BINS))
        self._im = self.ax.imshow(
            blank,
            aspect="auto",
            origin="lower",
            cmap="inferno",
            extent=[self._doppler_axis[0], self._doppler_axis[-1], 0, self._range_max],
        )
        self.fig.colorbar(self._im, ax=self.ax, label="Magnitude")
        self.ax.set_xlabel("Doppler (m/s)      [← 접근 | 멀어짐 →]")
        self.ax.set_ylabel("Range (m)")
        self.ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)

        self._scatter = self.ax.scatter([], [], c="red", s=50, marker="x",
                                        linewidths=1.5, zorder=5, label="CFAR pts")
        self.ax.legend(loc="upper right", fontsize=8)
        self._status = self.fig.text(0.5, 0.01, "Waiting for sensor...",
                                     ha="center", fontsize=9, color="gray")
        plt.tight_layout(rect=[0, 0.04, 1, 1])

    def draw(self, frame: dict) -> None:
        if "error" in frame:
            msg = f"Sensor error: {frame['error']}"
            print(f"[ERROR] {msg}")
            self._status.set_text(msg)
            self.fig.canvas.draw()
            return

        rd_raw = frame.get("rd_heatmap")
        n = frame["frame_number"]

        if rd_raw is None:
            msg = f"Frame #{n}: no rd_heatmap — guiMonitor param6 must be 1"
            if n % 30 == 0:
                print(f"[WARN] {msg}")
            self._status.set_text(msg)
            self.fig.canvas.draw()
            return

        expected = NUM_RANGE_BINS * NUM_DOPPLER_BINS
        if len(rd_raw) != expected:
            msg = f"Frame #{n}: rd_heatmap size={len(rd_raw)}, expected={expected}"
            print(f"[WARN] {msg}")
            self._status.set_text(msg)
            self.fig.canvas.draw()
            return

        heatmap = np.fft.fftshift(
            rd_raw.reshape(NUM_RANGE_BINS, NUM_DOPPLER_BINS), axes=1
        )[: self._r_end, :]

        peak = heatmap.max()
        if n % 10 == 0:
            print(f"[DEBUG] Frame #{n}  min={heatmap.min():.1f}  max={peak:.1f}"
                  f"  mean={heatmap.mean():.1f}  pts={len(frame.get('points', []))}")

        vmin = heatmap.min() if self._clim is None else self._clim[0]
        vmax = peak          if self._clim is None else self._clim[1]
        if vmin == vmax:
            vmax = vmin + 1

        self._im.set_data(heatmap)
        self._im.set_clim(vmin, vmax)

        pts = frame.get("points", [])
        if pts:
            rs = [np.sqrt(p["x"]**2 + p["y"]**2 + p["z"]**2) for p in pts]
            ds = [p["doppler"] for p in pts]
            vis = [(r, d) for r, d in zip(rs, ds) if r <= self._range_max]
            if vis:
                self._scatter.set_offsets(np.c_[[d for _, d in vis], [r for r, _ in vis]])
            else:
                self._scatter.set_offsets(np.empty((0, 2)))
        else:
            self._scatter.set_offsets(np.empty((0, 2)))

        self._status.set_text(f"Frame #{n}  |  CFAR pts: {len(pts)}  |  peak: {peak:.0f}")
        self.fig.canvas.draw()


# ── 진입점 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    frame_q: queue.Queue = queue.Queue()
    stop_evt = threading.Event()

    # 센서는 백그라운드 스레드에서 — 창은 즉시 열림
    t = threading.Thread(
        target=sensor_thread,
        args=(args.cli_port, args.data_port, args.config, frame_q, stop_evt),
        daemon=True,
    )
    t.start()

    plotter = RDMapPlotter(
        max_range=args.max_range,
        clim=tuple(args.clim) if args.clim else None,
    )

    logger.info("Range-Doppler Map 시작 (창을 닫거나 Ctrl+C로 종료)")
    try:
        while plt.fignum_exists(plotter.fig.number):
            try:
                frame = frame_q.get(timeout=0.05)
                plotter.draw(frame)
            except queue.Empty:
                pass
            plt.pause(0.01)   # Tkinter 이벤트 루프 구동
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        plt.close("all")
        logger.info("종료")


if __name__ == "__main__":
    main()
