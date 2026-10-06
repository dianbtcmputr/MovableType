import numpy as np, os, pyworld as pw, re, soundfile as sf
from io import BytesIO
from json import dumps, loads
from threading import Thread
from librosa import istft, load as librosa_load, resample, stft
from librosa.effects import pitch_shift, time_stretch
from pyttsx4 import init
from shutil import which
from subprocess import Popen
from tempfile import mkstemp
from time import monotonic, sleep
from ttkbootstrap.constants import *
from warnings import filterwarnings
from weakref import ref as weakref_ref
from zipfile import ZIP_DEFLATED, ZipFile
from time import strftime
from tkinter import Canvas, Listbox, Scale, filedialog, messagebox
from ttkbootstrap import (
    Combobox, Menu, Menubutton, BooleanVar, Button,
    Checkbutton, DoubleVar, Entry, Frame, Label,
    Labelframe, Notebook, Scrollbar, Separator, StringVar,
    Text, Toplevel, Window)

if 0 != 0: # 仅用于注释
    from main import Syllable, App

try:
    import sounddevice as sd
    HAS_SD = True
except Exception:
    HAS_SD = False

FONT = "Microsoft YaHei" if os.name == "nt" else "Helvetica"
PROJECT_EXT = ".mtproj"
PROJECT_META_ENTRY = "project.json"

_play_proc = None
_play_token = None
_play_callback = None

def safe_filename(s, fallback="joined", maxlen=60):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(s)).strip()
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = fallback
    return s[:maxlen]

class AsyncPlayer:
    @staticmethod
    def _dispatch_callback(cb):
        if cb is None:
            return
        try:
            import tkinter as _tk
            root = _tk._default_root
        except Exception:
            root = None
        if root is not None:
            try:
                root.after(0, cb)
                return
            except Exception:
                pass
        try:
            cb()
        except Exception:
            pass

    @staticmethod
    def _finish_playback(token):
        """自然播放结束时由 watcher 调用；与 stop_playback 互斥。"""
        global _play_token, _play_callback
        if _play_token is not token:
            return # 已被 stop 或已被新的播放替换
        _play_token = None
        cb, _play_callback = _play_callback, None
        if cb is not None:
            AsyncPlayer._dispatch_callback(cb)

    @staticmethod
    def _watch_sd(token):
        try:
            sd.wait()
        except Exception:
            pass
        AsyncPlayer._finish_playback(token)

    @staticmethod
    def _watch_proc(proc, token):
        while proc.poll() is None:
            if _play_token is not token:
                return
            sleep(0.05)
        AsyncPlayer._finish_playback(token)

    @staticmethod
    def _play_winsound_blocking(path, token):
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        except Exception:
            return
        AsyncPlayer._finish_playback(token)

    @staticmethod
    def stop_playback(trigger_after: bool = True):
        """停止所有正在播放的声音。

        trigger_after=True：
            若当前注册了 after 回调，会调用它（无论本次播放是自然结束还是被中止）。
        trigger_after=False：
            丢弃回调。play_array 内部重新开始播放时用这个，避免误触发旧回调。
        """
        global _play_proc, _play_token, _play_callback

        _play_token = None
    
        if HAS_SD:
            try:
                sd.stop()
            except Exception:
                pass
        if _play_proc is not None:
            try:
                _play_proc.terminate()
            except Exception:
                pass
            _play_proc = None
        if os.name == "nt":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

        cb, _play_callback = _play_callback, None
        if trigger_after and cb is not None:
            AsyncPlayer._dispatch_callback(cb)

    @staticmethod
    def play_array(y, sr, after=None):
        """非阻塞播放"""
        global _play_proc, _play_token, _play_callback

        AsyncPlayer.stop_playback(trigger_after=False)

        y = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
        if y.size == 0:
            AsyncPlayer._dispatch_callback(after)
            return
        y = np.clip(y, -1.0, 1.0)

        token = object()
        _play_token = token
        _play_callback = after

        if HAS_SD:
            try:
                sd.play(y, int(sr))
                if after is not None:
                    Thread(target=AsyncPlayer._watch_sd, args=(token, ), daemon=True).start()
                return
            except Exception:
                pass

        fd, path = mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, y, int(sr))

        if os.name == "nt":
            import winsound
            if after is not None:
                Thread(target=AsyncPlayer._play_winsound_blocking,
                       args=(path, token),
                       daemon=True).start()
            else:
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            for cmd in (["aplay", "-q", path],
                        ["afplay", path],
                        ["ffplay", "-nodisp", "-autoexit",
                         "-loglevel", "quiet", path]):
                if which(cmd[0]):
                    _play_proc = Popen(cmd)
                    if after is not None:
                        Thread(target=AsyncPlayer._watch_proc,
                               args=(_play_proc, token),
                               daemon=True).start()
                    break

class AudioUtil:
    @staticmethod
    def load_audio(path):
        try:
            y, sr = librosa_load(path, sr=None, mono=True)
            return np.asarray(y, dtype=np.float32), int(sr)
        except Exception:
            pass

        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        seg = seg.set_channels(1).set_sample_width(2)
        sr = seg.frame_rate
        a = np.array(seg.get_array_of_samples()).astype(np.float32) / 32768.0
        return a, int(sr)

    @staticmethod
    def decode_audio_bytes(data):
        """把 wav/flac 字节流解码成 (mono float32, sr)。"""
        y, sr = sf.read(BytesIO(data), dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1).astype(np.float32)
        y = np.ascontiguousarray(y)
        return y, int(sr)

    @staticmethod
    def db_to_gain(db):
        return float(10.0 ** (float(db) / 20.0))

    @staticmethod
    def _moving_avg_freq(x, w):
        """沿「频率轴」(axis=0) 做长度 w 的移动平均，边界用 edge 填充。"""
        nb, nt = x.shape
        pad = w // 2
        xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
        cs = np.cumsum(xp, axis=0)
        cs = np.concatenate([np.zeros((1, nt), dtype=cs.dtype), cs], axis=0)
        return (cs[w:w + nb] - cs[:nb]) / float(w)

    @staticmethod
    def _spectral_envelope(mag, sr, n_fft, smooth_hz=400.0):
        logm = np.log(mag + 1e-7)
        bin_hz = float(sr) / float(n_fft)
        w = int(round(smooth_hz / max(bin_hz, 1e-6)))
        w = max(3, w | 1)
        return np.exp(AudioUtil._moving_avg_freq(logm, w))

    @staticmethod
    def pitch_shift_keep_formant(y, sr, n_steps, keep_formant=True, strength=1.0):
        y = np.asarray(y, dtype=np.float32).ravel()
        if y.size == 0 or abs(n_steps) < 1e-6:
            return y.copy()

        y_shift = pitch_shift(y=y, sr=int(sr), n_steps=float(n_steps))
        y_shift = np.asarray(y_shift, dtype=np.float32).ravel()

        if not keep_formant or strength <= 0.0:
            return y_shift

        n = int(min(y.size, y_shift.size))
        if n < 256:
            return y_shift

        a = np.ascontiguousarray(y[:n], dtype=np.float32)
        b = np.ascontiguousarray(y_shift[:n], dtype=np.float32)

        n_fft = int(2 ** round(np.log2(max(256.0, float(sr) * 0.032))))
        n_fft = int(np.clip(n_fft, 256, 2048))
        hop = max(1, n_fft // 4)

        try:
            Da = stft(a, n_fft=n_fft, hop_length=hop, window="hann")
            Db = stft(b, n_fft=n_fft, hop_length=hop, window="hann")
        except Exception:
            return y_shift

        T = int(min(Da.shape[1], Db.shape[1]))
        if T < 2:
            return y_shift
        Da = Da[:, :T]
        Db = Db[:, :T]

        Ea = AudioUtil._spectral_envelope(np.abs(Da), sr, n_fft, smooth_hz=400.0)
        Eb = AudioUtil._spectral_envelope(np.abs(Db), sr, n_fft, smooth_hz=400.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            g = (Ea + 1e-6) / (Eb + 1e-6)
        g = np.nan_to_num(g, nan=1.0, posinf=1.0, neginf=1.0)

        if strength < 1.0:
            g = np.power(np.maximum(g, 1e-3), max(1e-3, float(strength)))
        g = np.clip(g, 0.25, 4.0)

        try:
            out = istft(Db * g, hop_length=hop, length=n)
        except Exception:
            return y_shift

        out = np.asarray(out, dtype=np.float32)

        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:
            out = out / peak * 0.99
        return out

    @staticmethod
    def flatten_tone(y, sr, target_f0=None, smooth_frames=3):
        y = y.astype(np.float64)

        f0, sp, ap = pw.wav2world(y, sr)

        voiced = f0 > 0

        if not np.any(voiced):
            return y.astype(np.float32)

        if target_f0 is None:
            target_f0 = np.median(f0[voiced])

        f0_new = f0.copy()
        f0_new[voiced] = target_f0

        diff = np.diff(voiced.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if voiced[0]:
            starts = np.r_[0, starts]
        if voiced[-1]:
            ends = np.r_[ends, len(voiced)]

        for s, e in zip(starts, ends):
            ramp_len = min(smooth_frames, (e - s) // 2)
            if ramp_len >= 2:
                f0_new[s:s+ramp_len] = np.linspace(0, target_f0, ramp_len)
                f0_new[e-ramp_len:e] = np.linspace(target_f0, 0, ramp_len)

        y_new = pw.synthesize(f0_new, sp, ap, sr)

        if len(y_new) > len(y):
            y_new = y_new[:len(y)]
        elif len(y_new) < len(y):
            y_new = np.pad(y_new, (0, len(y) - len(y_new)))

        return y_new.astype(np.float32)

class Micorphone:
    @staticmethod
    def list_input_devices() -> list[dict]:
        out: list[dict] = []
        try:
            devices = sd.query_devices()
        except Exception:
            return out
        for i, d in enumerate(devices):
            try:
                ch = int(d.get("max_input_channels", 0))
            except Exception:
                ch = 0
            if ch <= 0:
                continue
            out.append({
                "index": i,
                "name": str(d.get("name", f"设备 {i}")),
                "channels": ch,
                "default_sr": float(d.get("default_samplerate", 44100.0) or 44100.0),
            })
        return out

    @staticmethod
    def default_input_device() -> int | None:
        try:
            dev = sd.default.device
            idx = dev[0] if isinstance(dev, (tuple, list)) else dev
            if idx is None or int(idx) < 0:
                return None
            return int(idx)
        except Exception:
            return None

    @staticmethod
    def device_choices() -> list[tuple[int, str]]:
        """默认设备会附带(默认)后缀"""
        def_idx = Micorphone.default_input_device()
        out: list[tuple[int, str]] = []
        for d in Micorphone.list_input_devices():
            label = d["name"]
            if def_idx is not None and d["index"] == def_idx:
                label += "  (默认)"
            out.append((d["index"], label))
        return out

    @staticmethod
    def open_stream(sr: int, device, channels: int, callback):
        try:
            return sd.InputStream(
                samplerate=int(sr), channels=int(channels), dtype="float32",
                device=device, callback=callback, blocksize=0,
            )
        except Exception:
            try:
                if device is not None:
                    info = sd.query_devices(device, "input")
                else:
                    info = sd.query_devices(kind="input")
                fallback = int(info.get("default_samplerate", 44100) or 44100)
            except Exception:
                fallback = 44100
            return sd.InputStream(
                samplerate=fallback, channels=int(channels), dtype="float32",
                device=device, callback=callback, blocksize=0,
            )

class RecordDialog(Toplevel):
    WINDOW_SEC = 3.0          # 半屏显示的秒数
    REFRESH_MS = 50           # 波形刷新周期
    REFERENCE_PEAK = 1.0

    def __init__(self, master, sr: int = 44100):
        super().__init__(title="录制音频")
        self.transient(master)
        self.resizable(True, True)
        self.minsize(660, 400)
        self.focus_set()

        self.sr = int(sr)
        self.result = None

        self._frames: list[np.ndarray] = []     # 当前录音流累积的帧
        self._committed: list[np.ndarray] = []  # 已提交（暂停时合并）的音频块
        self._stream = None
        self._recording = False
        self._t0: float | None = None
        self._elapsed_accum = 0.0               # 之前各段的累计时长
        self._refresh_id = None
        self._device_index: int | None = None

        self._build_ui()
        self._refresh_devices(True)
        self._update_buttons()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._center_on_parent()

    def _build_ui(self):
        top = Frame(self, padding=(12, 12, 12, 6))
        top.pack(fill=X)

        Label(top, text="麦克风").pack(side=LEFT)
        self.device_var = StringVar()
        self.device_combo = Combobox(
            top, textvariable=self.device_var, state="readonly",
            width=42, bootstyle="info"
        )
        self.device_combo.pack(side=LEFT, padx=(8, 6))
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_selected)

        Button(top, text="↻ 刷新", bootstyle=(SECONDARY, OUTLINE), width=5,
               command=self._refresh_devices).pack(side=LEFT)

        wrap = Frame(self, padding=(12, 0, 12, 6))
        wrap.pack(fill=BOTH, expand=YES)

        self.canvas = Canvas(wrap, height=220, bg="#101820",
                             highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=YES)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        bot = Frame(self, padding=(12, 4, 12, 12))
        bot.pack(fill=X)

        self.start_cnf = dict(text="● 开始", bootstyle=DANGER, command=self.on_start)
        self.pause_cnf = dict(text="❚❚ 暂停", bootstyle=WARNING, command=self.on_pause)

        self.btn_startpause = Button(bot, width=10, **self.start_cnf)
        self.btn_ok = Button(bot, text="✔ 确认", bootstyle=SUCCESS,
                             width=10, command=self.on_confirm)
        self.btn_redo = Button(bot, text="↺ 重录",
                               bootstyle=(SECONDARY, OUTLINE),
                               width=10, command=self.on_redo)
        for b in (self.btn_startpause, self.btn_ok, self.btn_redo):
            b.pack(side=LEFT, padx=4)

        self.time_lbl = Label(bot, text="00:00.0",
                              font=("Consolas", 20), bootstyle="secondary")
        self.time_lbl.pack(side=RIGHT, padx=(0, 8))

    def _center_on_parent(self):
        self.update_idletasks()
        try:
            px = self.master.winfo_rootx()
            py = self.master.winfo_rooty()
            pw = self.master.winfo_width()
            ph = self.master.winfo_height()
            w, h = self.winfo_width(), self.winfo_height()
            x = px + (pw - w) // 2
            y = py + (ph - h) // 3
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _refresh_devices(self, sel_default=False):
        try:
            choices = Micorphone.device_choices()
        except Exception:
            choices = []

        if not choices:
            self.device_combo["values"] = ["（无可用设备）"]
            self.device_var.set("（无可用设备）")
            self._device_index = None
        else:
            labels = [lbl for _, lbl in choices]
            self.device_combo["values"] = labels
            cur = self.device_var.get()
            idx = labels.index(cur) if cur in labels else 0
            if sel_default:
                idx = Micorphone.default_input_device() or idx
            self.device_combo.current(idx)
            self._device_index = choices[idx][0]

        self._update_buttons()

    def _on_device_selected(self, event=None):
        try:
            choices = Micorphone.device_choices()
            sel = self.device_combo.current()
            if 0 <= sel < len(choices):
                self._device_index = choices[sel][0]
        except Exception:
            pass

    def _update_buttons(self):
        has_audio = bool(self._committed) or bool(self._frames)
        has_device = self._device_index is not None

        if self._recording:
            self.btn_ok.config(state=DISABLED)
            self.btn_redo.config(state=DISABLED)
            self.device_combo.config(state=DISABLED)
        elif has_audio:
            self.btn_ok.config(state=NORMAL)
            self.btn_redo.config(state=NORMAL)
            self.device_combo.config(state="readonly")
        else:
            self.btn_startpause.config(
                state=NORMAL if has_device else DISABLED)
            self.btn_ok.config(state=DISABLED)
            self.btn_redo.config(state=DISABLED)
            self.device_combo.config(state="readonly")

    def on_start(self):
        if self._recording:
            return
        if self._device_index is None:
            messagebox.showwarning("提示", "没有可用的录音设备", parent=self)
            return
        self._start_stream()
        self.btn_startpause.config(**self.pause_cnf)

    def on_pause(self):
        if not self._recording:
            return
        self._stop_stream()
        self._commit_frames()
        self._update_buttons()
        self._redraw()
        self.btn_startpause.config(**self.start_cnf)

    def on_confirm(self):
        if self._recording:
            self._stop_stream()
            self._commit_frames()

        if not self._committed:
            messagebox.showinfo("提示", "还没有录制任何音频", parent=self)
            return

        y = np.concatenate(self._committed).astype(np.float32)
        if y.size == 0:
            messagebox.showinfo("提示", "录制结果为空", parent=self)
            return

        self.result = (np.ascontiguousarray(y, dtype=np.float32), self.sr)
        self.destroy()

    def on_redo(self):
        if self._recording:
            self._stop_stream()
        self._frames = []
        self._committed = []
        self._elapsed_accum = 0.0
        self._t0 = None
        self.time_lbl.config(text="00:00.0")
        self._update_buttons()
        self._redraw()

    def _start_stream(self):
        device = self._device_index
        frames = self._frames

        def cb(indata, n_frames, t_info, status):
            frames.append(indata.copy())

        try:
            stream = Micorphone.open_stream(self.sr, device, 1, cb)
            stream.start()
        except Exception as e:
            messagebox.showerror("录音失败",
                                 f"无法打开录音设备：\n{e}", parent=self)
            return

        self._stream = stream
        self._recording = True
        self._t0 = monotonic()

        self._update_buttons()
        self._schedule_refresh()

    def _stop_stream(self):
        self._recording = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._t0 is not None:
            self._elapsed_accum += monotonic() - self._t0
            self._t0 = None

        if self._refresh_id is not None:
            try:
                self.after_cancel(self._refresh_id)
            except Exception:
                pass
            self._refresh_id = None

    def _commit_frames(self):
        if not self._frames:
            return
        data = np.concatenate(self._frames, axis=0)
        if data.ndim > 1:
            data = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.size:
            self._committed.append(data)
        self._frames = []

    def _schedule_refresh(self):
        self._refresh_id = self.after(self.REFRESH_MS, self._tick)

    def _tick(self):
        self._refresh_id = None
        if not self._recording:
            return

        el = self._elapsed_accum
        if self._t0 is not None:
            el += monotonic() - self._t0
        m = int(el // 60)
        s = el - m * 60
        self.time_lbl.config(text=f"{m:02d}:{s:04.1f}")

        self._redraw()
        self._schedule_refresh()

    def _collect_preview_samples(self):
        max_n = int(self.WINDOW_SEC * self.sr * 1.3)
        chunks: list[np.ndarray] = []
        total = 0

        for f in reversed(self._frames):
            a = f.ravel() if f.ndim > 1 else f
            chunks.append(a)
            total += a.size
            if total >= max_n:
                break

        if total < max_n:
            for a in reversed(self._committed):
                chunks.append(a)
                total += a.size
                if total >= max_n:
                    break

        if not chunks:
            return None

        arr = np.concatenate(list(reversed(chunks)))
        if arr.size > max_n:
            arr = arr[-max_n:]
        return arr

    def _redraw(self):
        c = self.canvas
        if c is None:
            return
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or H < 10:
            return

        c.delete("all")

        mid_x = W // 2
        mid_y = H / 2.0

        samples = self._collect_preview_samples()
        if samples is None or samples.size == 0:
            c.create_text(mid_x, mid_y, text="  按「开始」录制",
                          fill="#5a6b7a", font=(FONT, 13), anchor="w")
        else:
            self._draw_wave_left(c, samples, mid_x, mid_y, H)

        c.create_line(mid_x, 0, mid_x, H, fill="#ffb703", width=2)

    def _draw_wave_left(self, c, samples, mid_x, mid_y, H):
        if mid_x < 3:
            return

        window_samples = max(1, int(self.WINDOW_SEC * self.sr))
        k = int(min(samples.size, window_samples))
        if k <= 0:
            return

        tail = samples[-k:]
        abs_tail = np.abs(tail).astype(np.float32)
        ref = float(self.REFERENCE_PEAK) or 1.0
        scale = (H / 2.0 - 10) / ref

        span = k * mid_x / float(window_samples)
        if span < 1.0:
            span = 1.0

        x_left = int(round(mid_x - span))
        if x_left < 0:
            x_left = 0
        num_px = mid_x - x_left
        if num_px < 1:
            return

        spp = k / float(num_px)   # samples per pixel

        for i in range(num_px):
            x = x_left + i
            dist = num_px - i
            j_hi = int(round(k - (dist - 1) * spp))
            j_lo = int(round(k - dist * spp))
            if j_lo < 0:
                j_lo = 0
            if j_hi > k:
                j_hi = k
            if j_hi <= j_lo:
                if j_lo < k:
                    j_hi = j_lo + 1
                else:
                    continue
            chunk = abs_tail[j_lo:j_hi]
            peak = float(chunk.max()) if chunk.size else 0.0
            a = peak * scale
            if a < 1.0:
                a = 1.0
            c.create_line(x, mid_y - a, x, mid_y + a, fill="#3d8bfd")

        c.create_line(x_left, mid_y, mid_x, mid_y, fill="#2a3a4a")

    def _on_close(self):
        if self._recording:
            self._stop_stream()
        self.result = None
        self.destroy()

class TTSDialog(Toplevel):
    MIN_RATE = 50
    MAX_RATE = 400
    DEFAULT_RATE = 200

    def __init__(self, master):
        super().__init__(title="系统语音包")
        self.transient(master)
        self.resizable(True, True)
        self.minsize(560, 480)
        self.focus_set()

        self.result = None

        self._engine_ok = False
        self._voices: list = []
        self._temp_files: list[str] = []

        self._build_ui()
        self._init_engine()
        self._load_voices()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._center_on_parent()

    def _center_on_parent(self):
        self.update_idletasks()
        try:
            px = self.master.winfo_rootx()
            py = self.master.winfo_rooty()
            pw = self.master.winfo_width()
            ph = self.master.winfo_height()
            w, h = self.winfo_width(), self.winfo_height()
            x = px + (pw - w) // 2
            y = py + (ph - h) // 3
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _build_ui(self):
        bot = Frame(self, padding=(14, 4, 14, 14))
        bot.pack(side=BOTTOM, fill=X)

        self.btn_preview = Button(bot, text="🔊 预听", bootstyle=SUCCESS,
                                  width=10, command=self.on_preview)
        self.btn_ok = Button(bot, text="✔ 确定", bootstyle=PRIMARY,
                             width=10, command=self.on_confirm)
        self.btn_cancel = Button(bot, text="✖ 取消",
                                 bootstyle=(SECONDARY, OUTLINE),
                                 width=10, command=self._on_close)
        for b in (self.btn_preview, self.btn_ok, self.btn_cancel):
            b.pack(side=LEFT, padx=4)

        self.status_lbl = Label(bot, text="", bootstyle="secondary")
        self.status_lbl.pack(side=RIGHT, padx=(8, 0))

        wrap = Frame(self, padding=(14, 14, 14, 6))
        wrap.pack(fill=BOTH, expand=YES)

        Label(wrap, text="要朗读的文本：",
              font=(FONT, 10, "bold")).pack(anchor=W)

        txt_wrap = Frame(wrap)
        txt_wrap.pack(fill=BOTH, expand=YES, pady=(4, 12))

        self.text = Text(
            txt_wrap, height=8, wrap=WORD,
            font=(FONT, 12), bd=0,
            highlightthickness=1,
            highlightbackground="#c8d0da",
            highlightcolor="#3d8bfd",
            padx=8, pady=6,
        )
        txt_sb = Scrollbar(txt_wrap, orient=VERTICAL, command=self.text.yview)
        self.text.config(yscrollcommand=txt_sb.set)
        self.text.pack(side=LEFT, fill=BOTH, expand=YES)
        txt_sb.pack(side=RIGHT, fill=Y)
        self.text.focus_set()

        row1 = Frame(wrap)
        row1.pack(fill=X, pady=(0, 10))

        Label(row1, text="语速", font=(FONT, 10, "bold")).pack(side=LEFT)

        self.rate_var = DoubleVar(value=float(self.DEFAULT_RATE))
        self.rate_scale = Scale(
            row1, from_=self.MIN_RATE, to=self.MAX_RATE,
            resolution=10, orient=HORIZONTAL, variable=self.rate_var,
            command=self._on_rate_change, showvalue=False,
            highlightthickness=0, bd=0,
            troughcolor="#dfe6ec", activebackground="#3d8bfd",
            sliderrelief="flat", length=280,
        )
        self.rate_scale.pack(side=LEFT, padx=(10, 10), fill=X, expand=YES)

        self.rate_lbl = Label(row1, text=f"{self.DEFAULT_RATE}",
                              bootstyle="secondary", width=5)
        self.rate_lbl.pack(side=RIGHT)

        row2 = Frame(wrap)
        row2.pack(fill=X, pady=(0, 4))

        Label(row2, text="声音", font=(FONT, 10, "bold")).pack(side=LEFT)

        self.voice_var = StringVar()
        self.voice_combo = Combobox(
            row2, textvariable=self.voice_var,
            state="readonly", width=40, bootstyle="info",
        )
        self.voice_combo.pack(side=LEFT, padx=(10, 0), fill=X, expand=YES)

    def _on_rate_change(self, val=None):
        self.rate_lbl.config(
            text=f"{int(round(float(self.rate_var.get())))}")

    def _init_engine(self):
        try:
            e = init()
            self._voices = list(e.getProperty("voices") or [])
            self._engine_ok = True
        except Exception:
            self._engine_ok = False
            self._voices = []

    def _load_voices(self):
        if not self._engine_ok:
            self.voice_combo["values"] = ["（语音引擎不可用）"]
            self.voice_var.set("（语音引擎不可用）")
            self.voice_combo.config(state=DISABLED)
        elif not self._voices:
            self.voice_combo["values"] = ["（默认声音）"]
            self.voice_var.set("（默认声音）")
            self.voice_combo.config(state=DISABLED)
        else:
            labels = []
            for i, v in enumerate(self._voices):
                nm = getattr(v, "name", None) or getattr(v, "id", None) \
                     or f"声音 {i}"
                labels.append(str(nm))
            self.voice_combo["values"] = labels
            self.voice_combo.current(0)

        self._update_buttons()

    def _update_buttons(self):
        st = NORMAL if self._engine_ok else DISABLED
        self.btn_preview.config(state=st)
        self.btn_ok.config(state=st)

    def _selected_voice_id(self):
        if not self._voices:
            return None
        i = self.voice_combo.current()
        if not (0 <= i < len(self._voices)):
            return None
        return getattr(self._voices[i], "id", None)

    def _synthesize(self, text, rate, voice_id):
        fd, path = mkstemp(suffix=".wav")
        os.close(fd)
        self._temp_files.append(path)

        e = init()
        e.setProperty("rate", int(rate))
        if voice_id:
            e.setProperty("voice", voice_id)
        e.save_to_file(text, path)
        e.runAndWait()
        return path

    def _run_synth(self):
        text = self.text.get("1.0", END).strip()
        if not text:
            messagebox.showinfo("提示", "请输入要朗读的文本", parent=self)
            return None
        if not self._engine_ok:
            messagebox.showerror("错误", "系统语音引擎不可用", parent=self)
            return None

        rate = int(round(float(self.rate_var.get())))
        vid = self._selected_voice_id()

        self.status_lbl.config(text="正在合成……")
        self.update_idletasks()
        try:
            path = self._synthesize(text, rate, vid)
        except Exception as e:
            self.status_lbl.config(text="")
            messagebox.showerror("合成失败", str(e), parent=self)
            return None
        self.status_lbl.config(text="")
        return path

    def on_preview(self):
        self._cleanup()

        path = self._run_synth()
        if path is None:
            return
        try:
            y, sr = AudioUtil.load_audio(path)
        except Exception as e:
            messagebox.showerror("加载失败", str(e), parent=self)
            return
        if y.size == 0:
            messagebox.showinfo("提示", "合成结果为空", parent=self)
            return

        AsyncPlayer.play_array(y, sr)
        self.status_lbl.config(
            text=f"已合成 {len(y) / float(sr):.2f}s（{sr} Hz）")

    def on_confirm(self):
        path = self._run_synth()
        if path is None:
            return
        try:
            y, sr = AudioUtil.load_audio(path)
        except Exception as e:
            messagebox.showerror("加载失败", str(e), parent=self)
            return
        if y.size == 0:
            messagebox.showinfo("提示", "合成结果为空", parent=self)
            return

        self.result = (np.ascontiguousarray(y, dtype=np.float32), int(sr))
        self._cleanup()
        self.destroy()

    def _cleanup(self):
        for p in self._temp_files:
            try:
                os.remove(p)
            except Exception:
                pass
        self._temp_files = []

    def _on_close(self):
        self.result = None
        self._cleanup()
        self.destroy()

class NoteEnvDialog(Toplevel):
    def __init__(self, master, syllable: "Syllable"):
        super().__init__(title="音符包络")
        self.transient(master)
        self.focus_set()

        self.syllable = syllable
        self.result = None
        self._original = list(syllable.pitch_offsets)
        self._vars: list[DoubleVar] = []

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._center_on_parent()

    def _center_on_parent(self):
        self.update_idletasks()
        try:
            px = self.master.winfo_rootx()
            py = self.master.winfo_rooty()
            pw = self.master.winfo_width()
            ph = self.master.winfo_height()
            w, h = self.winfo_width(), self.winfo_height()
            x = px + (pw - w) // 2
            y = py + (ph - h) // 3
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _build_ui(self):
        head = Frame(self, padding=(20, 18, 20, 4))
        head.pack(fill=X)
        Label(head, text=f"「{self.syllable.char}」的相对音高",
              font=(FONT, 13, "bold")).pack()
        base = int(round(self.syllable.semitones))
        Label(head, text=f"基准音高 {base:+d} 半音 · 字长四等分",
              bootstyle="secondary").pack(pady=(2, 0))

        body = Frame(self, padding=(20, 10, 20, 6))
        body.pack(fill=X)

        for i in range(self.syllable.N_ENV):
            row = Frame(body)
            row.pack(fill=X, pady=3)
            Label(row, text=f"第 {i + 1} 段", width=6,
                  anchor=W, font=(FONT, 10)).pack(side=LEFT)

            v = DoubleVar(value=float(self.syllable.pitch_offsets[i]))
            self._vars.append(v)

            sc = Scale(row, from_=-24.0, to=24.0, resolution=1.0,
                       orient=HORIZONTAL, variable=v, showvalue=False,
                       highlightthickness=0, bd=0, length=260,
                       troughcolor="#dfe6ec", activebackground="#3d8bfd",
                       sliderrelief="flat")
            sc.pack(side=LEFT, padx=(4, 8), fill=X, expand=1)

            lbl = Label(row, text=f"{int(v.get()):+d}", width=4,
                        anchor=E, bootstyle="secondary")
            lbl.pack(side=LEFT)

            v.trace_add("write",
                        lambda *a, i=i, v=v, lbl=lbl:
                        self._on_change(i, v, lbl))

        btns = Frame(self, padding=(20, 4, 20, 18))
        btns.pack(fill=X)
        Button(btns, text="清零", bootstyle=(SECONDARY, OUTLINE), width=7,
               command=self._clear).pack(side=LEFT, padx=2)
        Button(btns, text="预听", bootstyle=SUCCESS, width=7,
               command=self._preview).pack(side=LEFT, padx=2)
        
        Button(btns, text="取消", bootstyle=(SECONDARY, OUTLINE), width=7,
               command=self._on_cancel).pack(side=RIGHT, padx=2)
        Button(btns, text="确定", bootstyle=PRIMARY, width=7,
               command=self._ok).pack(side=RIGHT, padx=2)

    def _on_change(self, i, var: DoubleVar, lbl: Label):
        v = int(round(float(var.get())))
        lbl.config(text=f"{v:+d}")
        self.syllable.pitch_offsets[i] = float(v)
        self.syllable._key = None
        self.syllable._out = None

    def _clear(self):
        if messagebox.askyesno("确认", "确定将四段全部置零？"):
            for i, v in enumerate(self._vars):
                v.set(0.0)

    def _preview(self):
        try:
            y = self.syllable.process()
        except Exception as e:
            messagebox.showerror("处理失败", str(e), parent=self)
            return
        AsyncPlayer.play_array(y, self.syllable.sr)

    def _ok(self):
        self.result = list(self.syllable.pitch_offsets)
        self.destroy()

    def _on_cancel(self):
        self.syllable.pitch_offsets = list(self._original)
        self.syllable._key = None
        self.syllable._out = None
        self.result = None
        self.destroy()

class History:
    def __init__(self, app: "App", max_size=80):
        self._app_ref = weakref_ref(app)
        self.max_size = max_size
        self.undo_stack: list[tuple[str, dict]] = []
        self.redo_stack: list[tuple[str, dict]] = []
        self._pending_key = None
        self._applying = False

    @property
    def app(self):
        return self._app_ref()

    def push(self, label, key=None):
        if self._applying:
            return
        if key is not None and key == self._pending_key:
            return
        snap = self.app._snapshot_state()
        self.undo_stack.append((label, snap))
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._pending_key = key
        self.app._update_history_ui()

    def commit(self):
        self._pending_key = None

    def cancel_last(self):
        if self.undo_stack:
            self.undo_stack.pop()
        self._pending_key = None
        self.app._update_history_ui()

    def undo(self):
        if not self.undo_stack:
            return
        label, snap = self.undo_stack.pop()
        self.redo_stack.append((label, self.app._snapshot_state()))
        self._apply(snap, f"已撤销：{label}")

    def redo(self):
        if not self.redo_stack:
            return
        label, snap = self.redo_stack.pop()
        self.undo_stack.append((label, self.app._snapshot_state()))
        self._apply(snap, f"已重做：{label}")

    def _apply(self, snap, msg):
        self._applying = True
        try:
            self.app._restore_state(snap)
            self.app.status.config(text=msg)
        finally:
            self._applying = False
        self._pending_key = None
        self.app._update_history_ui()

    @property
    def can_undo(self):
        return bool(self.undo_stack)

    @property
    def can_redo(self):
        return bool(self.redo_stack)

    def clear(self):
            self.undo_stack.clear()
            self.redo_stack.clear()
            self._pending_key = None
            self._applying = False

class HistoryViewLabel(Label):
    def __init__(self, app: "App", btn: Button):
        super().__init__(app, bootstyle=LIGHT)
        self.btn = btn
        self._app_ref = weakref_ref(app)

    @property
    def app(self):
        return self._app_ref()

    def update_text(self):
        if self.app.history.undo_stack:
            ...
        self.config("\n")