from utils import *

class Track:
    def __init__(self, name, y, sr, path=None):
        self.name = str(name)
        self.y = (np.ascontiguousarray(np.asarray(y, dtype=np.float32))
                  if y is not None else None)
        self.sr = int(sr) if sr is not None else None
        self.path = path
        self.sel_start = None
        self.sel_end = None
        self.canvas: None | Canvas = None
        self.frame = None
        self._drag_anchor = None

        self._drag_mode = None        # None | "new" | "left" | "right" | "move"
        self._drag_origin_t = None    # 按下时刻
        self._drag_start0 = None      # 按下时原始 sel_start
        self._drag_end0 = None        # 按下时原始 sel_end

    @property
    def duration(self):
        if self.y is None or not self.sr:
            return 0.0
        return len(self.y) / float(self.sr)

    @property
    def n_samples(self):
        return 0 if self.y is None else int(len(self.y))

class Syllable:
    MIN_LEN = 2048
    N_ENV = 4 # 字内分段调音四等分

    def __init__(self, char, audio, sr, start, end, sec_per_beat=0.5,
                 track_index=0):
        self.char = char
        self._audio = audio
        self.sr = int(sr)
        self.start = int(start)
        self.end = int(end)
        self.sec_per_beat = float(sec_per_beat)
        self.flatten = False
        self.track_index = int(track_index)
        
        self.pitch_offsets = [0.0] * self.N_ENV

        self.beats = App.DEFAULT_BEATS

        self.semitones = 0.0
        self.volume_db = 0.0
        self.keep_formant = True
        self._key = None
        self._out = None

    def has_note_env(self) -> bool:
        return any(abs(x) > 1e-3 for x in self.pitch_offsets)

    def bind_audio(self, audio, sr):
        self._audio = audio
        self.sr = int(sr)
        self._key = None
        self._out = None

    def audio_slice(self):
        n = len(self._audio)
        a = int(np.clip(self.start, 0, n))
        b = int(np.clip(self.end, 0, n))
        if b < a:
            a, b = b, a
        audio = self._audio[a:b]

        if self.flatten and audio.size >= self.MIN_LEN:
            return AudioUtil.flatten_tone(audio, self.sr)
        else:
            return audio

    @property
    def orig_dur(self):
        return max(0, int(self.end) - int(self.start)) / float(self.sr)

    @property
    def target_dur(self):
        return max(1e-3, float(self.beats) * float(self.sec_per_beat))

    @property
    def dur_ratio(self):
        d = self.orig_dur
        return 1.0 if d <= 0 else self.target_dur / d

    @property
    def cur_dur(self):
        return self.target_dur

    def process(self):
        key = (int(self.start),
               int(self.end),
               round(self.dur_ratio, 4),
               round(self.semitones, 3),
               tuple(round(x, 3) for x in self.pitch_offsets),
               bool(self.keep_formant),
               round(self.volume_db, 2),
               bool(self.flatten))
        if key == self._key and self._out is not None:
            return self._out

        y = np.array(self.audio_slice(), dtype=np.float32, copy=True)

        need_stretch = abs(self.dur_ratio - 1.0) > 1e-3
        need_pitch = abs(self.semitones) > 1e-3
        need_env = self.has_note_env()
        need_vol = abs(self.volume_db) > 1e-3

        if need_stretch:
            if len(y) < self.MIN_LEN:
                y = np.pad(y, (0, self.MIN_LEN - len(y)))
            rate = 1.0 / max(1e-3, self.dur_ratio)
            y = time_stretch(y=y, rate=rate)

        if need_env:
            y = self._apply_note_env(y)
        elif need_pitch:
            if len(y) < self.MIN_LEN:
                y = np.pad(y, (0, self.MIN_LEN - len(y)))
            y = AudioUtil.pitch_shift_keep_formant(
                y, self.sr, float(self.semitones),
                keep_formant=bool(self.keep_formant),
            )

        y = np.asarray(y, dtype=np.float32)

        if need_vol:
            y = y * AudioUtil.db_to_gain(self.volume_db)
            y = np.asarray(y, dtype=np.float32)

        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1.0:
            y = (y / peak) * 0.99

        self._out = y
        self._key = key
        return self._out

    def _apply_note_env(self, y):
        y = np.asarray(y, dtype=np.float32)
        n = len(y)
        if n <= 0:
            return y
    
        base = float(self.semitones)
        offsets = list(self.pitch_offsets) + [0.0] * (self.N_ENV - len(self.pitch_offsets))
        offsets = offsets[:self.N_ENV]
    
        if n < self.N_ENV * 32:
            avg = base + float(np.mean(offsets))
            if abs(avg) < 1e-3:
                return y
            if n < self.MIN_LEN:
                y2 = np.pad(y, (0, self.MIN_LEN - n))
                out = AudioUtil.pitch_shift_keep_formant(
                    y2, self.sr, avg, keep_formant=bool(self.keep_formant))
                return np.asarray(out, dtype=np.float32)[:n]
            out = AudioUtil.pitch_shift_keep_formant(
                y, self.sr, avg, keep_formant=bool(self.keep_formant))
            return np.asarray(out, dtype=np.float32)
    
        bounds = np.linspace(0, n, self.N_ENV + 1).astype(np.int64)
        fade_n = max(1, int(round(self.sr * 0.003)))
        pieces = []
    
        for i in range(self.N_ENV):
            a, b = int(bounds[i]), int(bounds[i + 1])
            seg = y[a:b].astype(np.float32, copy=True)
            if seg.size == 0:
                continue
    
            off = base + float(offsets[i])
            if abs(off) >= 1e-3:
                if seg.size < self.MIN_LEN:
                    pad = np.pad(seg, (0, self.MIN_LEN - seg.size))
                    out = AudioUtil.pitch_shift_keep_formant(
                        pad, self.sr, off, keep_formant=bool(self.keep_formant))
                    out = np.asarray(out, dtype=np.float32)[:seg.size]
                else:
                    out = AudioUtil.pitch_shift_keep_formant(
                        seg, self.sr, off, keep_formant=bool(self.keep_formant))
                    out = np.asarray(out, dtype=np.float32)
                    if out.size > seg.size:
                        out = out[:seg.size]
                    elif out.size < seg.size:
                        out = np.pad(out, (0, seg.size - out.size))
                seg = out
    
            if i > 0 and seg.size > 2 * fade_n:
                seg[:fade_n] *= np.linspace(0, 1, fade_n, dtype=np.float32)
            if i < self.N_ENV - 1 and seg.size > 2 * fade_n:
                seg[-fade_n:] *= np.linspace(1, 0, fade_n, dtype=np.float32)[::-1]
    
            pieces.append(seg)
    
        if not pieces:
            return y
        return np.concatenate(pieces).astype(np.float32)

    def reset(self):
        self.beats = App.DEFAULT_BEATS
        self.semitones = 0.0
        self.volume_db = 0.0
        self.keep_formant = True
        self.flatten = False
        self._key = None
        self._out = None
        self.pitch_offsets = [0.0] * self.N_ENV

class App(Window):
    REFERENCE_PEAK = 1.0
    HANDLE_PX = 6
    MIN_SEL_SEC = 0.002
    DEFAULT_BEATS = 1.0
    MAX_SCROLL_TIMES = 10

    _CURSORS = {
        "left":  "sb_h_double_arrow",
        "right": "sb_h_double_arrow",
        "move":  "fleur",
        "new":   "cross",
    }

    def __init__(self):
        super().__init__(themename="flatly")
        self.title("MovableType 活字乱刷术")
        self.minsize(980, 720)

        self.tracks: list[Track] = []
        self._active_track = 0
        self.history = History(self, max_size=80)

        self.syllables: list[Syllable] = []

        self._loading = False
        self._suppress_tab = False

        self.sec_per_beat = 0.5

        self._bg = self.style.colors.bg
        self._fg = self.style.colors.fg

        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.update()
        self.minsize(self.winfo_width(), self.winfo_height())

    @property
    def cur_track(self):
        if not self.tracks:
            return None
        if 0 <= self._active_track < len(self.tracks):
            return self.tracks[self._active_track]
        return None

    def _build_ui(self):
        bar = Frame(self, padding=(10, 8, 10, 4))
        bar.pack(fill=X)

        mbt = Menubutton(bar, text="➕️ 音频 / 音轨", bootstyle=(PRIMARY, OUTLINE))
        add_aud_menu = Menu(mbt, tearoff=0)
        add_aud_menu.add_command(label="📂 从文件导入", command=self.import_audio)
        add_aud_menu.add_command(label="🎤 录制", command=self.get_recording)
        add_aud_menu.add_command(label="🔊 系统语音包", command=self.make_tts)
        add_aud_menu.add_separator()
        add_aud_menu.add_command(label="🗑 删除当前音频", command=self.delete_current_track)
        mbt["menu"] = add_aud_menu
        mbt.pack(side=LEFT, padx=4)

        self.play_btn = Button(bar, text="▶ 播放", bootstyle=(SUCCESS, OUTLINE),
                               command=self.play_audio, width=8)
        self.play_btn.pack(side=LEFT, padx=4)

        Separator(bar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)

        self.undo_btn = Button(bar, text="↶", bootstyle=(SUCCESS, OUTLINE),
                               command=self.history.undo, state=DISABLED)
        self.redo_btn = Button(bar, text="↷", bootstyle=(SUCCESS, OUTLINE),
                               command=self.history.redo, state=DISABLED)
        
        self.undo_btn.pack(side=LEFT, padx=(0, 2))
        self.redo_btn.pack(side=LEFT, padx=(0, 2))
        self.bind_all("<Control-z>", lambda e: self.history.undo())
        self.bind_all("<Control-y>", lambda e: self.history.redo())
        self.bind_all("<Control-Shift-Z>", lambda e: self.history.redo())

        Button(bar, text="📦 打开工程", bootstyle=(INFO, OUTLINE),
                  command=self.import_project).pack(side=LEFT, padx=4)
        Button(bar, text="💾 保存工程", bootstyle=(INFO, OUTLINE),
                  command=self.export_project).pack(side=LEFT, padx=4)

        Separator(bar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)

        self.in_var = StringVar(value="0.000")
        e1 = Entry(bar, textvariable=self.in_var, width=5)
        e1.pack(side=LEFT, padx=4)
        e1.bind("<Return>", self.apply_entry_times)
        e1.bind("<FocusOut>", self.apply_entry_times)

        Label(bar, text=" ~ ").pack(side=LEFT)
        self.out_var = StringVar(value="0.000")
        e2 = Entry(bar, textvariable=self.out_var, width=5)
        e2.pack(side=LEFT, padx=4)
        e2.bind("<Return>", self.apply_entry_times)
        e2.bind("<FocusOut>", self.apply_entry_times)

        Button(bar, text="✂ 剪出", bootstyle=WARNING,
                  command=self.extract_syllable).pack(side=LEFT, padx=12)

        Button(bar, text="🔗 合成音乐", bootstyle=(PRIMARY, OUTLINE),
                  command=self.concat_export).pack(side=LEFT, padx=4)

        Separator(bar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)
        Label(bar, text="一拍(s)").pack(side=LEFT)
        self.spb_var = StringVar(value=f"{self.sec_per_beat:.3f}")
        spb_entry = Entry(bar, textvariable=self.spb_var, width=7,
                             justify=CENTER)
        spb_entry.pack(side=LEFT, padx=4)
        spb_entry.bind("<Return>", self.apply_spb)
        spb_entry.bind("<FocusOut>", self.apply_spb)

        self.bpm_lbl = Label(bar, text=f"= {60.0 / self.sec_per_beat:.1f} BPM",
                                bootstyle="secondary")
        self.bpm_lbl.pack(side=LEFT, padx=4)

        self.nb_wrap = Frame(self, height=200)
        self.nb_wrap.pack(fill=X, padx=10, pady=(4, 6))
        self.nb_wrap.pack_propagate(False)

        self.nb = Notebook(self.nb_wrap)
        self.nb.pack(fill=BOTH, expand=YES)
        self.nb.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.nb.bind("<Double-Button-1>", self.on_tab_double_click)

        self.empty_hint = Label(
            self.nb_wrap,
            text="点击「➕️ 音频 / 音轨」或「📦 打开工程」载入",
            font=(FONT, 11), bootstyle="secondary",
            anchor=CENTER, justify=CENTER,
        )

        main = Frame(self, padding=(10, 0, 10, 6))
        main.pack(fill=BOTH, expand=YES)

        lf = Labelframe(main, text=" 单字列表 ", padding=10)
        lf.pack(side=LEFT, fill=BOTH, padx=(0, 10))

        lb_wrap = Frame(lf)
        lb_wrap.pack(fill=BOTH, expand=YES, padx=4)

        self.listbox = Listbox(
            lb_wrap, width=8, height=1, font=(FONT, 16), exportselection=False,
            activestyle="none", justify=CENTER,
            bg=self._bg, fg=self._fg,
            selectbackground="#3d8bfd", selectforeground="white",
            highlightthickness=0, bd=0,
        )
        sb = Scrollbar(lb_wrap, orient=VERTICAL, command=self.listbox.yview)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.pack(side=LEFT, fill=BOTH, expand=YES)
        sb.pack(side=RIGHT, fill=Y)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.listbox.bind("<Double-Button-1>", lambda e: self.preview())

        btns = Frame(lf, padding=(0, 8, 0, 0))
        btns.pack(fill=X)
        Button(btns, text="🗑 删除", bootstyle=(DANGER, OUTLINE),
                  command=self.delete_syllable).pack(side=LEFT, padx=4, expand=1, fill=X)
        Button(btns, text="⬆️ 上移", bootstyle=(SECONDARY, OUTLINE),
                  command=lambda: self.move_syllable(-1)).pack(side=LEFT, padx=4, expand=1, fill=X)
        Button(btns, text="⬇️ 下移", bootstyle=(SECONDARY, OUTLINE),
                  command=lambda: self.move_syllable(1)).pack(side=LEFT, padx=4, expand=1, fill=X)

        right = Labelframe(main, text=" 单字编辑 ", padding=16)
        right.pack(side=LEFT, fill=BOTH, expand=YES)

        self.char_label = Label(right, text="—", font=(FONT, 28, "bold"),
                                   anchor=CENTER)
        self.char_label.pack(fill=X, pady=(6, 2))

        self.info_label = Label(right, text="请在左侧列表中选择一个单字",
                                   anchor=CENTER, bootstyle="secondary")
        self.info_label.pack(fill=X, pady=(0, 8))

        Separator(right).pack(fill=X, pady=(6, 0))

        row = Frame(right)
        row.pack(fill=X, pady=(10, 0))
        Label(row, text="时长（拍）", font=(FONT, 10, "bold")).pack(side=LEFT, pady=(16, 0))
        self.beats_lbl = Label(row, text="1.00 拍", bootstyle="secondary")
        self.beats_lbl.pack(side=RIGHT)

        self.beats_var = DoubleVar(value=1.0)
        self.beats_scale = Scale(
            right, from_=0.1, to=8.0, resolution=0.1, orient=HORIZONTAL,
            variable=self.beats_var, command=self.on_beats_change, showvalue=False,
            bg=self._bg, fg=self._fg, highlightthickness=0, bd=0,
            troughcolor="#dfe6ec", activebackground="#3d8bfd", sliderrelief="flat",
            length=320,
        )
        self.beats_scale.pack(fill=X, pady=(2, 2))

        row2 = Frame(right)
        row2.pack(fill=X)
        Label(row2, text="音高偏移", font=(FONT, 10, "bold")).pack(side=LEFT, pady=(16, 0))
        
        self.pitch_lbl = Label(row2, text="+0 半音", bootstyle="secondary")
        self.pitch_lbl.pack(side=RIGHT)

        self.pitch_var = DoubleVar(value=0.0)
        self.pitch_scale = Scale(
            right, from_=-24.0, to=24.0, resolution=1.0, orient=HORIZONTAL,
            variable=self.pitch_var, command=self.on_pitch_change, showvalue=False,
            bg=self._bg, fg=self._fg, highlightthickness=0, bd=0,
            troughcolor="#dfe6ec", activebackground="#3d8bfd", sliderrelief="flat",
            length=320,
        )
        self.pitch_scale.pack(fill=X, pady=(2, 10))

        chk_frame = Frame(right)
        chk_frame.pack(anchor=W)

        self.keep_formant_var = BooleanVar(value=True)
        self.formant_chk = Checkbutton(
            chk_frame, text="保持音色",
            variable=self.keep_formant_var, bootstyle="round-toggle",
            command=self.on_formant_toggle,
        )
        self.formant_chk.pack(side=LEFT, pady=4, padx=4)

        self.flatten_var = BooleanVar(value=False)
        self.flatten_chk = Checkbutton(
            chk_frame, text="音调校准",
            variable=self.flatten_var, bootstyle="round-toggle",
            command=self.on_flatten_toggle,
        )
        self.flatten_chk.pack(side=LEFT, padx=12)

        self.note_env_btn = Button(chk_frame, text="🎼 分段调音", bootstyle=(INFO, OUTLINE),
                                   width=12, command=self.edit_note_env, compound=LEFT)
        self.note_env_btn.pack(side=LEFT, padx=12)

        row3 = Frame(right)
        row3.pack(fill=X)
        Label(row3, text="音量", font=(FONT, 10, "bold")).pack(side=LEFT, pady=(16, 0))
        self.vol_lbl = Label(row3, text="+0.0 dB", bootstyle="secondary")
        self.vol_lbl.pack(side=RIGHT)

        self.vol_var = DoubleVar(value=0.0)
        self.vol_scale = Scale(
            right, from_=-24.0, to=24.0, resolution=0.5, orient=HORIZONTAL,
            variable=self.vol_var, command=self.on_vol_change, showvalue=False,
            bg=self._bg, fg=self._fg, highlightthickness=0, bd=0,
            troughcolor="#dfe6ec", activebackground="#3d8bfd", sliderrelief="flat",
            length=320,
        )
        self.vol_scale.pack(fill=X, pady=(2, 6))

        vol_btns = Frame(right)
        vol_btns.pack(fill=X, pady=(0, 8))
        Button(vol_btns, text="0 dB", bootstyle=(SECONDARY, OUTLINE),
                  width=6, command=lambda: self.set_volume(0.0)).pack(side=LEFT, padx=1)
        Button(vol_btns, text="-6 dB", bootstyle=(SECONDARY, OUTLINE),
                  width=6, command=lambda: self.set_volume(-6.0)).pack(side=LEFT, padx=1)
        Button(vol_btns, text="+6 dB", bootstyle=(SECONDARY, OUTLINE),
                  width=6, command=lambda: self.set_volume(6.0)).pack(side=LEFT, padx=1)

        for sc in (self.beats_scale, self.pitch_scale, self.vol_scale):
            sc.bind("<ButtonRelease-1>", lambda e: self.history.commit())

        Separator(right).pack(fill=X, pady=6)

        act = Frame(right)
        act.pack(fill=X, pady=(12, 0))
        Button(act, text="🔊 预听", bootstyle=SUCCESS, width=10,
                  command=self.preview).pack(side=LEFT, padx=2)
        Button(act, text="↺ 重置", bootstyle=(SECONDARY, OUTLINE), width=10,
                  command=self.reset_current).pack(side=LEFT, padx=2)
        Button(act, text="✏ 重命名", bootstyle=(INFO, OUTLINE), width=10,
                  command=self.rename_current).pack(side=LEFT, padx=2)
        Button(act, text="💾 导出", bootstyle=(PRIMARY, OUTLINE), width=10,
                  command=self.export_current).pack(side=LEFT, padx=2)

        self.status = Label(self, text="就绪 · 播放后端：{}"
                               .format("SoundDevice 库" if HAS_SD else "缓存文件 & 系统播放器"),
                               anchor=W, bootstyle="secondary", padding=(12, 4))
        self.status.pack(fill=X, side=BOTTOM)

        self._update_empty_hint()

    def _update_history_ui(self):
        self.undo_btn.config(
            state=NORMAL if self.history.can_undo else DISABLED,
            bootstyle=SUCCESS if self.history.can_undo else (SUCCESS, OUTLINE)
        )
        self.redo_btn.config(
            state=NORMAL if self.history.can_redo else DISABLED,
            bootstyle=SUCCESS if self.history.can_redo else (SUCCESS, OUTLINE)
        )

    def _update_empty_hint(self):
        """根据是否有音轨，显示/隐藏空状态提示。"""
        try:
            if self.tracks:
                self.empty_hint.place_forget()
            else:
                self.empty_hint.place(relx=0.5, rely=0.5, anchor=CENTER)
        except Exception:
            pass

    def add_track(self, y, sr, path=None, name=None, select=True):
        if name is None:
            if path:
                base = os.path.splitext(os.path.basename(path))[0]
                name = base or f"音轨 {len(self.tracks) + 1}"
            else:
                name = f"音轨 {len(self.tracks) + 1}"
        name = safe_filename(name, fallback=f"音轨 {len(self.tracks) + 1}",
                             maxlen=24)

        track = Track(name, y, sr, path)

        frame = Frame(self.nb)
        canvas = Canvas(frame, height=150, bg="#101820",
                           highlightthickness=0, cursor="cross")
        canvas.pack(fill=BOTH, expand=YES)
        canvas.bind("<Configure>", lambda e, t=track: self._draw_track(t))
        canvas.bind("<ButtonPress-1>", lambda e, t=track: self.on_canvas_press(e, t))
        canvas.bind("<B1-Motion>", lambda e, t=track: self.on_canvas_drag(e, t))
        canvas.bind("<ButtonRelease-1>", lambda e, t=track: self.on_canvas_release(e, t))
        canvas.bind("<Motion>", lambda e, t=track: self.on_canvas_motion(e, t))
        canvas.bind("<Leave>",  lambda e, t=track: self._set_cursor(t, "cross"))
        canvas.bind("<MouseWheel>", lambda e, t=track: self.on_canvas_scroll(e, t))
        track.canvas = canvas
        track.frame = frame

        self.tracks.append(track)
        self.nb.add(frame, text=name)

        if select:
            self.nb.select(frame)
            self._active_track = len(self.tracks) - 1
            self.on_tab_changed()

        self._update_empty_hint()
        return track

    def _clear_tracks(self):
        self._suppress_tab = True
        try:
            for t in self.tracks:
                if t.frame is not None:
                    try:
                        self.nb.forget(t.frame)
                        t.frame.destroy()
                    except Exception:
                        pass
        finally:
            self._suppress_tab = False
        self.tracks = []
        self._active_track = 0
        self._update_empty_hint()

    def delete_current_track(self):
        t = self.cur_track
        if t is None:
            messagebox.showinfo("提示", "当前没有音轨")
            return
        idx = self._active_track
        n_syl = sum(1 for s in self.syllables
                    if int(getattr(s, "track_index", 0)) == idx)
        msg = f"确定要删除音轨「{t.name}」吗？"
        if n_syl:
            msg += f"\n\n来自该音轨的 {n_syl} 个单字也会被一并删除。"
        if not messagebox.askyesno("确认", msg):
            return

        self.syllables = [s for s in self.syllables
                          if int(getattr(s, "track_index", 0)) != idx]
        for s in self.syllables:
            ti = int(getattr(s, "track_index", 0))
            if ti > idx:
                s.track_index = ti - 1

        self._suppress_tab = True
        try:
            try:
                self.nb.forget(t.frame)
                t.frame.destroy()
            except Exception:
                pass
        finally:
            self._suppress_tab = False

        self.tracks.pop(idx)
        if self.tracks:
            self._active_track = min(idx, len(self.tracks) - 1)
            try:
                self.nb.select(self.tracks[self._active_track].frame)
            except Exception:
                pass
        else:
            self._active_track = 0

        self.history.redo_stack.clear()
        self.history.undo_stack.clear()
        self._update_history_ui()

        self.refresh_list(select=None)
        self.on_tab_changed()
        self.status.config(text=f"已删除音轨「{t.name}」")
        self._update_empty_hint()

    def current_index(self) -> int | None:
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def _track_duration(self, track):
        if track is None or track.y is None or not track.sr:
            return 0.0
        return len(track.y) / float(track.sr)

    def _time_to_x(self, track, t):
        W = track.canvas.winfo_width() if track.canvas else 0
        td = self._track_duration(track)
        return 0 if td <= 0 else t / td * W

    def _x_to_time(self, track, x):
        W = track.canvas.winfo_width() if track.canvas else 0
        td = self._track_duration(track)
        if W <= 0 or td <= 0:
            return 0.0
        return max(0.0, min(td, x / W * td))

    def on_tab_changed(self, event=None):
        if self._suppress_tab:
            return
        try:
            idx = self.nb.index("current")
        except Exception:
            return
        if idx is None:
            return
        try:
            idx = int(idx)
        except Exception:
            return
        if idx < 0 or idx >= len(self.tracks):
            return
        self._active_track = idx
        t = self.tracks[idx]

        if t.sel_start is None or t.sel_end is None:
            self.in_var.set("0.000")
            self.out_var.set(f"{t.duration:.3f}")
        else:
            self.in_var.set(f"{t.sel_start:.3f}")
            self.out_var.set(f"{t.sel_end:.3f}")

        self._draw_track(t)

        idx = self.current_index()
        if idx is not None and idx < len(self.syllables):
            s = self.syllables[idx]
            if int(getattr(s, "track_index", 0)) == self._active_track:
                sr = int(s.sr or t.sr or 1)
                t.sel_start = s.start / float(sr)
                t.sel_end   = s.end   / float(sr)
                self.in_var.set(f"{t.sel_start:.3f}")
                self.out_var.set(f"{t.sel_end:.3f}")
    
    def on_tab_double_click(self, event):
        """双击 Notebook 标签页文本 → 重命名对应音轨。"""
        try:
            tab_id = self.nb.identify(event.x, event.y)
        except Exception:
            return
        if not tab_id:
            return
        try:
            idx = int(self.nb.index("current"))
        except Exception:
            return
        if 0 <= idx < len(self.tracks):
            self.rename_track(idx)

    def rename_track(self, idx: int):
        """弹出对话框重命名第 idx 条音轨，并同步标签页文本。"""
        if not (0 <= idx < len(self.tracks)):
            return
        t = self.tracks[idx]

        new = self.ask_string(
            title="重命名音轨",
            tip=f"请输入音轨「{t.name}」的新名称：",
            initial=t.name,
        )
        if new is None:
            return

        new = safe_filename(new, fallback=t.name, maxlen=24)
        if not new or new == t.name:
            return

        old = t.name
        t.name = new

        try:
            self.nb.tab(t.frame, text=new)
        except Exception:
            pass

        sel = self.current_index()
        self.refresh_list(select=sel)

        self.status.config(text=f"音轨「{old}」已重命名为「{new}」")

    def _draw_track(self, track: Track):
        c = track.canvas
        if c is None:
            return
        try:
            c.delete("all")
        except Exception:
            return
        W, H = c.winfo_width(), c.winfo_height()
        if W < 10 or H < 10:
            return

        if track.y is None or len(track.y) == 0:
            c.create_text(W // 2, H // 2, text="（空音轨）",
                          fill="#7f8c9a", font=(FONT, 11))
            return

        y = track.y
        n = len(y)
        mid = H / 2.0
        cols = max(1, min(int(W), 1200))

        absy = np.abs(y)
        edges = np.linspace(0, n, cols + 1).astype(np.int64)
        peaks = np.zeros(cols, dtype=np.float32)
        for i in range(cols):
            a, b = int(edges[i]), int(edges[i + 1])
            if b <= a:
                b = min(a + 1, n)
            if b > a:
                peaks[i] = absy[a:b].max()

        ref = float(self.REFERENCE_PEAK) or 1.0
        scale = (H / 2.0 - 10) / ref
        dx = W / float(cols)

        for i in range(cols):
            a = max(1.0, float(peaks[i]) * scale)
            x = i * dx
            c.create_line(x, mid - a, x, mid + a, fill="#3d8bfd")

        c.create_line(0, mid, W, mid, fill="#2a3a4a")

        if (track.sel_start is not None and track.sel_end is not None
                and track.sel_end > track.sel_start):
            x1 = self._time_to_x(track, track.sel_start)
            x2 = self._time_to_x(track, track.sel_end)

            fillcolor = "#ffb703" if track._drag_mode in ("new",) else "#0088ff"
            outcolor  = "#fb8500" if track._drag_mode in ("new",) else "#004ccf"
        
            c.create_rectangle(x1, 0, x2, H, fill=fillcolor, stipple="gray25",
                               outline=outcolor, width=2)
        
            for x in (x1, x2):
                c.create_line(x, 0, x, H, fill=outcolor, width=2)
                c.create_rectangle(x - 5, 0, x + 5, 10,
                                   fill=outcolor, outline="")
                c.create_rectangle(x - 5, H, x + 5, H - 10,
                                   fill=outcolor, outline="")

    def draw_wave(self, track=None):
        if track is None:
            track = self.cur_track
        if track is None:
            return
        self._draw_track(track)

    def on_canvas_motion(self, e, track):
        if track._drag_mode is not None:
            return
        self._set_cursor(track, self._CURSORS[self._hit_test(track, e.x)])
    
    def _set_cursor(self, track, cur):
        c = track.canvas
        if c is not None and c.cget("cursor") != cur:
            c.config(cursor=cur)

    def on_canvas_press(self, e, track: Track):
        if track.y is None:
            return
        if e.state & 0x0001: # Shift
            mode = "new"
        else:
            mode = self._hit_test(track, e.x)
        track._drag_mode = mode
        track._drag_origin_t = self._x_to_time(track, e.x)
    
        if mode == "new":
            track.sel_start = track.sel_end = track._drag_origin_t
        else:
            track._drag_start0 = track.sel_start
            track._drag_end0 = track.sel_end
    
        track._drag_anchor = track._drag_origin_t
        self._sync_entry_times(track)
        self._draw_track(track)
    
    def on_canvas_drag(self, e, track: Track):
        if track.y is None or track._drag_mode is None:
            return
    
        t = self._x_to_time(track, e.x)
        td = self._track_duration(track)
        mode = track._drag_mode
    
        if mode == "new":
            track.sel_start = min(track._drag_anchor, t)
            track.sel_end = max(track._drag_anchor, t)
    
        elif mode == "left":
            track.sel_start = max(0.0, min(t, track.sel_end - self.MIN_SEL_SEC))
    
        elif mode == "right":
            track.sel_end = min(td, max(t, track.sel_start + self.MIN_SEL_SEC))
    
        elif mode == "move":
            length = track._drag_end0 - track._drag_start0
            dt = t - track._drag_origin_t
            a = track._drag_start0 + dt
            b = track._drag_end0 + dt
            if a < 0.0:
                a, b = 0.0, length
            if b > td:
                b, a = td, td - length
            track.sel_start, track.sel_end = a, b
    
        self._sync_entry_times(track)
        self._draw_track(track)
    
    def on_canvas_release(self, e, track: Track):
        if track.y is None:
            return
    
        mode = track._drag_mode
    
        if mode == "new":
            t = self._x_to_time(track, e.x)
            if abs(t - track._drag_anchor) < 0.005:
                track.sel_start = track.sel_end = None   # 只是点击
    
        track._drag_mode = None
        track._drag_anchor = None
        track._drag_origin_t = None
        track._drag_start0 = None
        track._drag_end0 = None
    
        self._sync_entry_times(track)
        self._draw_track(track)

        if mode in ("left", "right", "move"):
            self._commit_selection_to_syllable(track)

    def on_canvas_scroll(self, e, track: Track):
        if e.delta < 0:
            App.REFERENCE_PEAK *= 1.1
            if App.REFERENCE_PEAK > 1.1 ** App.MAX_SCROLL_TIMES:
                App.REFERENCE_PEAK = 1.1 ** App.MAX_SCROLL_TIMES
        elif e.delta > 0:
            App.REFERENCE_PEAK /= 1.1
            if App.REFERENCE_PEAK < 1 / 1.1 ** App.MAX_SCROLL_TIMES:
                App.REFERENCE_PEAK = 1 / 1.1 ** App.MAX_SCROLL_TIMES
        self.draw_wave()

    def _hit_test(self, track, x):
        """返回 'left' / 'right' / 'move' / 'new'"""
        if (track.sel_start is None or track.sel_end is None
                or track.sel_end <= track.sel_start):
            return "new"

        x1 = self._time_to_x(track, track.sel_start)
        x2 = self._time_to_x(track, track.sel_end)
        d1, d2 = abs(x - x1), abs(x - x2)
        hp = self.HANDLE_PX

        if d1 <= hp and d2 <= hp:
            return "left" if d1 <= d2 else "right"
        if d1 <= hp:
            return "left"
        if d2 <= hp:
            return "right"
        if x1 < x < x2:
            return "move"
        return "new"

    def _sync_entry_times(self, track):
        if track.sel_start is None:
            self.in_var.set("0.000")
            self.out_var.set(f"{track.duration:.3f}")
        else:
            self.in_var.set(f"{track.sel_start:.3f}")
            self.out_var.set(f"{track.sel_end:.3f}")

    def apply_entry_times(self, event=None):
        t = self.cur_track
        if t is None or t.y is None:
            return
        try:
            a = float(self.in_var.get())
            b = float(self.out_var.get())
        except ValueError:
            return
        td = t.duration
        a = max(0.0, min(td, a))
        b = max(0.0, min(td, b))
        if b < a:
            a, b = b, a
        t.sel_start, t.sel_end = a, b
        self.in_var.set(f"{a:.3f}")
        self.out_var.set(f"{b:.3f}")
        self._draw_track(t)

        self._commit_selection_to_syllable(t)

    def apply_spb(self, event=None):
        try:
            spb = float(self.spb_var.get().strip())
        except ValueError:
            self.spb_var.set(f"{self.sec_per_beat:.3f}")
            return
        spb = max(0.01, min(60.0, spb))
        self.history.push("修改一拍时长")
        self.history.commit()
        self.sec_per_beat = spb
        self.spb_var.set(f"{spb:.3f}")
        self.bpm_lbl.config(text=f"= {60.0 / spb:.1f} BPM")

        for s in self.syllables:
            s.sec_per_beat = spb
            s._key = None

        idx = self.current_index()
        if idx is not None and idx < len(self.syllables):
            s = self.syllables[idx]
            self._loading = True
            self.beats_var.set(round(float(s.beats), 3))
            self._loading = False
            self._update_beats_ui(float(s.beats))
            self.update_info()

        self.status.config(
            text=f"一拍 = {spb:.3f}s （{60.0 / spb:.1f} BPM）")

    def _update_beats_ui(self, beats):
        spb = float(self.sec_per_beat)
        self.beats_lbl.config(text=f"{beats:.2f} 拍 = {beats * spb:.3f}s")

    def on_beats_change(self, val=None):
        beats = float(self.beats_var.get())
        self._update_beats_ui(beats)
        if self._loading:
            return
        idx = self.current_index()
        if idx is None:
            return
        s = self.syllables[idx]
        if abs(s.beats - beats) > 1e-6:
            self.history.push("改变时长", key=("beats", id(s)))
        s.beats = max(0.01, beats)
        self.update_info()

    def import_audio(self):
        paths = filedialog.askopenfilenames(
            title="选择音频文件（可多选）",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.wma"),
                       ("所有文件", "*.*")])
        if not paths:
            return

        self.status.config(text="正在解码，请稍候……")
        self.update_idletasks()

        fails = []
        added = 0
        for p in paths:
            try:
                y, sr = AudioUtil.load_audio(p)
            except Exception as e:
                fails.append((p, str(e)))
                continue
            self.add_track(y, sr, p)
            added += 1

        if fails:
            messagebox.showwarning(
                "部分文件加载失败",
                "\n".join(f"{os.path.basename(p)}: {m}" for p, m in fails))

        if added == 0:
            self.status.config(text="就绪")
            return

        self.status.config(
            text=f"已添加 {added} 个音轨   |   当前音轨："
                 f"{self.cur_track.name if self.cur_track else '—'}   |   "
                 f"共 {len(self.tracks)} 个音轨")

    def get_recording(self):
        dlg = RecordDialog(self)
        dlg.wait_window()
        if not dlg.result:
            return
        
        y, sr = dlg.result
        if y.size == 0:
            messagebox.showinfo("提示", "录音为空")
            return

        self.add_track(y, sr, path=None, name=f"录音 {strftime('%Y.%m.%d - %H.%M.%S')}", select=True)
        self.status.config(
            text=f"已录制 {len(y) / float(sr):.2f}s（{sr} Hz）"
                 f"   |   共 {len(self.tracks)} 条音轨")

    def make_tts(self):
        dlg = TTSDialog(self)
        dlg.wait_window()
        if not dlg.result:
            return
    
        y, sr = dlg.result
        if y.size == 0:
            messagebox.showinfo("提示", "合成结果为空")
            return
    
        self.add_track(
            y, sr, path=None,
            name=f"语音 {strftime('%Y.%m.%d - %H.%M.%S')}",
            select=True,
        )
        self.status.config(
            text=f"已合成 {len(y) / float(sr):.2f}s（{sr} Hz）"
                 f"   |   共 {len(self.tracks)} 条音轨")

    def export_project(self):
        if not self.tracks:
            messagebox.showinfo("提示", "还没有导入音频，无法保存工程")
            return

        default = "project"
        first_path = self.tracks[0].path
        if first_path:
            default = safe_filename(
                os.path.splitext(os.path.basename(first_path))[0],
                fallback="project", maxlen=40)

        path = filedialog.asksaveasfilename(
            title="保存工程",
            defaultextension=PROJECT_EXT,
            initialfile=f"{default}{PROJECT_EXT}",
            filetypes=[("活字乱刷工程", f"*{PROJECT_EXT}"),
                       ("所有文件", "*.*")])
        if not path:
            return

        self.status.config(text="正在保存工程……")
        self.update_idletasks()

        try:
            meta = {
                "format": "mtproj",
                "sec_per_beat": float(self.sec_per_beat),
                "tracks": [],
                "syllables": [],
            }

            with ZipFile(path, "w", ZIP_DEFLATED) as z:
                for i, t in enumerate(self.tracks):
                    if t.y is None or t.sr is None:
                        continue
                    entry = f"audio_{i}.wav"
                    buf = BytesIO()
                    sf.write(buf, np.asarray(t.y, dtype=np.float32),
                             int(t.sr), format="WAV", subtype="FLOAT")
                    z.writestr(entry, buf.getvalue())
                    meta["tracks"].append({
                        "file": entry,
                        "name": t.name,
                        "sample_rate": int(t.sr),
                        "samples": int(len(t.y)),
                        "source_path": os.path.basename(t.path or ""),
                    })

                for s in self.syllables:
                    meta["syllables"].append({
                        "char": s.char,
                        "track": int(getattr(s, "track_index", 0)),
                        "start": int(s.start),
                        "end": int(s.end),
                        "beats": round(float(s.beats), 6),
                        "semitones": round(float(s.semitones), 6),
                        "volume_db": round(float(s.volume_db), 6),
                        "keep_formant": bool(s.keep_formant),
                        "flatten": bool(s.flatten),
                        "pitch_offsets": [round(float(x), 6) for x in s.pitch_offsets],
                    })

                z.writestr(PROJECT_META_ENTRY,
                           dumps(meta, ensure_ascii=False, indent=2))

        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存工程：\n{e}")
            self.status.config(text="就绪")
            return

        total_s = sum(t.duration for t in self.tracks)
        self.status.config(
            text=f"工程已保存：{os.path.basename(path)}   |   "
                 f"{len(self.tracks)} 条音轨   |   "
                 f"{len(self.syllables)} 个字   |   共 {total_s:.2f}s 音频")

    def import_project(self):
        path = filedialog.askopenfilename(
            title="打开工程",
            filetypes=[("活字乱刷工程", f"*{PROJECT_EXT}"),
                       ("所有文件", "*.*")])
        if not path:
            return

        self.status.config(text="正在打开工程……")
        self.update_idletasks()

        try:
            with ZipFile(path, "r") as z:
                names = set(z.namelist())
                if PROJECT_META_ENTRY not in names:
                    raise ValueError("工程内缺少 project.json")
                meta = loads(z.read(PROJECT_META_ENTRY).decode("utf-8"))

                track_infos = meta.get("tracks")
                loaded = []   # (name, y, sr, source_path)

                for i, ti in enumerate(track_infos):
                    if not isinstance(ti, dict):
                        continue
                    entry = ti.get("file")
                    if not entry or entry not in names:
                        cands = sorted(
                            n for n in names
                            if re.match(rf"^audio_{i}\.", n)
                            or n.lower().endswith(
                                (".wav", ".flac", ".ogg", ".aiff", ".aif")))
                        entry = cands[0] if cands else None
                    if not entry or entry not in names:
                        raise ValueError(f"工程内找不到第 {i} 条音轨的音频")
                    y, sr = AudioUtil.decode_audio_bytes(z.read(entry))
                    loaded.append((
                        str(ti.get("name") or f"音轨 {i + 1}"),
                        y, sr,
                        ti.get("source_path") or None,
                    ))

        except Exception as e:
            messagebox.showerror("打开失败", f"无法读取工程：\n{e}")
            self.status.config(text="就绪")
            return

        if not loaded:
            messagebox.showerror("打开失败", "工程内没有可用音轨")
            self.status.config(text="就绪")
            return

        try:
            spb = float(meta.get("sec_per_beat", self.sec_per_beat))
        except (TypeError, ValueError):
            spb = self.sec_per_beat
        spb = max(0.01, min(60.0, spb))
        self.sec_per_beat = spb
        self.spb_var.set(f"{spb:.3f}")
        self.bpm_lbl.config(text=f"= {60.0 / spb:.1f} BPM")

        self._clear_tracks()
        self.syllables = []

        for name, y, sr, src in loaded:
            self.add_track(y, sr, src, name=name, select=False)

        if self.tracks:
            self._active_track = 0
            try:
                self.nb.select(self.tracks[0].frame)
            except Exception:
                pass

        for item in (meta.get("syllables") or []):
            if not isinstance(item, dict):
                continue
            try:
                ti = int(item.get("track", 0))
            except (TypeError, ValueError):
                ti = 0
            if ti < 0 or ti >= len(self.tracks):
                continue
            t = self.tracks[ti]

            try:
                a = int(item.get("start", 0))
                b = int(item.get("end", 0))
            except (TypeError, ValueError):
                continue
            n = t.n_samples
            a = max(0, min(a, n))
            b = max(0, min(b, n))
            if b <= a:
                continue

            ch = str(item.get("char", "") or "?").strip() or "?"
            s = Syllable(ch, t.y, t.sr, a, b,
                         sec_per_beat=self.sec_per_beat, track_index=ti)
            try:
                s.beats = max(0.01, float(item.get("beats", s.beats)))
                s.semitones = float(item.get("semitones", 0.0))
                s.volume_db = float(item.get("volume_db", 0.0))
            except (TypeError, ValueError):
                pass
            s.keep_formant = bool(item.get("keep_formant", True))
            s.flatten = bool(item.get("flatten", False))
            offs = item.get("pitch_offsets")
            if isinstance(offs, (list, tuple)) and len(offs) == s.N_ENV:
                s.pitch_offsets = [float(x) for x in offs]

            self.syllables.append(s)

        self.in_var.set("0.000")
        if self.cur_track is not None:
            self.out_var.set(f"{self.cur_track.duration:.3f}")
        else:
            self.out_var.set("0.000")

        self.refresh_list(select=0 if self.syllables else None)
        self.on_tab_changed()

        total_s = sum(t.duration for t in self.tracks)

        self.history.clear()
        self._update_history_ui()

        self.status.config(
            text=f"已打开工程：{os.path.basename(path)}   |   "
                 f"{len(self.tracks)} 条音轨   |   "
                 f"{len(self.syllables)} 个字   |   "
                 f"共 {total_s:.2f}s   |   1 拍 {self.sec_per_beat:.3f}s")

    def play_audio(self):
        t = self.cur_track
        if t is None or t.y is None:
            messagebox.showinfo("提示", "请先导入音频")
            return
    
        has_sel = (t.sel_start is not None and t.sel_end is not None
                   and t.sel_end - t.sel_start >= 0.005)
    
        if has_sel:
            a = int(t.sel_start * t.sr)
            b = int(t.sel_end * t.sr)
            y = t.y[a:b]
        else:
            y = t.y
    
        if y is None or len(y) == 0:
            messagebox.showinfo("提示", "没有可播放的音频")
            return
    
        self.play_btn.config(text="■ 停止",
                             bootstyle=(DANGER, OUTLINE),
                             command=AsyncPlayer.stop_playback)
        self.update()
    
        def _restore():
            self.play_btn.config(text="▶ 播放",
                                 bootstyle=(SUCCESS, OUTLINE),
                                 command=self.play_audio)
            self.update()
    
        AsyncPlayer.play_array(y, t.sr, after=_restore)

    def extract_syllable(self):
        t = self.cur_track
        if t is None or t.y is None:
            messagebox.showinfo("提示", "请先导入音频")
            return
        if (t.sel_start is None or t.sel_end is None
                or t.sel_end - t.sel_start < 0.005):
            messagebox.showinfo("提示", "请先在波形上拖拽出一段选区")
            return

        n = len(t.y)
        a = int(round(t.sel_start * t.sr))
        b = int(round(t.sel_end * t.sr))
        a = max(0, min(a, n))
        b = max(0, min(b, n))
        if b <= a:
            messagebox.showinfo("提示", "选区太短，无法剪出")
            return

        ch = self.ask_char(
            f"选中 {t.sel_end - t.sel_start:.3f} 秒，请输入对应的单字：")
        if not ch:
            return

        self.history.push("剪出单字")
        self.history.commit()
        syl = Syllable(ch, t.y, t.sr, a, b,
                       sec_per_beat=self.sec_per_beat,
                       track_index=self._active_track)
        syl.beats = float(self.DEFAULT_BEATS)
        self.syllables.append(syl)
        self.refresh_list(select=len(self.syllables) - 1)
        self.status.config(
            text=f"已剪出单字「{ch}」（音轨「{t.name}」）"
                 f" 原始 {syl.orig_dur:.3f}s = {syl.beats:.2f} 拍"
                 f"（一拍 {self.sec_per_beat:.3f}s）")

    def ask_char(self, tip="请输入单字："):
        return self.ask_string("输入单字", tip)

    def ask_string(self, title="输入", tip="请输入：", initial=""):
        """通用单行文本输入对话框；取消返回 None。"""
        dlg = Toplevel(title=title)
        dlg.transient(self)
        dlg.resizable(False, False)

        Label(dlg, text=tip, font=(FONT, 10)).pack(padx=24, pady=(18, 8))

        var = StringVar(value=str(initial or ""))
        ent = Entry(dlg, textvariable=var, width=26, justify=CENTER,
                       font=(FONT, 14))
        ent.pack(padx=24)
        ent.focus_set()
        ent.select_range(0, END)

        result = {"v": None}

        def ok(event=None):
            result["v"] = var.get().strip()
            dlg.destroy()

        btns = Frame(dlg)
        btns.pack(pady=16)
        Button(btns, text="确定", bootstyle=PRIMARY, width=8,
                  command=ok).pack(side=LEFT, padx=4)
        Button(btns, text="取消", bootstyle=(SECONDARY, OUTLINE), width=8,
                  command=dlg.destroy).pack(side=LEFT, padx=4)
        ent.bind("<Return>", ok)

        dlg.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - dlg.winfo_height()) // 3
        dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        dlg.grab_set()
        self.wait_window(dlg)
        return result["v"]

    def refresh_list(self, select=None):
        self.listbox.delete(0, END)
        for s in self.syllables:
            self.listbox.insert(END, s.char)
        if select is not None and 0 <= select < len(self.syllables):
            self.listbox.selection_clear(0, END)
            self.listbox.selection_set(select)
            self.listbox.see(select)
        self.on_select()

    def delete_syllable(self):
        idx = self.current_index()
        if idx is None:
            return
        if messagebox.askyesno("确认", "确定要删除此单字？"):
            self.history.push("删除单字")
            self.history.commit()
            s = self.syllables.pop(idx)
            self.refresh_list(select=min(idx, len(self.syllables) - 1)
                              if self.syllables else None)
            self.status.config(text=f"已删除「{s.char}」")

    def move_syllable(self, delta):
        idx = self.current_index()
        if idx is None:
            return
        new = idx + delta
        if not (0 <= new < len(self.syllables)):
            return
        self.history.push("移动单字")
        self.history.commit()
        self.syllables[idx], self.syllables[new] = \
            self.syllables[new], self.syllables[idx]
        self.refresh_list(select=new)

    def rename_current(self):
        idx = self.current_index()
        if idx is None:
            messagebox.showinfo("提示", "请先选择一个单字")
            return
        ch = self.ask_char("请输入新的单字：")
        if not ch:
            return
        self.history.push("重命名单字")
        self.history.commit()
        self.syllables[idx].char = ch
        self.refresh_list(select=idx)

    def on_select(self, event=None):
        idx = self.current_index()
        if idx is None or idx >= len(self.syllables):
            self.char_label.config(text="—")
            self.info_label.config(text="请在左侧列表中选择一个单字")
            return

        s = self.syllables[idx]
        self._show_syllable_on_track(s)
        self.char_label.config(text=s.char)

        self._loading = True

        beats = float(s.beats)
        top = max(8.0, beats * 1.5)
        if abs(float(self.beats_scale.cget("to")) - top) > 1e-6:
            self.beats_scale.config(to=top)
        self.beats_var.set(round(beats, 3))
        self._update_beats_ui(beats)

        self.pitch_var.set(round(s.semitones))
        self.pitch_lbl.config(text=f"{int(round(s.semitones)):+d} 半音")
        self.vol_var.set(round(s.volume_db, 1))
        self.vol_lbl.config(text=f"{s.volume_db:+.1f} dB")
        self.keep_formant_var.set(bool(s.keep_formant))
        self.flatten_var.set(bool(s.flatten))

        self._loading = False

        self.history.commit()
        self.update_info()
        self._update_note_env_btn(s)

    def _show_syllable_on_track(self, s):
        ti = int(getattr(s, "track_index", 0))
        if not (0 <= ti < len(self.tracks)):
            return
        track = self.tracks[ti]
    
        if self._active_track != ti:
            self._suppress_tab = True
            try:
                self.nb.select(track.frame)
            except Exception:
                pass
            finally:
                self._suppress_tab = False
            self._active_track = ti
    
        sr = int(s.sr or track.sr or 1)
        track.sel_start = s.start / float(sr)
        track.sel_end   = s.end   / float(sr)
    
        track._drag_mode = None
        track._drag_anchor = None
    
        self._sync_entry_times(track)
        self._draw_track(track)

    def _commit_selection_to_syllable(self, track):
        if track is None or track is not self.cur_track:
            return
        idx = self.current_index()
        if idx is None or idx >= len(self.syllables):
            return
        s = self.syllables[idx]
        if int(getattr(s, "track_index", 0)) != self._active_track:
            return
        if track.sel_start is None or track.sel_end is None:
            return
        if track.sel_end - track.sel_start < self.MIN_SEL_SEC:
            return
    
        sr = int(s.sr or track.sr or 1)
        n = len(s._audio) if s._audio is not None else 0
    
        a = int(round(track.sel_start * sr))
        b = int(round(track.sel_end   * sr))
        a = max(0, min(a, n))
        b = max(0, min(b, n))
        if b <= a:
            return
    
        if (a, b) == (int(s.start), int(s.end)):
            return
        
        self.history.push("调整选区")
        self.history.commit()
        s.start, s.end = a, b
        s._key = None
        s._out = None
        self.update_info()

    def update_info(self):
        idx = self.current_index()
        if idx is None or idx >= len(self.syllables):
            self.info_label.config(text="请在左侧列表中选择一个单字")
            return
        s = self.syllables[idx]
        tone = "保持音色" if s.keep_formant else "改变音色"
        flat = " · 音调校准" if s.flatten else ""

        env = ""
        if s.has_note_env():
            env = "   分段 " + "/".join(f"{int(x):+d}" for x in s.pitch_offsets)

        self.info_label.config(
            text=f"{s.orig_dur:.3f}s  →  {s.beats:.2f} 拍 = {s.cur_dur:.3f}s"
                 f"   {s.sr} Hz   {tone}{flat}    {s.volume_db:+.1f} dB"
                 f"{env}")

    def on_pitch_change(self, val=None):
        v = int(round(float(self.pitch_var.get())))
        self.pitch_lbl.config(text=f"{v:+d} 半音")
        if self._loading:
            return
        idx = self.current_index()
        if idx is None:
            return
        s = self.syllables[idx]
        if abs(s.semitones - float(v)) > 1e-6:
            self.history.push("改变音高", key=("pitch", id(s)))
        s.semitones = float(v)
        self.update_info()

    def edit_note_env(self):
        idx = self.current_index()
        if idx is None:
            messagebox.showinfo("提示", "请先选择一个单字")
            return
        s = self.syllables[idx]
        
        self.history.push("分段调音")
    
        dlg = NoteEnvDialog(self, s)
        dlg.wait_window()

        if dlg.result is None:
            self.history.cancel_last()
        else:
            self.history.commit()
    
        self._update_note_env_btn(s)
        self.update_info()
    
        if dlg.result is None:
            return
        if s.has_note_env():
            self.status.config(
                text=f"「{s.char}」多音符包络：" +
                     " ".join(f"{int(x):+d}" for x in s.pitch_offsets))
        else:
            self.status.config(text=f"「{s.char}」已恢复单音符")

    def _update_note_env_btn(self, s: Syllable):
        if s is None:
            self.note_env_btn.config(bootstyle=(INFO, OUTLINE))
        elif s.has_note_env():
            self.note_env_btn.config(bootstyle=INFO)
        else:
            self.note_env_btn.config(bootstyle=(INFO, OUTLINE))

    def on_vol_change(self, val=None):
        v = float(self.vol_var.get())
        self.vol_lbl.config(text=f"{v:+.1f} dB")
        if self._loading:
            return
        idx = self.current_index()
        if idx is None:
            return
        s = self.syllables[idx]
        if abs(s.volume_db - v) > 1e-6:
            self.history.push("改变音量", key=("vol", id(s)))
        s.volume_db = v
        self.update_info()

    def set_volume(self, db):
        idx = self.current_index()
        if idx is None:
            return
        self._loading = True
        self.vol_var.set(float(db))
        self.vol_lbl.config(text=f"{float(db):+.1f} dB")
        self._loading = False
        self.syllables[idx].volume_db = float(db)
        self.update_info()

    def on_formant_toggle(self):
        if self._loading:
            return
        idx = self.current_index()
        if idx is None:
            return
        s = self.syllables[idx]
        new = bool(self.keep_formant_var.get())
        if s.keep_formant != new:
            self.history.push("切换音色")
            self.history.commit()
            s.keep_formant = new
        self.update_info()

    def on_flatten_toggle(self):
        if self._loading:
            return
        idx = self.current_index()
        if idx is None:
            return
        s = self.syllables[idx]
        new = bool(self.flatten_var.get())
        if s.flatten != new:
            self.history.push("切换音调校准")
            self.history.commit()
            s.flatten = new
        self.update_info()

    def reset_current(self):
        idx = self.current_index()
        if idx is None:
            return
        s = self.syllables[idx]
        self.history.push("重置单字")
        self.history.commit()
        s.reset()

        self._loading = True
        beats = float(s.beats)
        top = max(8.0, beats * 1.5)
        self.beats_scale.config(to=top)
        self.beats_var.set(round(beats, 3))
        self._update_beats_ui(beats)

        self.pitch_var.set(0.0)
        self.pitch_lbl.config(text="+0 半音")
        self.vol_var.set(0.0)
        self.flatten_var.set(False)
        self.vol_lbl.config(text="+0.0 dB")
        self.keep_formant_var.set(True)
        self._loading = False

        self.update_info()
        self._update_note_env_btn(s)
        self.status.config(text=f"已重置「{s.char}」")

    def preview(self):
        idx = self.current_index()
        if idx is None:
            messagebox.showinfo("提示", "请先在左侧列表中选择一个单字")
            return
        s = self.syllables[idx]
        self.status.config(text=f"正在处理「{s.char}」……")
        self.update_idletasks()
        try:
            y = s.process()
        except Exception as e:
            messagebox.showerror("处理失败", str(e))
            self.status.config(text="就绪")
            return
        AsyncPlayer.play_array(y, s.sr)
        tone = "保持音色" if s.keep_formant else "改变音色"
        self.status.config(
            text=f"预听「{s.char}」 {s.beats:.2f} 拍 = {len(y)/s.sr:.3f}s  "
                 f"音高 {s.semitones:+.0f} 半音  音量 {s.volume_db:+.1f} dB  ({tone})")

    def export_current(self):
        idx = self.current_index()
        if idx is None:
            messagebox.showinfo("提示", "请先选择一个单字")
            return
        s = self.syllables[idx]
        path = filedialog.asksaveasfilename(
            title="导出单字", defaultextension=".wav",
            initialfile=f"{safe_filename(s.char)}.wav",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")])
        if not path:
            return
        self.status.config(text="正在导出……")
        self.update_idletasks()
        try:
            sf.write(path, s.process(), s.sr)
        except Exception as e:
            messagebox.showerror("导出失败", str(e))
            self.status.config(text="就绪")
            return
        self.status.config(text=f"已导出：{path}")

    def concat_export(self):
        if not self.syllables:
            messagebox.showinfo("提示", "单字列表为空，请先剪出单字")
            return

        # target_sr = int(self.syllables[0].sr)
        target_sr = int(max([s.sr for s in self.syllables]))

        self.status.config(text="正在处理并连接……")
        self.update_idletasks()

        pieces = []
        try:
            for i, s in enumerate(self.syllables):
                y = np.asarray(s.process(), dtype=np.float32).ravel()
                if int(s.sr) != target_sr:
                    y = resample(y, orig_sr=int(s.sr),
                                         target_sr=target_sr).astype(np.float32)
                pieces.append(y)

            full = (np.concatenate(pieces) if pieces
                    else np.zeros(0, dtype=np.float32))
        except Exception as e:
            messagebox.showerror("处理失败", f"拼接时出错：\n{e}")
            self.status.config(text="就绪")
            return

        peak = float(np.max(np.abs(full))) if full.size else 0.0
        if peak > 1.0:
            full = (full / peak) * 0.99

        if full.size == 0:
            messagebox.showinfo("提示", "拼接结果为空")
            self.status.config(text="就绪")
            return

        default_name = safe_filename(
            "".join(s.char for s in self.syllables), fallback="joined", maxlen=40)
        path = filedialog.asksaveasfilename(
            title="连接导出",
            defaultextension=".wav",
            initialfile=f"{default_name}.wav",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")])
        if not path:
            self.status.config(text="就绪")
            return

        try:
            sf.write(path, full, target_sr)
        except Exception as e:
            messagebox.showerror("导出失败", str(e))
            self.status.config(text="就绪")
            return

        total_s = len(full) / float(target_sr)
        total_beats = sum(float(s.beats) for s in self.syllables)
        self.status.config(
            text=f"已连接导出：{os.path.basename(path)}   |   "
                 f"{len(self.syllables)} 个字   |   共 {total_beats:.2f} 拍   |   "
                 f"总时长 {total_s:.2f}s   |   "
                 f"1 拍 {self.sec_per_beat:.3f}s   |   {target_sr} Hz")

    def _snapshot_state(self):
        return {
            "sec_per_beat": float(self.sec_per_beat),
            "active_track": int(self._active_track),
            "selected": self.current_index(),
            "tracks": [
                (t, (t.name, t.sel_start, t.sel_end))
                for t in self.tracks
            ],
            "syllables": [
                (s, (
                    s.char, s.sr, s.start, s.end, s.sec_per_beat,
                    s.flatten, s.track_index, tuple(s.pitch_offsets),
                    s.beats, s.semitones, s.volume_db, s.keep_formant,
                ))
                for s in self.syllables
            ],
        }

    def _restore_state(self, snap):
        self.sec_per_beat = snap["sec_per_beat"]
        self._active_track = snap["active_track"]
        self.spb_var.set(f"{self.sec_per_beat:.3f}")
        self.bpm_lbl.config(text=f"= {60.0 / self.sec_per_beat:.1f} BPM")
    
        self.tracks = [t for t, _ in snap["tracks"]]
        for t, (name, ss, se) in snap["tracks"]:
            t.name = name
            t.sel_start = ss
            t.sel_end = se
    
        self.syllables = [s for s, _ in snap["syllables"]]
        for s, (char, sr, start, end, spb, flat, ti,
                poff, beats, semi, vol, kf) in snap["syllables"]:
            s.char = char
            s.sr = int(sr)
            s.start = int(start)
            s.end = int(end)
            s.sec_per_beat = spb
            s.flatten = flat
            s.track_index = ti
            s.pitch_offsets = list(poff)
            s.beats = beats
            s.semitones = semi
            s.volume_db = vol
            s.keep_formant = kf
            s._key = None
            s._out = None
    
        for i, t in enumerate(self.tracks):
            try:
                self.nb.tab(t.frame, text=t.name)
            except Exception:
                pass
    
        self.refresh_list(select=snap["selected"]
                          if snap["selected"] is not None
                          and snap["selected"] < len(self.syllables) else None)
        self.on_tab_changed()
        self.on_select()
        for t in self.tracks:
            self._draw_track(t)

    def _on_close(self):
        try:
            AsyncPlayer.stop_playback()
        except Exception:
            pass
        self.destroy()

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    filterwarnings("ignore", category=FutureWarning)
    filterwarnings("ignore", category=RuntimeWarning, module="numba")
    main()