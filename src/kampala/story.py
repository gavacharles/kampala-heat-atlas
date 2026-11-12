"""Unit-dot story media in the Kiira Futures Lab house style (kampala-last-mile-map,
scripts/55_kiira_story_media.py, 2026-11-12): one hook, one number and one sub-line per beat,
dots that glide from one picture to the next, light background, Arial, credited Gava Charles.

Each story writes social_<slug>/ with clips_square/, clips_vertical/ (one clip per beat),
reel_square.mp4, reel_vertical.mp4, carousel/ (cover + one card per beat + end card, and a PDF),
film_<slug>.mp4 (1280 x 720) and highlight.gif (the square reel at 2x, 720 px).

Difference from the original: the moving figure ('stage') is drawn here in Python instead of
being captured from a web page with Selenium, so no browser is needed.
"""
import subprocess
from dataclasses import dataclass, field

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FR = "/System/Library/Fonts/Supplemental/Arial.ttf"
font = lambda s, b=False: ImageFont.truetype(FB if b else FR, s)
BG, INK, INK2, BLUE, MUT = (246, 246, 243), (11, 11, 11), (82, 81, 78), (42, 120, 214), (137, 135, 129)
LAB = "KAMPALA HEAT ATLAS"
STAGE_W, STAGE_H = 1260, 900
HOLD, GLIDE, FPS = 3.2, 2.6, 30


@dataclass
class Beat:
    id: str
    hook: str
    stat: str
    sub: str
    xy: np.ndarray                    # (N, 2) dot centres in stage pixels
    rgb: np.ndarray                   # (N, 3) 0-255
    radius: float = 4.0
    legend: list = field(default_factory=list)      # [(rgb, label)]
    labels: list = field(default_factory=list)      # [(x, y, text, anchor, size, bold)]
    background: Image.Image | None = None           # optional faint picture behind the dots
    alpha: np.ndarray | None = None                 # (N,) 0-1 dot opacity


def smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


class Story:
    def __init__(self, slug, title, intro, link, beats):
        self.slug, self.title, self.intro, self.link, self.beats = slug, title, intro, link, beats
        # timeline: beat i settles at s_i and holds for HOLD seconds; transitions take GLIDE seconds
        self.marks, t = [], 0.0
        for i in range(len(beats)):
            start = t
            if i:
                t += GLIDE
            settled = t
            t += HOLD
            self.marks.append((start, settled, t))
        self.T = t
        self._cache = {}

    # ---------------------------------------------------------------- the moving figure
    def _state(self, t):
        for i, (t0, ts, t1) in enumerate(self.marks):
            if t < t1 or i == len(self.marks) - 1:
                if i and t < ts:
                    return i - 1, i, smooth((t - t0) / GLIDE)
                return i, i, 1.0
        return len(self.beats) - 1, len(self.beats) - 1, 1.0

    def stage(self, t):
        key = round(t * FPS)
        if key in self._cache:
            return self._cache[key]
        a, b, u = self._state(t)
        A, B = self.beats[a], self.beats[b]
        im = Image.new("RGB", (STAGE_W, STAGE_H), BG)
        for bg, w in ((A.background, 1 - u), (B.background, u)):
            if bg is not None and w > 0.02:
                im = Image.blend(im, Image.alpha_composite(Image.new("RGBA", im.size, BG + (255,)), bg).convert("RGB"), 0.9 * w)
        dr = ImageDraw.Draw(im, "RGBA")
        xy = A.xy * (1 - u) + B.xy * u
        rgb = (A.rgb * (1 - u) + B.rgb * u).astype(int)
        al = ((A.alpha if A.alpha is not None else np.ones(len(xy))) * (1 - u) +
              (B.alpha if B.alpha is not None else np.ones(len(xy))) * u)
        r = A.radius * (1 - u) + B.radius * u
        for (x, y), c, aa in zip(xy, rgb, al):
            if aa > 0.02:
                dr.ellipse((x - r, y - r, x + r, y + r), fill=(int(c[0]), int(c[1]), int(c[2]), int(255 * aa)))
        # labels and legend belong to the nearer beat and fade through the transition
        for beat, w in ((A, 1 - u), (B, u)):
            if w < 0.5 and a != b:
                continue
            fade = 1.0 if a == b else min(1, abs(u - 0.5) / 0.3)
            col = lambda c_: tuple(int(BG[i] + (c_[i] - BG[i]) * fade) for i in range(3))
            x0 = 10
            for c_, lab in beat.legend:
                dr.ellipse((x0, 14, x0 + 16, 30), fill=col(tuple(c_)))
                dr.text((x0 + 24, 12), lab, font=font(20), fill=col(INK2))
                x0 += 24 + dr.textlength(lab, font=font(20)) + 26
            for x, y, text, anchor, size, bold in beat.labels:
                dr.text((x, y), text, font=font(size, bold), fill=col(INK2), anchor=anchor)
        self._cache[key] = im
        return im

    # ---------------------------------------------------------------- house-style card (as in Kiira)
    @staticmethod
    def _wrap(dr, text, f, width):
        words, lines, cur = text.split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if dr.textlength(t, font=f) <= width:
                cur = t
            else:
                lines.append(cur); cur = w
        return lines + [cur]

    def card(self, stage, W, H, beat, n, a=1.0):
        total = len(self.beats)
        im = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(im, "RGBA")
        pad = 56 if W == H or H < W * 1.4 else 64
        vertical = H > W * 1.4
        ts = 58 if vertical else 50
        y = 70 if vertical else 44
        dr.text((pad, y), f"{n:02d} / {total:02d}  ·  {self.title.upper()}  ·  {LAB}", font=font(22 if vertical else 19, True), fill=MUT)
        y += 44 if vertical else 36
        col = tuple(int(BG[i] + (INK[i] - BG[i]) * a) for i in range(3))
        for line in self._wrap(dr, beat.hook, font(ts, True), W - 2 * pad):
            dr.text((pad, y), line, font=font(ts, True), fill=col); y += ts + 8
        top = y + (40 if vertical else 14)
        foot = 330 if vertical else 170
        fw, fh = W - 2 * (pad - 24), H - top - foot
        s = min(fw / stage.width, fh / stage.height)
        st = stage.resize((int(stage.width * s), int(stage.height * s)), Image.LANCZOS)
        im.paste(st, ((W - st.width) // 2, top + (fh - st.height) // 2))
        yb = H - foot + (30 if vertical else 8)
        colb = tuple(int(BG[i] + (BLUE[i] - BG[i]) * a) for i in range(3))
        sf = font(64 if vertical else 46, True)
        for line in self._wrap(dr, beat.stat, sf, W - 2 * pad)[:2]:
            dr.text((pad, yb), line, font=sf, fill=colb); yb += (72 if vertical else 52)
        col2 = tuple(int(BG[i] + (INK2[i] - BG[i]) * a) for i in range(3))
        dr.text((pad, yb + 2), beat.sub, font=font(30 if vertical else 24), fill=col2)
        dr.line((pad, H - (78 if vertical else 50), W - pad, H - (78 if vertical else 50)), fill=(225, 224, 217), width=2)
        dr.text((pad, H - (40 if vertical else 22)), self.link, font=font(24 if vertical else 19, True), fill=BLUE, anchor="ls")
        dr.text((W - pad, H - (40 if vertical else 22)), "Gava Charles", font=font(24 if vertical else 19), fill=MUT, anchor="rs")
        return im

    def encode(self, path, W, H, segments):
        p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        for t0, t1, bi, switch in segments:
            for f in range(int((t1 - t0) * FPS)):
                t = t0 + f / FPS
                j = bi if t >= switch else bi - 1
                a = min(1, abs(t - switch) / 0.35) if bi > 0 else 1
                p.stdin.write(self.card(self.stage(t), W, H, self.beats[max(j, 0)], max(j, 0) + 1, a).tobytes())
        p.stdin.close(); p.wait()
        print("saved", path.name, flush=True)

    # ---------------------------------------------------------------- the full media set
    def render(self, out_root):
        out = out_root / f"social_{self.slug}"
        for d_ in ("clips_square", "clips_vertical", "carousel"):
            (out / d_).mkdir(parents=True, exist_ok=True)
        M = self.marks
        for i, b in enumerate(self.beats):
            t0, ts, t1 = M[i]
            seg = [(max(0, t0 - 1.0) if i else t0, t1, i, (t0 + ts) / 2 if i else 0)]
            self.encode(out / "clips_square" / f"S{i + 1:02d}_{b.id}.mp4", 1080, 1080, seg)
            self.encode(out / "clips_vertical" / f"S{i + 1:02d}_{b.id}.mp4", 1080, 1920, seg)
        reel = [(M[0][0], M[0][2], 0, 0)] + [(m[0], m[2], i, (m[0] + m[1]) / 2) for i, m in enumerate(M) if i]
        self.encode(out / "reel_square.mp4", 1080, 1080, reel)
        self.encode(out / "reel_vertical.mp4", 1080, 1920, reel)
        # carousel
        pages = []
        cov = Image.new("RGB", (1080, 1350), BG); dr = ImageDraw.Draw(cov)
        dr.text((64, 90), f"{LAB} · 3D URBAN CLIMATE MODEL", font=font(24, True), fill=MUT)
        y = 150
        for line in self._wrap(dr, self.title, font(84, True), 950):
            dr.text((64, y), line, font=font(84, True), fill=INK); y += 94
        y += 20
        for line in self._wrap(dr, self.intro + " Swipe →", font(34), 950):
            dr.text((64, y), line, font=font(34), fill=INK2); y += 44
        st0 = self.stage(M[0][2] - 0.1)
        s = min(950 / st0.width, (1350 - y - 160) / st0.height)
        st0 = st0.resize((int(st0.width * s), int(st0.height * s)), Image.LANCZOS)
        cov.paste(st0, ((1080 - st0.width) // 2, y + 30))
        dr.text((64, 1300), "By Gava Charles", font=font(26, True), fill=BLUE, anchor="ls")
        pages.append(cov)
        for i, (b, m) in enumerate(zip(self.beats, M)):
            pages.append(self.card(self.stage(m[2] - 0.1), 1080, 1350, b, i + 1))
        end = Image.new("RGB", (1080, 1350), BG); dr = ImageDraw.Draw(end)
        dr.text((540, 560), "Explore it yourself", font=font(72, True), fill=INK, anchor="mm")
        dr.text((540, 650), self.link, font=font(30, True), fill=BLUE, anchor="mm")
        dr.text((540, 720), "maps, the 3D model and every figure", font=font(30), fill=INK2, anchor="mm")
        dr.text((540, 1280), "Gava Charles · PhD research from open data; first-pass results", font=font(22), fill=MUT, anchor="mm")
        pages.append(end)
        for i, pg in enumerate(pages):
            pg.save(out / "carousel" / f"C{i + 1:02d}.png")
        pages[0].save(out / "carousel" / f"{self.slug}_carousel.pdf", save_all=True, append_images=pages[1:], resolution=150)
        self.encode(out / f"film_{self.slug}.mp4", 1280, 720, reel)
        vf = ("setpts=PTS/2,fps=12,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(out / "reel_square.mp4"),
                        "-vf", vf, str(out / "highlight.gif")], check=True)
        print(self.slug, "done:", round((out / "highlight.gif").stat().st_size / 1e6, 1), "MB gif", flush=True)
        return out
