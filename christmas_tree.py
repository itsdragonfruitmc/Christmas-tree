"""
Christmas tree made of text characters, drawn as a spinning 3D point cloud
in a Tkinter window.  Only needs numpy (Tkinter ships with Python).

    py -m pip install numpy
    py christmas_tree_tk.py

Controls:  SPACE = pause   R = replay the build-up   ESC = quit
"""
import math
import numpy as np

W = H = 805
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- geometry --
TREE_H, TREE_R, BASE_Y, TIERS = 2.5, 1.15, -1.2, 5
CAM_DIST, FOCAL, TILT = 9.0, 2250.0 * 805 / 900, 0.22

GOLD = (255, 205, 70)
RED = (230, 45, 50)
BAUBLES = [RED, GOLD, (70, 130, 255), (255, 125, 190), (255, 150, 40), (170, 90, 230)]

# angle(deg), distance from trunk, (sx, sy, sz), yaw(deg), box color, ribbon color, delay
GIFTS = [
    (200, 1.20, (.55, .42, .50), 25, (150, 100, 210), GOLD, 0.00),
    (250, 1.05, (.42, .55, .42), -15, (60, 170, 90), RED, 0.25),
    (320, 1.30, (.60, .40, .50), 40, RED, GOLD, 0.50),
    (10, 1.20, (.45, .35, .45), 10, (60, 100, 230), RED, 0.75),
    (80, 1.25, (.50, .45, .50), -30, (235, 120, 40), GOLD, 1.00),
    (140, 1.10, (.38, .30, .38), 20, RED, GOLD, 1.25),
]


def rot_y(p, a):
    c, s = math.cos(a), math.sin(a)
    return p @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T


def tree_radius(h):
    """Radius of the tree at relative height h (0 bottom .. 1 top), with drooping tiers."""
    frac = (h * TIERS) % 1.0
    return TREE_R * (1 - h) * (0.62 + 0.38 * (1 - frac))


def sphere(center, radius, n):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(center) + v * radius


def box_surface(size, n):
    """Random points on the surface of a box centered at the origin, plus face normals."""
    sx, sy, sz = size
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
    face = rng.choice(6, n, p=areas / areas.sum())
    pts = rng.uniform(-.5, .5, (n, 3)) * np.array(size)
    nrm = np.zeros((n, 3))
    for f in range(6):
        m = face == f
        axis, sign = f // 2, (1 if f % 2 else -1)
        pts[m, axis] = sign * size[axis] / 2
        nrm[m, axis] = sign
    return pts, nrm


class Cloud:
    """Collects points (position, color, time-it-appears)."""

    def __init__(self):
        self._p, self._c, self._t = [], [], []

    def add(self, pts, color, t0):
        pts = np.asarray(pts, float).reshape(-1, 3)
        n = len(pts)
        self._p.append(pts)
        self._c.append(np.broadcast_to(np.asarray(color, float), (n, 3)).copy())
        self._t.append(np.broadcast_to(np.asarray(t0, float), (n,)).copy())

    def finish(self):
        self.pos = np.concatenate(self._p)
        self.col = np.clip(np.concatenate(self._c), 0, 255)
        self.t0 = np.concatenate(self._t)
        return self


def build_scene():
    cloud = Cloud()

    # ground disc (snowy white)
    n = 34000
    r = 1.8 * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * np.pi
    pts = np.stack([r * np.cos(a), np.full(n, BASE_Y - 0.02), r * np.sin(a)], 1)
    shade = 190 + rng.random((n, 1)) * 60
    cloud.add(pts, np.hstack([shade * .85, shade * .93, shade]), r * 0.35)

    # trunk
    n = 120
    a = rng.random(n) * 2 * np.pi
    pts = np.stack([.13 * np.cos(a), BASE_Y + rng.random(n) * .3, .13 * np.sin(a)], 1)
    cloud.add(pts, (110, 70, 40), 0.2)

    # tree body: mostly on the surface, grows from the bottom up
    n = 22000
    h = rng.random(n) ** 0.85
    a = rng.random(n) * 2 * np.pi
    r = tree_radius(h) * (1 - rng.random(n) ** 3 * 0.85)
    pts = np.stack([r * np.cos(a), BASE_Y + h * TREE_H, r * np.sin(a)], 1)
    k = rng.random((n, 1))
    col = np.hstack([15 + 35 * k, 105 + 125 * k, 25 + 40 * k])
    cloud.add(pts, col, 0.4 + h * 2.8 + rng.random(n) * .25)

    # garland spiral
    n = 3500
    h = np.sort(rng.uniform(0.07, 0.90, n))
    ang = h * 5.5 * 2 * np.pi
    r = tree_radius(h) * 1.04
    pts = np.stack([r * np.cos(ang), BASE_Y + h * TREE_H + rng.uniform(-.02, .02, n), r * np.sin(ang)], 1)
    cloud.add(pts, np.array(RED) * rng.uniform(.8, 1.1, (n, 1)), 3.4 + (h - .07) * 2.2)

    # baubles
    for i in range(20):
        h, a = rng.uniform(0.08, 0.85), rng.uniform(0, 2 * np.pi)
        r = tree_radius(h) * 0.97
        c = (r * math.cos(a), BASE_Y + h * TREE_H - .03, r * math.sin(a))
        cloud.add(sphere(c, .1, 260), BAUBLES[i % len(BAUBLES)], 4.6 + i * .12)

    # star on top
    n, cy = 800, BASE_Y + TREE_H + 0.13
    star = []
    while len(star) < n:
        x, y = rng.uniform(-1, 1, 2)
        ang, rad = math.atan2(y, x), math.hypot(x, y)
        d = abs((ang - math.pi / 2 + math.pi / 5) % (2 * math.pi / 5) - math.pi / 5)
        if rad <= 1.0 + (0.42 - 1.0) * d / (math.pi / 5):
            star.append((x * .19, cy + y * .19, rng.uniform(-.04, .04)))
    cloud.add(star, (255, 215, 80), 4.2)

    # gifts: boxes, crossed ribbons and a bow
    for deg, dist, size, yaw, color, rib, delay in GIFTS:
        a, y = math.radians(deg), math.radians(yaw)
        sx, sy, sz = size
        base = np.array([dist * math.cos(a), BASE_Y + sy / 2, dist * math.sin(a)])
        pts, nrm = box_surface(size, 3200)
        w = .06 * (sx + sz) / 1.0
        is_rib = ((np.abs(pts[:, 0]) < w) & (nrm[:, 0] == 0)) | ((np.abs(pts[:, 2]) < w) & (nrm[:, 2] == 0))
        col = np.where(is_rib[:, None], np.array(rib, float), np.array(color, float))
        col = col * rng.uniform(.85, 1.1, (len(pts), 1))
        t0 = 5.4 + delay + (pts[:, 1] / sy + .5) * .4 + is_rib * .35
        cloud.add(rot_y(pts, y) + base, col, t0)
        for sgn in (-1, 1):
            bow = sphere((sgn * .07, sy / 2 + .03, 0), .055, 40)
            cloud.add(rot_y(bow, y) + base, rib, 6.6 + delay)

    return cloud.finish()


def project(pos, angle):
    """Rotate around the vertical axis, tilt the camera, apply perspective."""
    p = rot_y(pos, angle)
    c, s = math.cos(TILT), math.sin(TILT)
    y = p[:, 1] * c - p[:, 2] * s
    z = p[:, 1] * s + p[:, 2] * c
    k = FOCAL / (CAM_DIST - z)
    return W / 2 + p[:, 0] * k, H / 2 - y * k, z


# ----------------------------------------------------------- rendering ------
import time

# The screen is a grid of 5x7-pixel character cells. Points are depth-sorted
# into that grid (nearest point wins a cell), then every cell is drawn as an
# opaque character: bright strokes on a darker background of the same color.
CW, CH = 5, 7
COLS, ROWS = W // CW, H // CH            # 161 x 115 cells for 805 x 805 px
GLYPH_ROWS = {
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "#": [".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."],
    "$": ["..#..", ".####", "#.#..", ".###.", "..#.#", "####.", "..#.."],
    "%": ["##..#", "##..#", "...#.", "..#..", ".#...", "#..##", "#..##"],
    "&": [".##..", "#..#.", "#.#..", ".#...", "#.#.#", "#..#.", ".##.#"],
    "*": [".....", "#.#.#", ".###.", "#####", ".###.", "#.#.#", "....."],
    "+": [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."],
    "=": [".....", ".....", "#####", ".....", "#####", ".....", "....."],
    "@": [".###.", "#...#", "#.###", "#.#.#", "#.###", "#....", ".###."],
}
GL = np.array([[[c == "#" for c in row] for row in GLYPH_ROWS[ch]] for ch in "01#$%&*+=@"])
GL = np.concatenate([GL, np.zeros((1, CH, CW), bool)])      # last entry = empty cell (index -1)
NSNOW = 380


def render(scene, snow, t, angle):
    """Return an H x W x 3 uint8 image of the scene at time t."""
    vis = np.nonzero(scene.t0 <= t)[0]
    x, y, z = project(scene.pos[vis], angle)
    order = np.argsort(z)                                    # far -> near
    vis, x, y, z = vis[order], x[order], y[order], z[order]

    cx, cy = (x // CW).astype(np.int32), (y // CH).astype(np.int32)
    ok = (cx >= 0) & (cx < COLS) & (cy >= 0) & (cy < ROWS)
    vis, z, cell = vis[ok], z[ok], (cy * COLS + cx)[ok]

    shade = np.clip(0.45 + 0.55 * (z + 2.2) / 4.4, 0.45, 1.0)
    gid = np.full(ROWS * COLS, -1, np.int8)
    col = np.zeros((ROWS * COLS, 3), np.uint8)
    gid[cell] = (vis % 10).astype(np.int8)                   # nearest (last) point wins the cell
    col[cell] = (scene.col[vis] * shade[:, None]).astype(np.uint8)

    # expand cells -> pixels: strokes at full brightness, background darker
    mask = GL[gid.reshape(ROWS, COLS)].transpose(0, 2, 1, 3).reshape(H, W)
    px = np.repeat(np.repeat(col.reshape(ROWS, COLS, 3), CH, 0), CW, 1)
    mul = np.where(mask, 256, 97).astype(np.uint16)
    img = ((px.astype(np.uint16) * mul[..., None]) >> 8).astype(np.uint8)

    # snow: small white dots drawn on top
    sx, sy, sz = project(snow, angle)
    sx, sy = sx.astype(np.int32), sy.astype(np.int32)
    v = (255 * np.clip(0.5 + 0.5 * (sz + 2.2) / 4.4, 0.5, 1.0)).astype(np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            m = (sx + dx >= 0) & (sx + dx < W) & (sy + dy >= 0) & (sy + dy < H)
            img[(sy + dy)[m], (sx + dx)[m]] = v[m, None]
    return img


def main():
    import tkinter as tk

    scene = build_scene()
    snow = np.stack([rng.uniform(-3, 3, NSNOW), rng.uniform(BASE_Y, 2.4, NSNOW),
                     rng.uniform(-3, 3, NSNOW)], 1)
    fall, phase = rng.uniform(.25, .7, NSNOW), rng.uniform(0, 6.28, NSNOW)

    root = tk.Tk()
    root.title("Christmas Tree")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=W, height=H, bg="black", highlightthickness=0)
    canvas.pack()
    photo = tk.PhotoImage(width=W, height=H)
    canvas.create_image(0, 0, image=photo, anchor="nw")

    state = {"t": 0.0, "angle": 0.0, "paused": False, "last": time.time()}

    def key(e):
        if e.keysym == "Escape":
            root.destroy()
        elif e.keysym == "space":
            state["paused"] = not state["paused"]
        elif e.keysym.lower() == "r":
            state["t"] = 0.0

    root.bind("<Key>", key)

    def frame():
        start = time.time()
        dt, state["last"] = start - state["last"], start
        if state["paused"]:
            dt = 0.0
        state["t"] += dt
        state["angle"] += dt * 0.5

        snow[:, 1] -= fall * dt
        snow[:, 0] += np.sin(state["t"] * 1.3 + phase) * .15 * dt
        gone = snow[:, 1] < BASE_Y
        snow[gone, 1] = 2.4
        snow[gone, 0] = rng.uniform(-3, 3, gone.sum())

        img = render(scene, snow, state["t"], state["angle"])
        photo.configure(data=b"P6 %d %d 255\n" % (W, H) + img.tobytes())
        root.after(max(1, int(1000 / 30 - (time.time() - start) * 1000)), frame)   # aim for 30 fps

    frame()
    root.mainloop()


if __name__ == "__main__":
    main()
