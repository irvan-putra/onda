#!/usr/bin/env python3
"""Launcher icon generator: one picture plus the shared version badge.

Deterministic. Same version in, same PNGs out. Inputs are pinned: Pillow
PILLOW_PIN and DejaVu Sans Bold with sha256 FONT_SHA256. The script fails
instead of falling back to another font or Pillow version, because a
different rasterizer or font changes the pixels.
    pip install pillow==12.3.0   (fonts-dejavu-core provides the font)

Badge: numbers only, a full-width strip across the bottom 24% of the icon.
Picture safety: every picture pixel is checked to lie inside the circle that
touches the icon edges (shrunk by a margin), so round launcher masks do not
clip it; the version text is checked the same way.
"""
import hashlib, math, os, re, sys
import PIL
from PIL import Image, ImageChops, ImageDraw, ImageFont

PILLOW_PIN = "12.3.0"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SHA256 = "d1c3ff99f1e1ce1827a33efd4dad81f40babda06bff9e43bd7591c86662a287b"
DENSITIES = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
S = 768
STRIP_TOP = 0.76      # strip covers 76%..100% of the icon height
TEXT_SIZE = 0.115     # numeral size, fraction of icon size
SAFE_R = 0.485        # picture and text must stay inside this radius (fraction)

if PIL.__version__ != PILLOW_PIN:
    sys.exit("Pillow %s required, found %s" % (PILLOW_PIN, PIL.__version__))


def font(px):
    if not os.path.exists(FONT_PATH):
        sys.exit("missing font " + FONT_PATH)
    if hashlib.sha256(open(FONT_PATH, "rb").read()).hexdigest() != FONT_SHA256:
        sys.exit("unexpected font file " + FONT_PATH)
    return ImageFont.truetype(FONT_PATH, px)


def circle_mask(size, r=SAFE_R):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).ellipse([size * (.5 - r), size * (.5 - r), size * (.5 + r), size * (.5 + r)], fill=255)
    return m


def assert_inside_circle(layer, what, size=S, r=SAFE_R):
    alpha = layer.getchannel("A").point(lambda v: 255 if v > 8 else 0)
    outside = ImageChops.multiply(alpha, ImageChops.invert(circle_mask(size, r)))
    box = outside.getbbox()
    if box:
        sys.exit("%s leaves the round-mask safe circle at %s" % (what, box))


def new_layer(size=S):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def sparkle(d, cx, cy, r, fill, size=S):
    pts = []
    for i in range(96):
        t = i * 2 * math.pi / 96
        pts.append((cx * size + r * size * math.cos(t) ** 3, cy * size + r * size * math.sin(t) ** 3))
    d.polygon(pts, fill=fill)


def rrect(d, box, radius, fill, size=S):
    d.rounded_rectangle([c * size for c in box], radius=radius * size, fill=fill)


def compose(bg, art, version, strip, text, rounded=None, size=S, circle_check=True):
    """Background, then the picture layer, then the badge strip."""
    if circle_check:
        assert_inside_circle(art, "picture", size)
    base = Image.new("RGBA", (size, size), bg) if bg else Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base.alpha_composite(art)
    d = ImageDraw.Draw(base)
    y0 = int(size * STRIP_TOP)
    d.rectangle([0, y0, size, size], fill=strip)
    tl, td = new_layer(size)
    f = font(int(size * TEXT_SIZE))
    td.text((size / 2, (y0 + size) / 2), version, font=f, fill=text, anchor="mm")
    if circle_check:
        assert_inside_circle(tl, "version text", size)
    base.alpha_composite(tl)
    if rounded:
        m = Image.new("L", (size, size), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius=rounded * size, fill=255)
        base.putalpha(ImageChops.multiply(base.getchannel("A"), m))
    return base


def parse_version(argv, fallback=None):
    v = argv[1] if len(argv) > 1 else fallback
    if not v or not re.fullmatch(r"\d+\.\d+\.\d+", v):
        sys.exit("version must be X.Y.Z")
    return v


def write_densities(img, res_dir, name="ic_launcher.png"):
    for dens, px in DENSITIES.items():
        out = os.path.join(res_dir, "mipmap-" + dens)
        os.makedirs(out, exist_ok=True)
        img.resize((px, px), Image.LANCZOS).save(os.path.join(out, name), optimize=True)


def _capsule(d, p, q, wd, fill):
    d.line([(p[0] * S, p[1] * S), (q[0] * S, q[1] * S)], fill=fill, width=int(wd * S))
    for x, y in (p, q):
        d.ellipse([(x - wd / 2) * S, (y - wd / 2) * S, (x + wd / 2) * S, (y + wd / 2) * S], fill=fill)


def _hand(d, c, thumb_dir, cyborg):
    SKIN, STEEL, DARK, LED = (244, 196, 160, 255), (176, 190, 210, 255), (70, 84, 120, 255), (80, 230, 255, 255)
    fill = STEEL if cyborg else SKIN
    line = DARK if cyborg else (206, 150, 116, 255)
    # fingers, left to right (tip heights)
    tips = [.31, .23, .19, .24]
    for i, tip in enumerate(tips):
        x = c - .108 + i * .072
        _capsule(d, (x, tip), (x, .52), .062, fill)
    # palm and wrist
    d.rounded_rectangle([(c - .145) * S, .45 * S, (c + .145) * S, .70 * S], radius=.045 * S, fill=fill)
    # thumb toward the centre of the icon
    tx = c + thumb_dir * .15
    _capsule(d, (tx, .56), (c + thumb_dir * .205, .41), .062, fill)
    if cyborg:
        for i, tip in enumerate(tips):                      # finger joints and knuckle lights
            x = c - .108 + i * .072
            for y in (tip + .07, tip + .14):
                d.line([((x - .028) * S, y * S), ((x + .028) * S, y * S)], fill=DARK, width=int(.009 * S))
            d.ellipse([(x - .011) * S, (.50 - .011) * S, (x + .011) * S, (.50 + .011) * S], fill=LED)
        d.ellipse([(c - .05) * S, (.545 - .05) * S, (c + .05) * S, (.545 + .05) * S], outline=DARK, width=int(.012 * S))
        d.ellipse([(c - .022) * S, (.545 - .022) * S, (c + .022) * S, (.545 + .022) * S], fill=LED)
        d.line([((c - .145) * S, .64 * S), ((c + .145) * S, .64 * S)], fill=DARK, width=int(.012 * S))
    else:
        for i in range(1, 4):
            x = c - .108 + i * .072 - .036
            d.line([(x * S, .50 * S), (x * S, .52 * S)], fill=line, width=int(.006 * S))


def onda(version):
    # a human hand and a cyborg hand
    BG, WHITE = (67, 76, 205, 255), (255, 255, 255, 255)
    art, d = new_layer()
    _hand(d, .295, +1, False)
    _hand(d, .705, -1, True)
    return compose(BG, art, version, (20, 24, 80, 255), WHITE, rounded=.2)



def main():
    version = parse_version(sys.argv)
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    write_densities(onda(version), os.path.join(root, "app/src/main/res"))
    print("wrote icons for", version)


if __name__ == "__main__":
    main()
