from PIL import ImageFont

FONT_NAMES = ('arial.ttf', 'DejaVuSans.ttf')


def label_font(size=15):
    for name in FONT_NAMES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    raise RuntimeError('No se encontró una fuente con tildes (arial.ttf o DejaVuSans.ttf)')
