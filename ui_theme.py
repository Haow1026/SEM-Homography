"""Shared light/dark workbench colors; CTk accepts each (light, dark) pair."""

BACKGROUND = ('#f3f5f7', '#14181e')
SURFACE = ('#ffffff', '#1d232c')
SUBTLE = ('#edf1f5', '#28313c')
SELECTION = ('#cee2fc', '#354f70')
BORDER = ('#d6dee8', '#354150')
TEXT = ('#202d3b', '#e5ebf2')
MUTED = ('#53657a', '#a6b4c6')
PRIMARY = ('#205faf', '#72aaf2')
PRIMARY_HOVER = ('#174b8d', '#8cbaf5')
PRIMARY_TEXT = ('#ffffff', '#142334')
SUCCESS = ('#24714e', '#7aca9c')
WARNING = ('#8b5100', '#e8bc70')
ERROR = ('#b3261e', '#ff9b92')
# A neutral image field stays constant while the surrounding chrome changes.
CANVAS = '#252a31'
CANVAS_TEXT = '#bdc8d5'


def native_color(pair, mode):
    """Resolve a CTk pair for native Tk controls that cannot use color tuples."""
    return pair[1 if str(mode).lower() == 'dark' else 0]
