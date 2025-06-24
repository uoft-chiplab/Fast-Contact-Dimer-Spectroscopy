
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

# plt settings
frame_size = 1.5
markers = ["o", "s", "^", "D", "h"]
	
paper_settings = {
				'font.size': 8,          # Base font size
				'axes.labelsize': 8,       # Axis label font size
				'axes.titlesize': 8,       # Title font size (if used)
				'xtick.labelsize': 7,      # Tick label font size (x-axis)
				'ytick.labelsize': 7,      # Tick label font size (y-axis)
				'legend.fontsize': 7,      # Legend font size
				'figure.dpi': 300,        # Publication-ready resolution
				'lines.linewidth': 1,      # Thinner lines for compactness
				"lines.linestyle":'',
				'axes.linewidth': 0.5,      # Thin axis spines
				'xtick.major.width': 0.5,    # Tick mark width
				'ytick.major.width': 0.5,
				'xtick.direction': 'in',     # Ticks pointing inward
				'ytick.direction': 'in',
				'xtick.major.size': 3,      # Shorter tick marks
				'ytick.major.size': 3,
				'font.family': 'sans-serif',
				# 'text.usetex': True,       # Use LaTeX for typesetting, needs local LaTeX install
				'axes.grid': False,       # No grid for PRL figures}
				}

# matplotlib default colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

light_colors = []
dark_colors = []

tintshade = 0.6

def tint_shade_color(color, amount=0.5):
    """
    Tints or shades the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
	
	From https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Examples:
    >> tint_shade_color('g', 0.3)
    >> tint_shade_color('#F034A3', 0.6)
    >> tint_shade_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

for color in colors:
	light_colors.append(tint_shade_color(color, amount=1+tintshade))
	dark_colors.append(tint_shade_color(color, amount=1-tintshade))

styles = [{'color':dark_color, 'mec':dark_color, 'mfc':light_color,
					 'marker':marker} for dark_color, light_color, marker in \
						   zip(dark_colors, light_colors, markers)]
	
	
def generate_plt_styles(colors, markers=markers, ts=tintshade):
	""" Generates style dictionary for use in plt.plot and plt.errorbar """
	light_colors = [tint_shade_color(color, amount=1+ts) for color in colors]
	dark_colors = [tint_shade_color(color, amount=1-ts) for color in colors]
	styles = [{'color':dark_color, 'mec':dark_color, 'mfc':light_color,
					 'marker':marker} for dark_color, light_color, marker in \
						   zip(dark_colors, light_colors, markers)]
	return styles
	

def set_marker_color(color):
	"""
	Sets marker colors s.t. the face color is light and the edge color is like
	a la standard published plot schemes.
	"""
	light_color = tint_shade_color(color, amount=1+tintshade)
	dark_color = tint_shade_color(color, amount=1-tintshade)
	plt.rcParams.update({"lines.markeredgecolor": dark_color,
				   "lines.markerfacecolor": light_color,
				   "lines.color": dark_color})
	
def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
	
def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.tick_params(labelbottom=False, labelleft=False)