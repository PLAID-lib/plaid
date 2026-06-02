from pathlib import Path

# Values read from the provided logs screenshot: Torch dataloader + send to GPU durations, in seconds.
datasets = ["ShapeNetCar", "2D ElPlDyn", "Ahmed"]
series = {
    "CGNS": [1.7468, 59.2142, 322.8863],
    "HF datasets": [0.7718, 9.3384, 145.5219],
    "Zarr": [1.9196, 54.7496, 106.0490],
}
colors = {"CGNS": "#4C78A8", "HF datasets": "#59A14F", "Zarr": "#F28E2B"}

width, height = 760, 360
margin = dict(left=70, right=25, top=55, bottom=80)
plot_w = width - margin["left"] - margin["right"]
plot_h = height - margin["top"] - margin["bottom"]
min_v, max_v = 0, 350

def y(v):
    return margin["top"] + (max_v - v) / (max_v - min_v) * plot_h

def fmt(v):
    return f"{v:.1f}s" if v >= 10 else f"{v:.2f}s"

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">')
svg.append('<title id="title">PLAID dataloader and GPU transfer timings</title>')
svg.append('<desc id="desc">Grouped bar chart comparing CGNS, Hugging Face datasets, and Zarr backends on three datasets using a linear time scale.</desc>')
svg.append('<style>text{font-family:Inter,Roboto,Arial,sans-serif;fill:#263238}.muted{fill:#667085}.grid{stroke:#D0D5DD;stroke-width:1}.axis{stroke:#667085;stroke-width:1.2}.label{font-size:13px}.small{font-size:11px}.title{font-size:19px;font-weight:700}.subtitle{font-size:12px;fill:#667085}.bar-label{font-size:10px;fill:#344054}</style>')
svg.append(f'<rect width="{width}" height="{height}" rx="18" fill="#FFFFFF"/>')
svg.append('<text x="24" y="30" class="title">Dataloader + GPU transfer timings</text>')
svg.append('<text x="24" y="49" class="subtitle">Representative timings from selected PLAID datasets · linear scale · lower is better</text>')

# grid and y-axis labels
for tick in [0, 50, 100, 150, 200, 250, 300, 350]:
    yy = y(tick)
    svg.append(f'<line x1="{margin["left"]}" y1="{yy:.1f}" x2="{width-margin["right"]}" y2="{yy:.1f}" class="grid" opacity="0.65"/>')
    svg.append(f'<text x="{margin["left"]-10}" y="{yy+4:.1f}" text-anchor="end" class="small muted">{tick:g}s</text>')
svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"]+plot_h}" class="axis"/>')
svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"]+plot_h}" x2="{width-margin["right"]}" y2="{margin["top"]+plot_h}" class="axis"/>')

# bars
cluster_w = plot_w / len(datasets)
bar_w = 34
series_names = list(series)
for i, ds in enumerate(datasets):
    cx = margin["left"] + cluster_w * (i + 0.5)
    svg.append(f'<text x="{cx:.1f}" y="{height-40}" text-anchor="middle" class="label">{ds}</text>')
    for j, name in enumerate(series_names):
        v = series[name][i]
        x = cx - (len(series_names)*bar_w + (len(series_names)-1)*8)/2 + j*(bar_w+8)
        yy = y(v)
        h = margin["top"] + plot_h - yy
        svg.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w}" height="{h:.1f}" rx="5" fill="{colors[name]}"/>')
        svg.append(f'<text x="{x+bar_w/2:.1f}" y="{yy-5:.1f}" text-anchor="middle" class="bar-label">{fmt(v)}</text>')

# legend
legend_x = width - 325
for k, name in enumerate(series_names):
    x = legend_x + k * 105
    svg.append(f'<rect x="{x}" y="24" width="12" height="12" rx="3" fill="{colors[name]}"/>')
    svg.append(f'<text x="{x+18}" y="34" class="small">{name}</text>')

svg.append('<text x="24" y="334" class="small muted">Note: timings depend on hardware.</text>')
svg.append('</svg>')

out = Path('plaid_io_performance.svg')
out.write_text('\n'.join(svg), encoding='utf-8')
print(out)