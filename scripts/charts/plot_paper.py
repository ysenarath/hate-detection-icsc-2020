import matplotlib.pyplot as plt
import numpy as np

schemas = ['M7', 'M6', 'M5', 'M4', 'M3', 'M2', 'M1']
schemas.reverse()
labels = schemas[1:]

# Generate a normal distribution, center at x=0 and y=5
dwmw17_f1 = [96.87000, 96.76000, 96.73000, 96.70000, 96.50000, 96.44000, 95.06000]
dwmw17_f1.reverse()
# dwmw17_f1_delta = [x2 - x1 for (x1, x2) in zip(dwmw17_f1, dwmw17_f1[1:])]
x0 = dwmw17_f1[0]
dwmw17_f1_delta = [(x1 - x0) * 100 / x0 for x1 in dwmw17_f1[1:]]

fdcl18_f1 = [78.82000, 77.54000, 77.51000, 77.52000, 77.13000, 77.07000, 76.54000]
fdcl18_f1.reverse()
# fdcl18_f1_delta = [(x2 - x1) * 100 / (x1) for (x1, x2) in zip(fdcl18_f1, fdcl18_f1[1:])]
x0 = fdcl18_f1[0]
fdcl18_f1_delta = [(x1 - x0) * 100 / x0 for x1 in fdcl18_f1[1:]]

print('\t'.join(labels))
print('\t'.join([str(x) for x in dwmw17_f1_delta]))
print('\t'.join([str(x) for x in fdcl18_f1_delta]))

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

ax.bar(x - width / 2, dwmw17_f1_delta, width, label='DWMW17')
ax.bar(x + width / 2, fdcl18_f1_delta, width, label='FDCL18')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Improvement in F1 Score (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.patch.set_facecolor('white')
fig.tight_layout()

# plt.show()

# fig.savefig(scratch_path("fig-f1-scores.png"))

# tikzplotlib.save(scratch_path("fig-f1-scores.tex"))
