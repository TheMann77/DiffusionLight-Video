import matplotlib.pyplot as plt
import cv2

from scripts_depthlight.utility_functions import load_exr, save_hdr_as_ldr
import numpy as np
from scipy.optimize import curve_fit

def model(x, a, b, c, d):
    return np.tan((np.pi / c) * (x + d)) * a + b

"""def model(x, a, b):
    return a * x + b"""

DiffusionLight = load_exr("intermediate/ball_frames/naive/hdr/example0.exr")
LEDiff = cv2.imread("test/frames/hdr_0.hdr", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

all_params = []
yms = []
plt.figure()
for i, c in enumerate(["red", "green", "blue"]):
    DL_flat = DiffusionLight[:,:,i].flatten()
    LE_flat = LEDiff[:,:,i].flatten()

    p0 = [1.0, 1.0, 5.0, 0.5]
    #p0=[.0,.0]

    params, _ = curve_fit(model, DL_flat, LE_flat, p0=p0)
    
    x_curve = np.linspace(np.min(DL_flat), np.max(DL_flat), 500)
    y_curve = model(x_curve, *params)

    all_params.append(params)
    print(params)
    #plt.figure()
    plt.scatter(DiffusionLight[:,:,i].flatten(), LEDiff[:,:,i].flatten(), s=.01, c=c)
    #plt.plot(x_curve, y_curve, color="yellow", linewidth=2)
    #plt.xlabel("DiffusionLight intensity")
    #plt.ylabel("LEDiff intensity")
    #plt.savefig(f"{c}.png")

x_curve = np.linspace(np.min(DiffusionLight), np.max(DiffusionLight), 500)
y_curve = model(x_curve, .66, 1.25, 1.52, -.47)
plt.plot(x_curve, y_curve, color="yellow", linewidth=2)
plt.xlabel("DiffusionLight intensity")
plt.ylabel("LEDiff intensity")
plt.savefig(f"colmap2.png")

save_hdr_as_ldr(DiffusionLight, "DiffusionLight.png")
save_hdr_as_ldr(LEDiff, "LEDiff.png")

for i in range(3):
    DiffusionLight_col = DiffusionLight[:, :, i]
    a, b, c, d = all_params[i]
    DiffusionLight_col = np.tan((np.pi / c) * (DiffusionLight_col + d)) * a + b
    DiffusionLight[:, :, i] = DiffusionLight_col
save_hdr_as_ldr(DiffusionLight, "DiffusionLightNew.png")