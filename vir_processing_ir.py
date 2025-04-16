import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
# Load IR data arrays
cube_array_ir = np.load("data/cube_array_ir.npy")
dark_corrected_cube_ir = np.load("data/dark_corrected_cube_ir.npy")
spec_radian_ir = np.load("data/spec_radian_ir.npy")
reflectance_val_ir = np.load("data/reflectance_val_ir.npy")
cube_array_cal_ir = np.load("data/cube_array_cal_ir.npy")

wvlen_center_ir = np.load("data/wvlen_center_ir.npy")
spec_radian_cen_ir = np.load("data/spec_radian_cen_ir.npy")
reflectance_val_cen_ir = np.load("data/reflectance_val_cen_ir.npy")
actual_cal_ir = np.load("data/actual_cal_ir.npy")
diff_spe_ir = np.load("data/diff_spe_ir.npy")

# Visualization helper
def show_band_image_ir(cube, band_index, cmap_val="gray", stretch=True, drop_sample_ranges=None, title="", clip_min=None, clip_max=None, percentile=(2, 98)):
    img = cube[band_index]

    if drop_sample_ranges is not None:
        mask = np.ones(img.shape[0], dtype=bool)
        for start, end in drop_sample_ranges:
            mask[start:end] = False
        img = img[mask, :]

    if stretch:
        if clip_min is not None and clip_max is not None:
            img = np.clip((img - clip_min) / (clip_max - clip_min), 0, 1)
        else:
            min_val, max_val = np.nanpercentile(img, percentile)
            img = np.clip((img - min_val) / (max_val - min_val), 0, 1)

    fig, ax = plt.subplots()
    im = ax.imshow(img.T, cmap=cmap_val, origin='lower', aspect='auto')
    ax.set_title(f"Band {band_index} - {wvlen_center_ir[band_index]:.2f} µm for {title}")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

