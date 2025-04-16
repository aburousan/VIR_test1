# vir_processing.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
# Final computed arrays after processing (assume these are saved from full pipeline)
cube_array = np.load("data/cube_array.npy")
dark_corrected_cube = np.load("data/dark_corrected_cube.npy")
spec_radian = np.load("data/spec_radian.npy")
reflectance_val = np.load("data/reflectance_val.npy")
cube_array_cal = np.load("data/cube_array_cal.npy")

wvlen_center = np.load("data/wvlen_center.npy")
spec_radian_cen = np.load("data/spec_radian_cen.npy")
reflectance_val_cen = np.load("data/reflectance_val_cen.npy")
actual_cal = np.load("data/actual_cal.npy")
diff_spe = np.load("data/diff_spe.npy")

# Visualization Helper
def show_band_image(cube, band_index, cmap_val="gray", stretch=True, drop_sample_ranges=None, title="", clip_min=None, clip_max=None, percentile=(2, 98)):
    """
    Show a single band image from the cube with optional sample range removal and enhanced controls.

    Parameters:
        cube : np.ndarray
            Data cube of shape (bands, samples, lines)
        band_index : int
            Index of the band to display
        cmap_val : str
            Colormap used for plotting
        stretch : bool
            Whether to stretch contrast
        drop_sample_ranges : list of tuple(int, int)
            List of (start_sample, end_sample) to drop from the image (sample axis)
        clip_min : float or None
            Manual minimum value to clip
        clip_max : float or None
            Manual maximum value to clip
        percentile : tuple
            Percentile stretch range (e.g. (2, 98))
    """
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
    ax.set_title(f"Band {band_index} - {wvlen_center[band_index]:.2f} µm for {title}")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

