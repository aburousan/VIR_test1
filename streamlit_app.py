import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO

# Load detilt array for VIS only
try:
    detilt_array = np.load("data/detilt_array.npy")
except FileNotFoundError:
    detilt_array = None

from vir_processing import (
    cube_array,
    dark_corrected_cube,
    spec_radian,
    reflectance_val,
    cube_array_cal,
    wvlen_center,
    spec_radian_cen,
    reflectance_val_cen,
    actual_cal,
    diff_spe,
    show_band_image
)
from vir_processing_ir import (
    cube_array_ir,
    dark_corrected_cube_ir,
    spec_radian_ir,
    reflectance_val_ir,
    cube_array_cal_ir,
    wvlen_center_ir,
    spec_radian_cen_ir,
    reflectance_val_cen_ir,
    actual_cal_ir,
    diff_spe_ir,
    show_band_image_ir
)

# Mode selection
mode = st.sidebar.radio("Select Mode", ["VIS", "IR"])

if mode == "VIS":
    st.title("DAWN VIR VIS Data Viewer")
    x_pixel = cube_array.shape[1] // 2
    y_pixel = cube_array.shape[2] // 2
    selected_data = {
        "cube": cube_array,
        "detilt": detilt_array,
        "dark": dark_corrected_cube,
        "rad": spec_radian,
        "ref": reflectance_val,
        "cal": cube_array_cal,
        "wl": wvlen_center,
        "rad_cen": spec_radian_cen,
        "ref_cen": reflectance_val_cen,
        "nasa": actual_cal,
        "diff": diff_spe,
        "show_func": show_band_image
    }
else:
    st.title("DAWN VIR IR Data Viewer")
    x_pixel = cube_array_ir.shape[1] // 2
    y_pixel = cube_array_ir.shape[2] // 2
    selected_data = {
        "cube": cube_array_ir,
        "dark": dark_corrected_cube_ir,
        "rad": spec_radian_ir,
        "ref": reflectance_val_ir,
        "cal": cube_array_cal_ir,
        "wl": wvlen_center_ir,
        "rad_cen": spec_radian_cen_ir,
        "ref_cen": reflectance_val_cen_ir,
        "nasa": actual_cal_ir,
        "diff": diff_spe_ir,
        "show_func": show_band_image_ir
    }

st.markdown("**Author:** Rousan | JRF, NISER")

# View and options
image_views = ["Dark Corrected", "Radiance", "Reflectance"]
if mode == "VIS":
    image_views.insert(0, "Detilted")
    if selected_data["detilt"] is not None:
        image_views.insert(1, "Detilted (from file)")

view_option = st.sidebar.selectbox("Choose image view:", image_views)

plot_option = st.sidebar.multiselect("Spectral plots:", 
    ["Radiance vs Wavelength", "Reflectance vs Wavelength", 
     "Comparison with NASA Calibrated", "Calibration Error"])

cmap_val = st.sidebar.selectbox("Colormap", plt.colormaps(), index=plt.colormaps().index("gray"))
percentile_min = st.sidebar.slider("Percentile Min", 0, 10, 2)
percentile_max = st.sidebar.slider("Percentile Max", 90, 100, 98)
drop_min = st.sidebar.number_input("Drop sample range start (x-axis)", value=0)
drop_max = st.sidebar.number_input("Drop sample range end (x-axis)", value=3)
drop_range = [(drop_min, drop_max)]

selected_band_index = st.slider("Select wavelength (band index)", 0, len(selected_data["wl"]) - 1, len(selected_data["wl"]) // 2)
save_requested = st.sidebar.checkbox("Enable Image Save")

st.subheader(f"Image Comparison at Band {selected_band_index} ({selected_data['wl'][selected_band_index]*1000:.2f} nm)")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Raw Image")
    fig1 = plt.figure()
    selected_data["show_func"](selected_data["cube"], selected_band_index, cmap_val="gray", stretch=True, drop_sample_ranges=drop_range, title="Raw")
    if save_requested:
        buf1 = BytesIO()
        fig1.savefig(buf1, format="png")
        st.download_button("Download Raw Image", buf1.getvalue(), file_name="raw_image.png", mime="image/png")

with col2:
    st.markdown(f"### {view_option} Image")
    fig2 = plt.figure()
    if view_option == "Detilted":
        selected_data["show_func"](selected_data["cube"], selected_band_index, cmap_val=cmap_val, stretch=True, drop_sample_ranges=drop_range, title="Detilted")
    elif view_option == "Detilted (from file)" and selected_data["detilt"] is not None:
        selected_data["show_func"](selected_data["detilt"], selected_band_index, cmap_val=cmap_val, stretch=True, drop_sample_ranges=drop_range, title="Detilted (from file)")
    elif view_option == "Dark Corrected":
        selected_data["show_func"](selected_data["dark"], selected_band_index, cmap_val=cmap_val, stretch=True, drop_sample_ranges=drop_range, title="Dark Corrected")
    elif view_option == "Radiance":
        selected_data["show_func"](selected_data["rad"], selected_band_index, cmap_val=cmap_val, stretch=True, drop_sample_ranges=drop_range, title="Radiance")
    elif view_option == "Reflectance":
        selected_data["show_func"](selected_data["ref"], selected_band_index, cmap_val=cmap_val, stretch=True, drop_sample_ranges=drop_range, title="Reflectance")
    if save_requested:
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png")
        st.download_button(f"Download {view_option} Image", buf2.getvalue(), file_name=f"{view_option.lower().replace(' ', '_')}_image.png", mime="image/png")

st.subheader("Spectral Plots")

if "Radiance vs Wavelength" in plot_option:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data["wl"] * 1000, y=selected_data["rad_cen"],
                             mode='lines+markers', name='Radiance',
                             line=dict(color='black')))
    fig.update_layout(title=f"Radiance at Pixel ({x_pixel}, {y_pixel})",
                      xaxis_title="Wavelength (nm)",
                      yaxis_title="Radiance (W m⁻² μm⁻¹ sr⁻¹)",
                      hovermode="x unified",
                      xaxis_showgrid=True, yaxis_showgrid=True)
    st.plotly_chart(fig, use_container_width=True)

if "Reflectance vs Wavelength" in plot_option:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data["wl"] * 1000, y=selected_data["ref_cen"],
                             mode='lines+markers', name='Reflectance',
                             line=dict(color='green')))
    fig.update_layout(title=f"Reflectance at Pixel ({x_pixel}, {y_pixel})",
                      xaxis_title="Wavelength (nm)",
                      yaxis_title="Reflectance",
                      hovermode="x unified",
                      xaxis_showgrid=True, yaxis_showgrid=True)
    st.plotly_chart(fig, use_container_width=True)

if "Comparison with NASA Calibrated" in plot_option:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data["wl"] * 1000, y=selected_data["rad_cen"],
                             mode='lines+markers', name='Your Calibrated',
                             line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=selected_data["wl"] * 1000, y=selected_data["nasa"],
                             mode='lines+markers', name='NASA Calibrated',
                             line=dict(color='red')))
    fig.update_layout(title="Radiance Comparison",
                      xaxis_title="Wavelength (nm)",
                      yaxis_title="Radiance (W m⁻² μm⁻¹ sr⁻¹)",
                      hovermode="x unified",
                      xaxis_showgrid=True, yaxis_showgrid=True)
    st.plotly_chart(fig, use_container_width=True)

if "Calibration Error" in plot_option:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data["wl"] * 1000, y=selected_data["diff"],
                             mode='lines+markers', name='Difference',
                             line=dict(color='red')))
    fig.update_layout(title="Error (Your Cal - NASA Cal)",
                      xaxis_title="Wavelength (nm)",
                      yaxis_title="Radiance Difference",
                      hovermode="x unified",
                      xaxis_showgrid=True, yaxis_showgrid=True)
    st.plotly_chart(fig, use_container_width=True)
