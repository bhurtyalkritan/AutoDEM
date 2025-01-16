import streamlit as st
import numpy as np
import plotly.graph_objects as go
import rasterio
from rasterio.enums import Resampling
from skimage.filters import sobel
import matplotlib.pyplot as plt


def load_dem(file):
    """Loads DEM and returns x, y, elevation, and metadata."""
    with rasterio.open(file) as dem:
        elevation = dem.read(1, resampling=Resampling.bilinear)
        transform = dem.transform
        rows, cols = elevation.shape

        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        x, y = transform * (x, y)
        return x, y, elevation, dem.meta

def compute_slope_aspect(elevation, transform):
    """Computes slope and aspect maps."""
    dx, dy = np.gradient(elevation, transform[0], transform[4])
    slope = np.sqrt(dx ** 2 + dy ** 2)
    aspect = np.arctan2(-dy, dx)
    return slope, aspect

# Terrain Classification
def classify_terrain(elevation):
    """Classifies terrain into categories based on elevation."""
    terrain = np.zeros_like(elevation, dtype=int)
    terrain[elevation < 500] = 1  # Lowlands
    terrain[(elevation >= 500) & (elevation < 1500)] = 2  # Highlands
    terrain[(elevation >= 1500) & (elevation < 2500)] = 3  # Mountains
    terrain[elevation >= 2500] = 4  # High Mountains
    return terrain

def generate_contours(elevation, contour_interval):
    """Generates contours from elevation data."""
    levels = np.arange(np.nanmin(elevation), np.nanmax(elevation), contour_interval)
    return levels

def generate_shaded_relief(elevation):
    """Generates a shaded relief map using Sobel filter."""
    relief = sobel(elevation)
    return relief

st.set_page_config(layout="wide", page_title="Advanced DEM Visualizer")

st.sidebar.title("DEM Visualizer Tools")
uploaded_file = st.sidebar.file_uploader("Upload a DEM file (e.g., .tif)", type=["tif", "asc"])
second_file = st.sidebar.file_uploader("Optional: Upload a second DEM for comparison", type=["tif", "asc"])
st.sidebar.markdown("---")

if uploaded_file:
    try:
        st.sidebar.success("File uploaded successfully!")

        # Load DEM data
        x, y, elevation, metadata = load_dem(uploaded_file)
        if second_file:
            x2, y2, elevation2, _ = load_dem(second_file)

        # Visualization Settings
        st.sidebar.header("Visualization Settings")
        color_scale = st.sidebar.selectbox(
            "Color Scale",
            ["Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo", "Terrain", "Earth"],
            index=0
        )
        transparency = st.sidebar.slider("Transparency (Surface View)", 0.1, 1.0, 1.0)

        # Contours
        st.sidebar.header("Contours")
        show_contours = st.sidebar.checkbox("Show Contour Lines", value=False)
        contour_interval = st.sidebar.slider("Contour Interval", 5, 100, 20)

        # Elevation Filter
        st.sidebar.header("Elevation Filter")
        elevation_min = float(np.nanmin(elevation))
        elevation_max = float(np.nanmax(elevation))
        elevation_range = st.sidebar.slider(
            "Elevation Range", elevation_min, elevation_max, (elevation_min, elevation_max)
        )
        elevation_filtered = np.where(
            (elevation >= elevation_range[0]) & (elevation <= elevation_range[1]),
            elevation,
            np.nan
        )

        # Advanced Analysis Tools
        st.sidebar.header("Analysis Tools")
        slope, aspect = compute_slope_aspect(elevation_filtered, metadata["transform"])
        analysis_view = st.sidebar.selectbox("Analysis View", ["None", "Slope", "Aspect", "Terrain Classification", "Shaded Relief"])

        # Interactive Cropping
        st.sidebar.header("Cropping Tools")
        crop_min_x = st.sidebar.number_input("Min Longitude", value=x.min(), step=1.0)
        crop_max_x = st.sidebar.number_input("Max Longitude", value=x.max(), step=1.0)
        crop_min_y = st.sidebar.number_input("Min Latitude", value=y.min(), step=1.0)
        crop_max_y = st.sidebar.number_input("Max Latitude", value=y.max(), step=1.0)
        crop_mask = (x >= crop_min_x) & (x <= crop_max_x) & (y >= crop_min_y) & (y <= crop_max_y)
        x_cropped = x[crop_mask]
        y_cropped = y[crop_mask]
        elevation_cropped = elevation_filtered[crop_mask]

        st.title("DEM Visualizer")

        st.subheader("3D Visualization")
        fig = go.Figure()
        fig.add_surface(
            z=elevation_filtered,
            surfacecolor=elevation_filtered,
            colorscale=color_scale.lower(),
            showscale=True,
            opacity=transparency
        )
        if show_contours:
            fig.add_contour(
                z=elevation_filtered,
                contours=dict(z=dict(show=True, size=contour_interval, color="black"))
            )
        fig.update_layout(
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                zaxis_title="Elevation"
            ),
            width=1000,
            height=800
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Analysis")
        if analysis_view == "Slope":
            st.image(np.clip(slope, 0, 1), caption="Slope Map", use_container_width=True)
        elif analysis_view == "Aspect":
            plt.figure(figsize=(10, 6))
            plt.imshow(aspect, cmap="twilight", origin="upper")
            plt.colorbar(label="Aspect (radians)")
            plt.title("Aspect Map")
            st.pyplot(plt)
        elif analysis_view == "Terrain Classification":
            terrain_classes = classify_terrain(elevation_filtered)
            plt.figure(figsize=(10, 6))
            plt.imshow(terrain_classes, cmap="terrain", origin="upper")
            plt.colorbar(label="Terrain Classes")
            plt.title("Terrain Classification")
            st.pyplot(plt)
        elif analysis_view == "Shaded Relief":
            shaded_relief = generate_shaded_relief(elevation_filtered)
            plt.figure(figsize=(10, 6))
            plt.imshow(shaded_relief, cmap="gray", origin="upper")
            plt.colorbar(label="Shaded Relief")
            plt.title("Shaded Relief Map")
            st.pyplot(plt)

        st.subheader("Elevation Histogram")
        plt.figure(figsize=(10, 6))
        plt.hist(elevation_filtered[~np.isnan(elevation_filtered)], bins=50, color="blue", edgecolor="black")
        plt.title("Elevation Histogram")
        plt.xlabel("Elevation")
        plt.ylabel("Frequency")
        st.pyplot(plt)

        if second_file:
            st.subheader("DEM Comparison")
            difference = elevation - elevation2
            st.image(difference, caption="Elevation Difference", use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.title("Welcome to the Advanced DEM Visualizer")
    st.write("Upload a DEM file in the sidebar to start exploring!")
