import pathlib
import tempfile
from matplotlib import patches
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.measure import find_contours


import _tools

st.title("Polar maps around blob centroids")
st.text(
"This app calculates the polar intensity map around each detected blob. " 
"It transforms pixel intensities in the region surrounding each blob "
"into polar coordinates (radius and angle), enabling detailed radial "
"and angular analysis."
"\n\n"
"Useful for characterizing local structure, symmetry, or directional features."
)

with st.container(border=True):
    labeled_mask_file = st.file_uploader("Upload blobs labeled mask", type=_tools.KNOWN_IMAGE_FORMATS)
    im_files = st.file_uploader("Upload a single multichannel file or multiple single channel files", type=_tools.KNOWN_IMAGE_FORMATS, accept_multiple_files=True)

    if not im_files or labeled_mask_file is None:
        st.stop()

    intensity_imgs: dict[str, _tools.IntensityImage] = {}
    try:
        for im_file in im_files:
            im = _tools.read_intensity_image_or_stack(im_file)
            if _tools.is_intensity_image(im):
                intensity_imgs[im_file.name] = im
            elif _tools.is_intensity_stack(im):
                for ndx in range(im.shape[0]):
                    intensity_imgs[im_file.name + f"_{ndx+1}"] = im[ndx, :, :]
            else:
                raise Exception(f"In {im_file}, cannot handle an intensity image of {im.ndim} dimensions")
    except Exception as e:
        intensity_imgs = {}
        st.error(f"Error reading intensity file: {e}")
        
    try:
        labeled_mask = _tools.read_labeled_image(labeled_mask_file)
    except Exception as e:
        labeled_mask = None
        st.error(f"Error reading labeled mask: {e}")

    if not intensity_imgs or labeled_mask is None:
        st.stop()

    st.success("Images loaded successfully.")

    count_blobs = len(np.unique(labeled_mask)) - 1
    txt = "\n\n".join(f"- {k}: {v.shape}" for k, v in intensity_imgs.items())

    wrong_size = any(labeled_mask.shape != v.shape for v in intensity_imgs.values())

    st.info(
        f"Labeled mask shape: {labeled_mask.shape}\n\n"
        f"Number of blobs: {count_blobs}\n\n"
        f"Intensity images: {len(intensity_imgs)}\n\n{txt}\n\n"
   )

    st.subheader("Map settings")

    col1, col2 = st.columns(2)
    max_abs_radius = col1.number_input("Maximum absolute radius", min_value=1, max_value=200, value=50, step=1)
    max_rel_radius = col2.number_input("Maximum relative radius", min_value=1, max_value=50, value=10, step=1)

    col1, col2, col3  = st.columns(3)
    subpixel_divisions = col1.number_input("Subpixel divisions", min_value=1, max_value=10, value=1, step=1,
                                           help="Number of subdivisions per pixel used to increase spatial resolution. Each pixel is split into subpixel_divisions evenly spaced positions, allowing for finer sampling of coordinates or smoother accumulation in polar or histogram-based analyses. Higher values increase accuracy but also computational cost.")
    radial_bins = col2.number_input("Number of radial bins", min_value=5, max_value=50, value=10, step=1)
    angular_bins = col3.number_input("Number of angular bins", min_value=5, max_value=30, value=12, step=1)

    CHANNELS = tuple(intensity_imgs.keys())
    col1, col2, col3  = st.columns(3)
    red_channel = col1.selectbox("Red channel", options= CHANNELS + (None, ), index=0)
    green_channel = col2.selectbox("Green channel", options=CHANNELS + (None, ), index=1)
    blue_channel = col3.selectbox("Blue channel", options=CHANNELS + (None, ), 
                                index=max(len(CHANNELS) - 1, 1))


    if st.button("🚀 Generate histograms"):
        
        with st.spinner("Calculating polar values ..."):
            result = _tools.polar_values(labeled_mask, intensity_imgs, max_abs_radius, max_rel_radius, subpixel_divisions)

        with st.spinner("Binning ..."):
            result_hist = _tools.build_polar_histogram(result, radial_bins, angular_bins)
        
        red_channel = CHANNELS.index(red_channel) if red_channel is not None else None
        green_channel = CHANNELS.index(green_channel) if green_channel is not None else None
        blue_channel = CHANNELS.index(blue_channel) if blue_channel is not None else None

        pending = [
            'tab:orange', 'tab:purple', 'tab:brown',
            'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
        ]

        colors = ["X"] * 3
        for ch, col in zip((red_channel, green_channel, blue_channel), ("tab:red", "tab:green", "tab:blue")):
            if ch is None:
                pending.append(col)
            else:
                colors[ch] = col

        colors = [c if c != "X" else pending.pop() for c in colors]
        colors += pending

        with st.spinner("Generating output document ..."):
            with tempfile.TemporaryDirectory() as folder:
                folder = pathlib.Path(folder)
                file = folder / "results.pdf"

                # (3, M, N)
                color_img = _tools.rescale(
                    _tools.to_color_image(
                        np.stack(list(
                            intensity_imgs.values()
                        )).transpose(1, 2, 0),
                        red=red_channel,
                        green=green_channel,
                        blue=blue_channel,
                    )
                )

                # (3, M, N)
                # v.statistic

                with PdfPages(file) as pdf:
                    fig, ax = plt.subplots(1,1)
                    ax.axis(False)
                    table = ax.table(
                        cellText=[
                            [str(max_abs_radius)],
                            [str(max_rel_radius)],
                            [str(subpixel_divisions)],
                            [str(radial_bins)],
                            [str(angular_bins)],
                        ],
                        colLabels=["Value"],
                        rowLabels=[
                            "Maximum absolute radius",
                            "Maximum relative radius",
                            "Subpixel divisions",
                            "Number of radial bins",
                            "Number of angular bins",
                            ],
                        loc='center',
                        cellLoc='center'
                    )   
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)

                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                    for (k, v), rp in zip(result_hist.items(), _tools.typed_regionprops(labeled_mask)):
                        assert k == rp.label

                        fig, axs = plt.subplots(2, 3)
                        fig.suptitle(f"Label: {k}")

                        # Full image
                        ax = axs[0, 0]
                        ax.imshow(color_img)

                        minr, minc, maxr, maxc = rp.bbox  # (min_row, min_col, max_row, max_col)
                        
                        x = minc
                        y = minr
                        width = maxc - minc
                        height = maxr - minr
                        rect = patches.Rectangle((x, y), width, height,
                                                linewidth=2, edgecolor='white', facecolor='none')
                        ax.add_patch(rect)
                        ax.set_xticks([])
                        ax.set_yticks([])

                        # Crop
                        ax = axs[1, 0]

                        max_radius = v.x_edge.max()

                        crop_minr = max(int(minr - max_radius/2), 0)
                        crop_maxr = min(int(maxr + max_radius/2), color_img.shape[0])

                        crop_minc = max(int(minc - max_radius/2), 0)
                        crop_maxc = min(int(maxc + max_radius/2), color_img.shape[1])

                        ax.imshow(_tools.rescale((color_img[crop_minr:crop_maxr, crop_minc:crop_maxc, :])))
                        ax.set_xticks([])
                        ax.set_yticks([])

                        contours = find_contours(labeled_mask[crop_minr:crop_maxr, crop_minc:crop_maxc] == k, level=0.5)
                        for contour in contours:
                            ax.plot(
                                contour[:, 1], 
                                contour[:, 0], 
                                color='white', linewidth=1)


                        # 2D histogram
                        ax = axs[0, 1]

                        # (M, N, 3)
                        color_hist = _tools.rescale(
                            _tools.to_color_image(
                                v.statistic.transpose(2, 1, 0),
                                red=red_channel,
                                green=green_channel,
                                blue=blue_channel,
                            )
                        )

                        im = ax.imshow(color_hist, origin="lower",
                                 #   extent=[v.x_edge[0], v.x_edge[-1], v.y_edge[0], v.y_edge[-1]],
                                  aspect='auto')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # without this line, some images show white in the PDF
                        im.set_rasterized(True)

                        # radial histogram
                        ax = axs[1, 1]
                        stat = np.nansum(v.statistic, axis=2)
                        indep = 0.5 * (v.x_edge[:-1] + v.x_edge[1:])
                        for ndx, (channel, color) in enumerate(zip(v.channels, colors)):
                            ax.plot(indep, stat[ndx], label=channel, color=color)
                        ax.set_xlabel(v.x_label)
                        ax.set_ylabel("Intensity")

                        # angular histogram
                        ax = axs[0, 2]
                        stat = np.nansum(v.statistic, axis=1)
                        indep = 0.5 * (v.y_edge[:-1] + v.y_edge[1:])
                        for ndx, (channel, color) in enumerate(zip(v.channels, colors)):
                            ax.plot(stat[ndx], indep, label=channel, color=color)
                        ax.set_xlabel("Intensity")
                        ax.set_ylabel(v.y_label)
                        ax.yaxis.tick_right()
                        ax.yaxis.set_label_position("right")

                        # Corner
                        ax = axs[1, 2]
                        ax.axis(False)
                        handles, labels = axs[0, 2].get_legend_handles_labels()
                        ax.legend(handles, labels, loc='center', frameon=False)

                        ###
                        fig.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)

                st.success("🎉 Done")

                st.download_button("Download results", 
                                data=file.read_bytes(),
                                file_name="results.pdf",
                                icon=":material/download:"
                                )