
import pathlib
import tempfile
import streamlit as st
import numpy as np
import pandas as pd

import _tools

st.title("Statistics around blob")
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


    st.subheader("Inner region")

    col1, col2 = st.columns(2)
    internal = col1.number_input("Erode/Dilate blob", min_value=-50, max_value=50, value=0, step=1)

    st.subheader("Ring 1")

    col1, col2 = st.columns(2)
    ring1_inner = col1.number_input("Erode/Dilate blob for inner boundary", min_value=-50, max_value=50, value=0, step=1)
    ring1_outer = col2.number_input("Erode/Dilate blob for outer boundary", min_value=-50, max_value=50, value=2, step=1)

    st.subheader("Ring 2")
    col1, col2  = st.columns(2)
    ring2_inner = col1.number_input("Erode/Dilate  blob for min boundary", min_value=-50, max_value=50, value=6, step=1)
    ring2_outer = col2.number_input("Erode/Dilate  blob for max boundary", min_value=-50, max_value=50, value=8, step=1)

    if st.button("🚀 Analyze image"):

        with st.spinner("Working ..."):

            out = _tools.labeled_image_stats(
                labeled_mask, 
                intensity_imgs,
                internal,
                [
                    (ring1_inner, ring1_outer),
                    (ring2_inner, ring2_outer),
                ]
            )

            df_settings = pd.DataFrame.from_records(
                [
                    {"key": "labeled mask", "value": str(labeled_mask_file)},
                    {"key": "image files", "value": str(im_files)},
                    {"key": "internal", "value": internal},
                    {"key": "ring1 inner", "value": ring1_inner},
                    {"key": "ring1 outer", "value": ring1_outer},
                    {"key": "ring2 inner", "value": ring2_inner},
                    {"key": "ring2 outer", "value": ring2_outer},
                ]
            )

            df = pd.DataFrame.from_records([
                {"label": key, **value} for key, value in out.items()
            ])

            with tempfile.TemporaryDirectory() as folder:
                folder = pathlib.Path(folder)
                file = folder / "results.xlsx"

                _tools.generate_excel_file(file,
                                           {"results": df,
                                            "settings": df_settings})

                st.success("🎉 Done")

                st.download_button("Download results", 
                                data=file.read_bytes(),
                                file_name="results.xlsx",
                                icon=":material/download:"
                                )