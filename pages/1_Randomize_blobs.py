import streamlit as st
import numpy as np

import _tools

st.title("Random relocation")
st.text(
"This app randomly repositions 'blobs' (given as labeled masks) within a predefined valid region. "
"Each time the function is triggered, the blobs are relocated to new random positions that respect "
"the constraints of the allowed area—ensuring no overlap with forbidden zones and maintaining all "
"blobs fully inside the valid region.\n\n"
"This tool is useful for testing statistical estimator robustness, generating random initial conditions "
"for simulations, or creating dynamic, non-repetitive visualizations."
)

with st.container(border=True):
    labeled_mask_file = st.file_uploader("Upload blobs labeled mask", type=_tools.KNOWN_IMAGE_FORMATS)
    binary_mask_file = st.file_uploader("Upload valid region binary mask", type=_tools.KNOWN_IMAGE_FORMATS)

    if labeled_mask_file is None or binary_mask_file is None:
        st.stop()

    try:
        labeled_mask = _tools.read_labeled_image(labeled_mask_file)
    except Exception as e:
        st.error(f"Error reading labeled mask: {e}")
        labeled_mask = None
    try:
        binary_mask = _tools.read_mask_image(binary_mask_file)
    except Exception as e:
        st.error(f"Error reading binary mask: {e}")
        binary_mask = None

    if labeled_mask is None or binary_mask is None:
        st.stop()

    count_blobs = len(np.unique(labeled_mask)) - 1
    st.success("Images loaded successfully.")
    
    perc_im = (100*np.sum(binary_mask) / binary_mask.size)
    perc_mask = (100*np.count_nonzero(labeled_mask) / np.sum(binary_mask))
    st.info(
        f"Labeled mask shape: {labeled_mask.shape}\n\n"
        f"Binary mask Shape: {binary_mask.shape}\n\n"
        f"Number of blobs: {count_blobs}\n\n"
        f"Number of pixels in binary mask: {np.sum(binary_mask)} ({perc_im:.2f}% of the image)\n\n"
        f"Number of pixels in labeled mask: {np.count_nonzero(labeled_mask)} ({perc_mask:.2f}% of the mask)"
    )

    if binary_mask.shape != labeled_mask.shape:
        st.error("Binary mask and labeled mask must have the same shape.")
        st.stop()

    st.subheader("Randomization settings")
    num_variations = st.number_input("Number of random images to generate:", min_value=1, max_value=50, value=1, step=1)

    if st.button("🚀 Generate randomized version"):
        out: list[_tools.LabeledImage]  = []
        with st.spinner("Working ..."):
            bar = st.progress(0, "")
            cnt = 0
            for ndx in range(num_variations):
                bar.progress(ndx / num_variations, f"{ndx + 1} out of {num_variations}")
                ok, im = _tools.randomly_place_blobs(labeled_mask, binary_mask)
                if ok is True:
                    cnt += 1
                out.append(im)
            bar.progress(100, "done")

        if cnt != num_variations:
            st.success(f"🎉 Generated {cnt} random images.")
        else:
            st.warning(f"Could not place all blobs in {num_variations-cnt} images.")

        st.download_button("Download images", 
                        data=_tools.create_zip_in_memory(
                            {f"random{ndx:03d}_{len(np.unique(im))-1}_{count_blobs}.tif": _tools.image_to_bytes(im, "tif")
                                for ndx, im in enumerate(out, 1)}), 
                        file_name="random.zip",
                        icon=":material/download:"
                        )