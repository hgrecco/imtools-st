import io
import pathlib
import platform
from typing import Any, Literal, NamedTuple, TypeGuard, overload
import zipfile


import pandas as pd
from scipy.stats import binned_statistic_2d
from skimage.io import imread, imsave # type: ignore
from skimage.measure import regionprops # type: ignore
import skimage.morphology as skm

import tifffile
import numpy as np

type MaskImage = np.ndarray[tuple[int, int], np.dtype[np.bool]]
type LabeledImage = np.ndarray[tuple[int, int], np.dtype[np.integer]]
type IntensityImage = np.ndarray[tuple[int, int], np.dtype[np.integer]]
type IntensityStack = np.ndarray[tuple[int, int, int], np.dtype[np.integer]]

type RGBImage = np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.float64]]

type FloatCoodArray = np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]
type FloatVector = np.ndarray[tuple[int, ...], np.dtype[np.floating]]

type Float = float | np.floating[Any]

type Label = int | np.integer[Any]

__version__ = '2025-08-20'

def _check(obj: Any, dtype: np.typing.DTypeLike, ndim: int) -> bool:
    if not isinstance(obj, np.ndarray):
        return False
    if obj.ndim != ndim:
        return False
    if not np.issubdtype(obj.dtype, dtype):  # type: ignore
        return False
    return True


def is_mask_image(obj: Any) -> TypeGuard[MaskImage]:
    return _check(obj, np.bool_, 2)

def is_labeled_image(obj: Any) -> TypeGuard[LabeledImage]:
    return _check(obj, np.unsignedinteger, 2)

def is_intensity_image(obj: Any) -> TypeGuard[IntensityImage]:
    return _check(obj, np.unsignedinteger, 2)

def is_intensity_stack(obj: Any) -> TypeGuard[IntensityStack]:
    return _check(obj, np.unsignedinteger, 3)


class Regionprops(NamedTuple):
    label: Label
    bbox: tuple[int, int, int, int]
    centroid: FloatCoodArray
    equivalent_diameter_area: Float
    image: MaskImage


def typed_regionprops(labeled_mask: LabeledImage) -> list[Regionprops]:
    return regionprops(labeled_mask) # type: ignore

class PolarValues(NamedTuple):
    radii: FloatVector
    angles: FloatVector
    intensities: dict[str, FloatVector]

class BinnedStatistic2dResultPerChannel[M: int, N: int, S: int](NamedTuple):
    # this is wrong, shoudl be M+1 and N+1
    x_edge: np.ndarray[tuple[M, ...], np.dtype[np.floating]]
    y_edge: np.ndarray[tuple[N, ...], np.dtype[np.floating]]
    statistic: np.ndarray[tuple[S, M, N], np.dtype[np.floating]]
    channels: list[str]
    x_label: str = "x"
    y_label: str = "y"


def stats(vec: FloatVector) -> dict[str, Any]:
    if len(vec) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "max": np.nan,
            "min": np.nan,
            "size": 0,
        }
    return {
        "mean": np.mean(vec),
        "std": np.std(vec),
        "median": np.median(vec),
        "iqr": np.subtract(*np.percentile(vec, [75, 25])),
        "max": np.max(vec),
        "min": np.min(vec),
        "size": vec.size,
    }


KNOWN_IMAGE_FORMATS = ("png", "tiff")

def _imread(file: Any) -> Any:
    if hasattr(file, "name"):
        name: str = file.name.lower()
        if name.endswith("tif") or name.endswith("tiff"):
            return tifffile.imread(file)
    return imread(file) # type: ignore

def read_mask_image(file: Any) -> MaskImage:
    """Reads a mask image from a file-like object."""
    return _imread(file) > 0 # type: ignore

def read_labeled_image(file: Any) -> LabeledImage:
    """Reads a labeled image from a file-like object."""
    return _imread(file).astype(np.uint32)   # type: ignore

def read_intensity_image(file: Any) -> IntensityImage:
    """Reads a intensity image from a file-like object."""
    return _imread(file)   # type: ignore

def read_intensity_image_or_stack(file: Any) -> IntensityImage | IntensityStack:
    """Reads a intensity image from a file-like object."""
    return _imread(file)   # type: ignore

def try_place_blob(
    blob_coords: FloatCoodArray,
    placed_mask: LabeledImage,
    label_id: int
) -> bool:

    # Bounds check
    if (
        (blob_coords[:, 0] < 0).any() or
        (blob_coords[:, 1] < 0).any() or
        (blob_coords[:, 0] >= placed_mask.shape[0]).any() or
        (blob_coords[:, 1] >= placed_mask.shape[1]).any()
    ):
        return False

    # Overlap check
    if np.any(placed_mask[blob_coords[:, 0], blob_coords[:, 1]]):
        return False

    # Place blob
    placed_mask[blob_coords[:, 0], blob_coords[:, 1]] = label_id
    return True


def randomly_place_blobs(
    labeled_mask: LabeledImage,
    binary_mask: MaskImage,
) -> tuple[bool, np.ndarray]:
    blobs = typed_regionprops(labeled_mask)
    placed_mask = np.zeros_like(binary_mask, dtype=np.int32)

    available_spots = np.copy(binary_mask)

    blobs.sort(key=lambda b: b.area, reverse=True)

    for blob in blobs:        
        if blob.area == 0:
            continue 

        blob_coords = np.argwhere(blob.image).astype(int)
        blob_coords = blob_coords - np.min(blob_coords, axis=0)

        possible_positions = np.argwhere(available_spots)
        np.random.shuffle(possible_positions)

        for position in possible_positions:
            coords = blob_coords + position
            if try_place_blob(coords, placed_mask, blob.label):
                available_spots[coords[:, 0], coords[:, 1]] = False
                break
        else:
            print(f"Cannot place blob {blob.area}, available {possible_positions.shape}")
            return False, placed_mask

    return True, placed_mask


def polar_values(
    labeled_mask: LabeledImage,
    intensity_images: dict[str, IntensityImage],
    max_absoulute_radius: float = 100.0,
    max_relative_radius: float = 10.0,
    subpixel_divisions: int = 1,
) -> dict[Label, PolarValues]:

    out: dict[Label, PolarValues] = {}

    rows: FloatVector
    cols: FloatVector

    rows, cols = [val.flatten() for val in np.indices(labeled_mask.shape)]
    intensities = {
        name: 1.* intensity_image.flatten()
        for name, intensity_image in intensity_images.items()
    }

    for region in typed_regionprops(labeled_mask):

        r, c = region.centroid # type: ignore

        drows = rows - r
        dcols = cols - c

        radii = np.hypot(drows, dcols)
        angles = np.arctan2(drows, dcols)

        if subpixel_divisions == 1:
            sel = np.logical_and(
                radii <= max_absoulute_radius,
                2 * radii / region.equivalent_diameter_area <= max_relative_radius 
            )

            out[region.label] = PolarValues(
                radii=radii[sel],
                angles=angles[sel],
                intensities={
                    name: intensity[sel] 
                    for name, intensity in intensities.items()
                    }
            )
        else:
            sel = np.logical_and(
                radii <= max_absoulute_radius + 1,
                2 * radii / region.equivalent_diameter_area <= max_relative_radius * 1.1
            )

            drows = drows[sel]
            dcols = dcols[sel]
            values = {
                name: intensity[sel]
                for name, intensity in intensities.items()
            }

            deltas = np.linspace(-0.5, +0.5, subpixel_divisions + 2, endpoint=True)
            deltas = deltas[1:-1][None, :]

            drows = (drows.reshape(-1, 1) + deltas).flatten()
            dcols = (dcols.reshape(-1, 1) + deltas).flatten()
            values = {
                name: (intensity.reshape(-1, 1) + 0 * deltas).flatten()/subpixel_divisions 
                for name, intensity in values.items()
            }
            
            radii = np.hypot(drows, dcols)
            angles = np.arctan2(drows, dcols)

            sel = np.logical_and(
                radii <= max_absoulute_radius,
                2 * radii / region.equivalent_diameter_area <= max_relative_radius 
            )

            out[region.label] = PolarValues(
                radii=radii[sel],
                angles=angles[sel],
                intensities={name: intensity[sel] for name, intensity in values.items()}
            )

    return out

def erode_dilate(image: MaskImage, radius: int) -> MaskImage:
    if radius == 0:
        return image
    footprint = skm.disk(abs(radius))

    if radius > 0:
        return skm.binary_dilation(image, footprint)
    else:
        return skm.binary_erosion(image, footprint)


def labeled_image_stats(
    labeled_mask: LabeledImage,
    intensity_images: dict[str, IntensityImage],
    internal: int,
    rings: list[tuple[int, int]] 
) -> tuple[dict[Label, dict[str, Any]], MaskImage]:

    out: dict[Label, dict[str, Any]] = {}

    full_mask = np.zeros((2 + len(rings), ) + labeled_mask.shape, dtype=np.bool_)

    full_mask[0] = labeled_mask > 0

    for region in typed_regionprops(labeled_mask):

        mask = labeled_mask == region.label

        m = erode_dilate(mask, internal)

        full_mask[1] = np.logical_or(full_mask[1], m)

        tmp = {
            f"ch_{kim}_inner_{kstat}": v 
            for kim, im in intensity_images.items()
            for kstat, v in stats(im[m].flatten()).items()
        }

        for ndx, (inner, outer) in enumerate(rings):
            m = np.logical_and(erode_dilate(mask, outer), np.logical_not(erode_dilate(mask, inner)))
            full_mask[ndx+2] = np.logical_or(full_mask[ndx+2], m)
            tmp.update( {
                f"ch_{kim}_ring{ndx}_{kstat}": v 
                for kim, im in intensity_images.items()
                for kstat, v in stats(im[m].flatten()).items()
            })

        out[region.label] = tmp

    return out, full_mask



@overload
def build_polar_histogram(content: dict[Label, PolarValues], radial_bins: int, angular_bins: int) -> dict[Label, BinnedStatistic2dResultPerChannel[int, int, int]]:
    ...

@overload
def build_polar_histogram(content: PolarValues, radial_bins: int, angular_bins: int) -> BinnedStatistic2dResultPerChannel[int, int, int]:
    ...

def build_polar_histogram(content: dict[Label, PolarValues] | PolarValues, radial_bins: int, angular_bins: int) -> dict[Label, BinnedStatistic2dResultPerChannel[int, int, int]] | BinnedStatistic2dResultPerChannel[int, int, int]:

    if isinstance(content, dict):
        return {
            key: build_polar_histogram(pv, radial_bins, angular_bins) 
            for key, pv in content.items()
        }
    
    assert len(set([v.shape for v in content.intensities.values()])) == 1

    hists =  binned_statistic_2d(
            content.radii,
            content.angles,
            list(content.intensities.values()),
            statistic="mean",
            bins=[radial_bins, angular_bins],
        ) 

    return BinnedStatistic2dResultPerChannel(
        x_edge=hists.x_edge,
        y_edge=hists.y_edge,
        statistic=np.asarray(hists.statistic),
        channels=list(content.intensities.keys()),
        x_label="radius [px]",
        y_label="angle",
    )


def create_zip_in_memory(files: dict[str, bytes]) -> bytes:
    """Creates a ZIP file in memory and returns bytes."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename, file_content in files.items():
            zipf.writestr(filename, file_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def image_to_bytes(im: LabeledImage, format: str = "png") -> bytes:
    """Converts an image to bytes in the specified format."""
    if format == "png":
        buf = io.BytesIO()
        imsave(buf, im, format='png')
        return buf.getvalue()
    elif format in ("tif", "tiff"):
        from PIL import Image
        if im.ndim == 2:
            image = Image.fromarray(im)
            buf = io.BytesIO()
            image.save(buf, format='TIFF')
            return buf.getvalue()
        elif im.ndim == 3:
            images = [Image.fromarray(im[ndx]) for ndx in range(im.shape[0])]
            buf = io.BytesIO()
            images[0].save(buf, format="TIFF", save_all=True, append_images=images[1:])
            return buf.getvalue()
    else:
        raise ValueError(f"Unsupported image format: {format}")
    

def to_color_image(input_array: IntensityStack, red: int | None, green: int | None, blue: int | None) -> RGBImage:

    valid = tuple(range(input_array.shape[2]))
    
    rgb = np.zeros(input_array.shape[:2] + (3, ), dtype=np.float64)

    for target, source in enumerate((red, green, blue)):
        if source not in valid:
            continue
        rgb[:, :, target] = input_array[:, :, source]

    return rgb


def rescale(input_array: RGBImage) -> RGBImage:

    rescaled_array = np.zeros_like(input_array, dtype=np.float64)

    for ndx in range(3): 
        channel = input_array[:, :, ndx]
        p5 = np.percentile(channel, 5)
        p95 = np.percentile(channel, 95)

        # Avoid division by zero if p5 == p95
        if p95 == p5:
            rescaled_array[:, :, ndx] = 0.5  # Neutral gray if no range
        else:
            rescaled_array[:, :, ndx] = (channel - p5) / (p95 - p5)

    # Clip just to avoid any numerical overshoot
    rescaled_array = np.clip(rescaled_array, 0, 1)

    return rescaled_array

def versions() -> list[tuple[str, str]]:
    """Return list of used packages and their versions.
    """
    import scipy as sp
    import matplotlib as mpl
    import PIL
    import skimage as sk
    import tifffile as tf
    import openpyxl

    return [
        ("Python", platform.python_version()),
        ("NumPy", np.version.version),
        ("SciPy", sp.__version__),
        ("Pandas", pd.__version__),
        ("matplotlib", mpl.__version__),
        ("pillow", PIL.__version__),
        ("skimage", sk.__version__),
        ("tifffile", tf.__version__),
        ("openpyxl", openpyxl.__version__),
        ("imtools-st", __version__),
    ]


def generate_excel_file(path: pathlib.Path, sheetname_2_df: dict[str, pd.DataFrame]):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheetname_2_df.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

