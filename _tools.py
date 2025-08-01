import io
from typing import Any, Literal, NamedTuple, TypeGuard, overload
import zipfile

from scipy.stats import binned_statistic_2d
from skimage.io import imread, imsave # type: ignore
from skimage.measure import regionprops # type: ignore
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


def typed_regionprops(labeled_mask: LabeledImage) -> list[Regionprops]:
    return regionprops(labeled_mask) # type: ignore


# Blob structure for extracted blobs
class Blob(NamedTuple):
    label: Label
    mask: MaskImage

    @property
    def shape(self) -> tuple[int, int]:
        return self.mask.shape
    
    @property
    def area(self) -> int:
        return np.sum(self.mask, dtype=int, initial=0)
    

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


def extract_blobs(labeled_mask: LabeledImage) -> list[Blob]:
    blobs: list[Blob] = []
    for region in typed_regionprops(labeled_mask):
        minr, minc, maxr, maxc = region.bbox
        blob = labeled_mask[minr:maxr, minc:maxc] == region.label
        blobs.append(Blob(region.label, blob))
    return blobs


def try_place_blob(blob: MaskImage, position: tuple[int, int], placed_mask: LabeledImage, label_id: Label) -> bool:
    rows, cols = blob.shape
    r, c = position
    if r + rows > placed_mask.shape[0] or c + cols > placed_mask.shape[1]:
        return False
    if np.any(placed_mask[r:r+rows, c:c+cols][blob]):
        return False
    placed_mask[r:r+rows, c:c+cols][blob] = label_id
    return True

def randomly_place_blobs(labeled_mask: LabeledImage, binary_mask: MaskImage) -> tuple[bool, LabeledImage]:
    blobs = extract_blobs(labeled_mask)
    blobs.sort(key=lambda b: b.area, reverse=True)
    placed_mask = np.zeros_like(binary_mask, dtype=np.int32)
    for blob in blobs:
        candidate_mask = binary_mask & (placed_mask == 0)
        possible_positions = np.argwhere(candidate_mask)
        np.random.shuffle(possible_positions)
        for r, c in possible_positions:
            if try_place_blob(blob.mask, (r, c), placed_mask, blob.label):
                break
        else:
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
        buf = io.BytesIO()
        tifffile.imwrite(buf)
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
