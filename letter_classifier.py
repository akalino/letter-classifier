"""
AMAT 584 test 1 solutions. I'm extra.
"""
import gudhi
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize


# Problem 1: defining the fonts

def font_props(_font):
    """
    Managing the matplotlib font families.

    :param _font: string font name
    :return: fontmanager object for table generation.
    """
    _properties = font_manager.FontProperties(family=_font, weight="bold")
    _font_path = font_manager.findfont(_properties, fallback_to_default=False)
    return font_manager.FontProperties(fname=_font_path)



def build_font_reference(_fonts, _out_file):
    """
    Address the first question in a little table for latex solutions.
    :param _fonts: the list of fonts used.
    :param _out_file: PDF path.
    :return: output path.
    """
    _alphabet = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
    _number_fonts = len(_fonts)
    _figure_height = 0.82 * _number_fonts
    _figure, _axis = plt.subplots(
        figsize=(9.0, _figure_height),
        constrained_layout=False)
    _axis.set_xlim(0.0, 1.0)
    _axis.set_ylim(0.0, _number_fonts)
    _axis.axis("off")
    _label_properties = font_props("DejaVu Sans")

    for _row, _font in enumerate(_fonts):
        _bottom = _number_fonts - _row - 0.88
        _center = _bottom + 0.38
        _font_properties = font_props(_font)
        _axis.text(
            0.045,
            _center,
            _font,
            fontproperties=_label_properties,
            fontsize=11,
            color="#17324D",
            ha="left",
            va="center"
        )
        _axis.text(
            0.255,
            _center,
            _alphabet,
            fontproperties=_font_properties,
            fontsize=14,
            color="#111820",
            ha="left",
            va="center"
        )

    _figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    _figure.savefig(_out_file, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(_figure)
    print(f"Font reference written to {_out_file}")
    return _out_file


# Problem 2: characters, filtrations, barcodes/PDs, and signature classification
# scaling up to homology_dim 2, 3 saved for later script

def build_letter(_letter, _font, _im_sz=256, _ft_sz=200):
    """
    Builds a binary representation of a letter given an input font.

    :param _letter: Latin capital letter.
    :param _font: string font name.
    :param _im_sz: image size.
    :param _ft_sz: font size in image.
    :return: binary mask and matplotlib managed font.
    """
    if _letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raise ValueError("Invalid letter specified.")
    _font_pth = font_manager.findfont(font_manager.FontProperties(family=_font),
                                      fallback_to_default=False)
    _font = ImageFont.truetype(_font_pth, _ft_sz)
    # use L for mode - greyscale
    _image = Image.new("L", (_im_sz, _im_sz), color=0)
    _draw = ImageDraw.Draw(_image)
    _text = _letter
    _box = _draw.textbbox((0, 0), _text, font=_font)
    _text_width = _box[2] - _box[0]
    _text_height = _box[3] - _box[1]
    _x = (_im_sz - _text_width) / 2 - _box[0]
    _y = (_im_sz - _text_height) / 2 - _box[1]

    _draw.text((_x, _y), _text, fill=255, font=_font)
    _mask = np.asarray(_image) >= 128
    return _mask, _font_pth


def compute_homology_simple(_letter_mask):
    """
    Compute Betti numbers for a binary letter mask.
    All planar so just use Euler's formula.

    :param _letter_mask: bool image
    :return: dict with beta_0, beta_1, _hole_mask
    """
    _components = label(_letter_mask, connectivity=2)
    _beta_0 = int(_components.max())
    _chi = int(euler_number(_letter_mask, connectivity=2))
    _beta_1 = _beta_0 - _chi
    _hole_mask = ndimage.binary_fill_holes(_letter_mask) & ~_letter_mask
    _empty_intervals = np.empty((0, 2), dtype=float)

    return {
        "beta_0": _beta_0,
        "beta_1": _beta_1,
        "hole_mask": _hole_mask,
        "filtration": _letter_mask.astype(float),
        "persistence": [],
        "persistence_0": _empty_intervals.copy(),
        "persistence_1": _empty_intervals.copy(),
        "evaluation_scale": 0.0,
        "homology_type": "simple",
    }


def compute_homology_cc_gudhi(_letter_mask):
    """
    Use a cubical complex on the binary letter grid via GUDHI.

    :param _letter_mask:bool image
    :return: dict of metrics
    """
    _inside = ndimage.distance_transform_edt(_letter_mask)
    _outside = ndimage.distance_transform_edt(~_letter_mask)
    _filt = _outside - _inside
    _cubical_complex = gudhi.CubicalComplex(top_dimensional_cells=_filt)
    _persistence = _cubical_complex.persistence(homology_coeff_field=2,
                                                 min_persistence=0.0)
    _betti_nums = _cubical_complex.persistent_betti_numbers(0.0, 0.0)
    _beta_0 = int(_betti_nums[0]) if len(_betti_nums) > 0 else 0
    _beta_1 = int(_betti_nums[1]) if len(_betti_nums) > 1 else 0
    _persistence_0 = _cubical_complex.persistence_intervals_in_dimension(0)
    _persistence_1 = _cubical_complex.persistence_intervals_in_dimension(1)
    _hole_mask = ndimage.binary_fill_holes(_letter_mask) & ~_letter_mask

    return {"beta_0": _beta_0,
            "beta_1": _beta_1,
            "hole_mask": _hole_mask,
            "filtration": _filt,
            "persistence": _persistence,
            "persistence_0": _persistence_0,
            "persistence_1": _persistence_1,
            "evaluation_scale": 0.0,
            "homology_type": "cubical"}


def compute_homology_vr_gudhi(_letter_mask):
    """
    Compute VR persistence from a sample of the letter skeleton.
    The scale then depends on nearest neighbor distances in the resulting
    point cloud.

    :param _letter_mask: bool image
    :return: dict of metrics
    """
    _skeleton = skeletonize(_letter_mask)
    _points = np.column_stack(np.nonzero(_skeleton)).astype(float)

    if len(_points) == 0:
        raise ValueError("Cant build Rips complex from empty mask.")

    _maximum_points = 1000
    if len(_points) > _maximum_points:
        _center = _points.mean(axis=0)
        _first_index = int(np.argmax(np.sum((_points - _center) ** 2,
                                            axis=1)))
        _selected_indices = [_first_index]
        _minimum_distances = np.sum(
            (_points - _points[_first_index]) ** 2,
            axis=1,
        )

        for _ in range(1, _maximum_points):
            _next_index = int(np.argmax(_minimum_distances))
            _selected_indices.append(_next_index)
            _new_distances = np.sum(
                (_points - _points[_next_index]) ** 2,
                axis=1,
            )
            _minimum_distances = np.minimum(
                _minimum_distances,
                _new_distances,
            )

        _points = _points[_selected_indices]

    _coordinate_differences = _points[:, None, :] - _points[None, :, :]
    _distance_matrix = np.sqrt(
        np.sum(_coordinate_differences ** 2, axis=2)
    )
    np.fill_diagonal(_distance_matrix, np.inf)
    _nearest_distances = np.min(_distance_matrix, axis=1)
    _evaluation_scale = 1.5 * float(np.percentile(_nearest_distances, 90))
    _max_edge_length = max(
        4.0 * _evaluation_scale,
        0.15 * min(_letter_mask.shape),
    )

    _rips_complex = gudhi.RipsComplex(
        points=_points,
        max_edge_length=_max_edge_length,
    )
    _simplex_tree = _rips_complex.create_simplex_tree(max_dimension=1)
    _simplex_tree.collapse_edges()
    _simplex_tree.expansion(2)
    _persistence = _simplex_tree.persistence(
        homology_coeff_field=2,
        min_persistence=0.0,
        persistence_dim_max=True
    )
    _betti_nums = _simplex_tree.persistent_betti_numbers(
        _evaluation_scale,
        _evaluation_scale,
    )
    _beta_0 = int(_betti_nums[0]) if len(_betti_nums) > 0 else 0
    _beta_1 = int(_betti_nums[1]) if len(_betti_nums) > 1 else 0
    _persistence_0 = _simplex_tree.persistence_intervals_in_dimension(0)
    _persistence_1 = _simplex_tree.persistence_intervals_in_dimension(1)
    _hole_mask = ndimage.binary_fill_holes(_letter_mask) & ~_letter_mask
    _sample_mask = np.zeros_like(_letter_mask, dtype=bool)
    _sample_rows = _points[:, 0].astype(int)
    _sample_columns = _points[:, 1].astype(int)
    _sample_mask[_sample_rows, _sample_columns] = True

    return {
        "beta_0": _beta_0,
        "beta_1": _beta_1,
        "hole_mask": _hole_mask,
        "filtration": _sample_mask,
        "persistence": _persistence,
        "persistence_0": _persistence_0,
        "persistence_1": _persistence_1,
        "evaluation_scale": _evaluation_scale,
        "homology_type": "vr",
    }


def find_ends(_skeleton):
    """
    Find the endpoints of letter skeleton. Uses the image convolution to look at all the
    neighboring pixels to check if they have the binary mask or not.

    :param _skeleton: boolean skeleton of letter.
    :return: endpoint masks, cheap beta_0
    """
    _kernel = np.ones((3,3), dtype=int)
    _neighbors = ndimage.convolve(_skeleton.astype(int),
                                  _kernel,
                                  mode='constant',
                                  cval=0) - _skeleton
    _end_pt_mask = _skeleton & (_neighbors == 1)
    _end_components = label(_end_pt_mask, connectivity=2)
    _num_ends = int(_end_components.max())
    return _end_components, _num_ends


def analyze_single_letter(_letter, _font, _im_sz=256, _ft_sz=200, _hom_type="simple"):
    """
    Gather metrics for a letter, font pair, with optional adjustment to
    homology computation.

    :param _letter: A-Z.
    :param _font: font name.
    :param _im_sz: image size.
    :param _ft_sz: font size in the image.
    :param _hom_type: type of homology being used. Can be "simple" for the planar case, "cubical", or "vr".
    :return: dict with Bettis and homology metrics.
    """
    _letter_mask, _font_path =build_letter(_letter, _font, _im_sz, _ft_sz)
    if _hom_type == "simple":
        _homology = compute_homology_simple(_letter_mask)
    elif _hom_type == "cubical":
        _homology = compute_homology_cc_gudhi(_letter_mask)
    elif _hom_type == "vr":
        _homology = compute_homology_vr_gudhi(_letter_mask)
    else:
        raise ValueError("Unknown homology type")
    _skeleton = skeletonize(_letter_mask)
    _end_mask, _number_ends = find_ends(_skeleton)

    return {
        "letter": _letter,
        "font_name": _font,
        "font_path": _font_path,
        "letter_mask": _letter_mask,
        "hole_mask": _homology["hole_mask"],
        "skeleton": _skeleton,
        "end_mask": _end_mask,
        "beta_0": _homology["beta_0"],
        "beta_1": _homology["beta_1"],
        "number_ends": _number_ends,
        "filtration": _homology["filtration"],
        "persistence": _homology["persistence"],
        "persistence_0": _homology["persistence_0"],
        "persistence_1": _homology["persistence_1"],
        "evaluation_scale": _homology["evaluation_scale"],
        "homology_type": _hom_type
    }


def loop_fonts(_font_list, _im_sz=256, _ft_sz=200, _hom_type="simple"):
    """
    Tries a few different fonts in a list to see how homology changes.

    :param _font_list: lis of fonts.
    :param _im_sz: image size.
    :param _ft_sz: font size in the image.
    :param _hom_type: type of homology being used.
    :return: results list.
    """
    _results = []
    for _font in _font_list:
        for _letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            _res = analyze_single_letter(_letter, _font, _im_sz, _ft_sz, _hom_type)
            _results.append(_res)
    return _results


def latex_escape(_text):
    _replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    _escaped_text = "".join(
        _replacements.get(_character, _character)
        for _character in str(_text)
    )
    return _escaped_text


def write_latex_table(_results, _output_file):
    _lines = []
    _fonts = []

    for _result in _results:
        _font = _result["font_name"]
        if _font not in _fonts:
            _fonts.append(_font)

    for _table_number, _font in enumerate(_fonts, start=1):
        _font_label = latex_escape(_font)
        _font_results = [
            _result for _result in _results
            if _result["font_name"] == _font
        ]
        _lines.extend([
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{GUDHI homology at filtration value zero "
            f"and end counts for capital Latin letters in {_font_label}.}}",
            f"\\label{{tab:letter-font-homology-{_table_number}}}",
            r"\begin{tabular}{clcc}",
            r"\hline",
            r"Letter & Font & $\beta_1(X)$ & $\beta_0(e(X))$ \\",
            r"\hline",
        ])

        for _result in _font_results:
            _lines.append(
                f"{_result['letter']} & {_font_label} & "
                f"{_result['beta_1']} & {_result['number_ends']} \\\\"
            )

        _lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ])

    with open(_output_file, "w", encoding="utf-8") as _table_file:
        _table_file.write("\n".join(_lines) + "\n")

    print(f"LaTeX table written to {_output_file}")


def plot_hollowing(_letter, _font, _im_sz=256, _ft_sz=200,
                    _hom_type="simple"):
    """
    Show thickening of the eps-ball to identify holes and ends,
    second panel isolates the bounded background holes,
    third panel thins the ink to a one-pixel skeleton,
    final panel marks the endpoint set E(X) in red.
    :param _hom_type: homology computation type.
    :return:
    """
    _analysis = analyze_single_letter(
        _letter,
        _font,
        _im_sz,
        _ft_sz,
        _hom_type
    )
    _figure, _axes = plt.subplots(1, 4, figsize=(14, 4))

    _axes[0].imshow(_analysis["letter_mask"], cmap="gray_r")
    _axes[0].set_title(rf"Character: $\beta_0={_analysis['beta_0']}$")

    _axes[1].imshow(_analysis["letter_mask"], cmap="gray_r", alpha=0.35)
    _axes[1].imshow(
        np.ma.masked_where(~_analysis["hole_mask"], _analysis["hole_mask"]),
        cmap="Blues",
    )
    _axes[1].set_title(rf"Holes: $\beta_1={_analysis['beta_1']}$")

    _axes[2].imshow(_analysis["skeleton"], cmap="gray_r")
    _axes[2].set_title("One-pixel skeleton")

    _axes[3].imshow(_analysis["skeleton"], cmap="gray_r")
    _end_rows, _end_columns = np.nonzero(_analysis["end_mask"])
    _axes[3].scatter(
        _end_columns,
        _end_rows,
        color="red",
        edgecolor="white",
        linewidth=0.5,
        s=45,
        label=r"$e(X)$",
    )
    _axes[3].set_title(
        rf"Ends: $\beta_0(E(X))={_analysis['number_ends']}$"
    )
    if _analysis["number_ends"]:
        _axes[3].legend(loc="upper right")

    for _axis in _axes:
        _axis.axis("off")

    _figure.suptitle(
        f"Letter {_letter} in {_font} | "
        rf"signature $({_analysis['beta_0']}, "
        rf"{_analysis['beta_1']}, {_analysis['number_ends']})$ | "
        f"{_analysis['homology_type']}"
    )
    _figure.tight_layout()
    plt.show()
    return _analysis


def plot_persistence(_analysis, _min_persistence=0.0):
    _persistence = []

    for _dimension, _interval in _analysis["persistence"]:
        _birth, _death = _interval
        _lifetime = _death - _birth
        if np.isinf(_death) or _lifetime >= _min_persistence:
            _persistence.append((_dimension, (_birth, _death)))

    _figure, _axes = plt.subplots(1, 3, figsize=(17, 5))

    if _analysis["homology_type"] == "cubical":
        _filtration_plot = _axes[0].imshow(
            _analysis["filtration"],
            cmap="coolwarm",
        )
        _axes[0].contour(
            _analysis["filtration"],
            levels=[0.0],
            colors="black",
            linewidths=0.8,
        )
        _axes[0].set_title("Signed-distance filtration")
        _figure.colorbar(
            _filtration_plot,
            ax=_axes[0],
            fraction=0.046,
            pad=0.04,
            label="Filtration value (pixels)",
        )
    elif _analysis["homology_type"] == "vr":
        _axes[0].imshow(
            _analysis["letter_mask"],
            cmap="gray_r",
            alpha=0.2,
        )
        _sample_rows, _sample_columns = np.nonzero(
            _analysis["filtration"]
        )
        _axes[0].scatter(
            _sample_columns,
            _sample_rows,
            s=8,
            color="darkred",
        )
        _axes[0].set_title(
            f"Rips sample, scale={_analysis['evaluation_scale']:.2f}"
        )
    else:
        _axes[0].imshow(_analysis["letter_mask"], cmap="gray_r")
        _axes[0].set_title("Binary letter mask")

    _axes[0].axis("off")

    if _persistence:
        gudhi.plot_persistence_barcode(
            _persistence,
            axes=_axes[1],
            legend=True,
            fontsize=10,
        )
        _axes[1].set_title("Persistence barcode")

        gudhi.plot_persistence_diagram(
            _persistence,
            axes=_axes[2],
            legend=True,
            fontsize=10,
        )
        _axes[2].set_title("Persistence diagram")
    else:
        for _axis in _axes[1:]:
            _axis.text(
                0.5,
                0.5,
                "No persistence computed\nfor the simple method",
                ha="center",
                va="center",
            )
            _axis.axis("off")

    _figure.suptitle(
        f"{_analysis['homology_type']} homology for letter "
        f"{_analysis['letter']} in {_analysis['font_name']}"
    )
    _figure.tight_layout()
    plt.show()



def write_signature_latex_table(_results, _output_file):
    """
    Group letters by measured topological signature for each font.

    :param _results: output from loop_fonts.
    :param _output_file: path for the tables.
    :return: nested dict of fonts, signatures, and letters.
    """
    _grouped_results = {}

    for _result in _results:
        _font = _result["font_name"]
        _signature = (
            _result["beta_0"],
            _result["beta_1"],
            _result["number_ends"],
        )

        if _font not in _grouped_results:
            _grouped_results[_font] = {}
        if _signature not in _grouped_results[_font]:
            _grouped_results[_font][_signature] = []

        _grouped_results[_font][_signature].append(_result["letter"])

    _lines = []

    for _table_number, _font in enumerate(_grouped_results, start=1):
        _font_label = latex_escape(_font)
        _font_results = [
            _result for _result in _results
            if _result["font_name"] == _font
        ]
        _homology_label = latex_escape(_font_results[0]["homology_type"])
        _lines.extend([
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{Topological signature classes for {_font_label} "
            f"using {_homology_label} homology.}}",
            f"\\label{{tab:letter-signatures-{_table_number}}}",
            r"\begin{tabular}{c p{0.62\textwidth}}",
            r"\hline",
            r"Signature $(\beta_0(X),\beta_1(X),\beta_0(e(X)))$ & Letters \\",
            r"\hline",
        ])

        for _signature in sorted(_grouped_results[_font]):
            _letters = sorted(_grouped_results[_font][_signature])
            _letter_list = ", ".join(_letters)
            _signature_label = (
                f"$({_signature[0]}, {_signature[1]}, {_signature[2]})$"
            )
            _lines.append(
                f"{_signature_label} & {_letter_list} \\\\"
            )

        _lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ])

    with open(_output_file, "w", encoding="utf-8") as _table_file:
        _table_file.write("\n".join(_lines) + "\n")

    print(f"Signature tables written to {_output_file}")
    return _grouped_results


if __name__ == "__main__":
    letter = "B"
    font = "DejaVu Sans"
    homology_type = "vr"

    fonts = [
        "DejaVu Sans",
        "DejaVu Serif",
        "DejaVu Sans Mono"
    ]

    output_file = "capital_letter_font_reference.pdf"
    build_font_reference(fonts, output_file)

    image_size = 256
    font_size = 200
    # Filters short bars from the plots only. It does not change the table.
    minimum_persistence = 1.0
    output_file = "letter_font_homology_tables.tex"
    sig_out_file = "font_signature_classes_{}.tex".format(homology_type)

    results = loop_fonts(
        fonts,
        image_size,
        font_size,
        homology_type
    )
    write_latex_table(results, output_file)

    signature_groups = write_signature_latex_table(
        results,
        sig_out_file
    )

    selected_analysis = plot_hollowing(
        letter,
        font,
        image_size,
        font_size,
        homology_type
    )
    plot_persistence(selected_analysis, minimum_persistence)
