"""Kaggle ``cattle-weight-detection-model-dataset-12k`` annotation parser.

Reads the COCO keypoint JSONs under ``Vector/{B2,B3,B4}/{Side,Rear}/data/``
and emits per-image :class:`KaggleSample` records with canonicalized keypoint
names. See ``docs/kaggle_dataset_notes.md`` for the underlying schema findings.

Design notes
------------

* Keypoint schemas differ per batch and per category. Each JSON's
  ``categories[].keypoints`` array is treated as the source of truth; we map
  every entry by name into a canonical 9-keypoint side-view or 4-keypoint
  rear-view dict. Missing keypoints map to ``None``.
* The annotators put ``_bottom`` keypoints before ``_top`` ones in array
  positions 3-8 of B3 Side — we never index ``annotation["keypoints"]`` by
  hardcoded position. The remap is computed once per ``(json_path, category_id)``
  pair from the category's keypoint-name array.
* Weight ground truth is **encoded in image filenames**, not in the COCO
  annotations. B2 lacks weight; B3/B4 follow distinct regexes parsed below.
* Open questions documented in ``docs/kaggle_dataset_notes.md``: B4 Side
  keypoint schema and B2 Side_2 anonymous mapping are unresolved. The parser
  currently raises :class:`UnknownSchemaError` for unrecognized
  ``(batch, category_name)`` pairs rather than silently mis-mapping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from cattle_phenotyping.utils.log import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------- canonical names

CANONICAL_SIDE_KEYPOINTS: tuple[str, ...] = (
    "wither",
    "pinbone",
    "shoulderbone",
    "front_girth_top",
    "front_girth_bottom",
    "rear_girth_top",
    "rear_girth_bottom",
    "height_top",
    "height_bottom",
)

CANONICAL_REAR_KEYPOINTS: tuple[str, ...] = ("top", "bottom", "left", "right")


# ----------------------------------------------------------- per-batch name maps
#
# Keys are the literal names appearing in the COCO ``categories[].keypoints``
# array; values are canonical names. ``None`` (or absence) drops that keypoint
# from canonical output. The parser warns once per unmapped name so future
# schema surprises are visible without crashing.
#
# Side-view maps -------------------------------------------------------------

# B3 Side — the gold reference (2,603 images), full 9-keypoint anatomical schema.
_SIDE_NAME_MAP_B3: dict[str, str] = {
    "1_wither": "wither",
    "2_pinbone": "pinbone",
    "3_shoulderbone": "shoulderbone",
    "4_front_girth_top": "front_girth_top",
    "5_front_girth_bottom": "front_girth_bottom",
    "6_rear_girth_top": "rear_girth_top",
    "7_rear_girth_bottom": "rear_girth_bottom",
    "8_Height_top": "height_top",
    "9_Height_bottom": "height_bottom",
}

# B2 Side / cattle-biometrics — 6 keypoints, partial anatomy, hyphen-separated.
# Missing: shoulderbone, height_top, height_bottom.
_SIDE_NAME_MAP_B2_BIOMETRICS: dict[str, str] = {
    "wither": "wither",
    "pinbone": "pinbone",
    "front-top": "front_girth_top",
    "front-bottom": "front_girth_bottom",
    "rear-top": "rear_girth_top",
    "rear-bottom": "rear_girth_bottom",
}

# B4 Side — schema not yet confirmed; assume it matches B3 by name. If B4 uses
# different names this will surface as warnings until the dict is fixed.
_SIDE_NAME_MAP_B4_ASSUMED: dict[str, str] = dict(_SIDE_NAME_MAP_B3)

SIDE_NAME_MAPS: dict[tuple[str, str], dict[str, str]] = {
    ("B2", "cattle-biometrics  NEW"): _SIDE_NAME_MAP_B2_BIOMETRICS,
    ("B3", "b3_side"): _SIDE_NAME_MAP_B3,
    # B4 placeholder until the inspector confirms; see docs/kaggle_dataset_notes.md.
    ("B4", "cattle_side"): _SIDE_NAME_MAP_B4_ASSUMED,
    ("B4", "b4_side"): _SIDE_NAME_MAP_B4_ASSUMED,
}

# Rear-view maps -------------------------------------------------------------

_REAR_NAME_MAP_NAMED: dict[str, str] = {
    "1_top": "top",
    "2_bottom": "bottom",
    "3_left": "left",
    "4_right": "right",
    # B3 second category uses bare PascalCase.
    "Top": "top",
    "Bottom": "bottom",
    "Left": "left",
    "Right": "right",
}

# B2 Rear keypoints are anonymous numeric IDs ["27","26","24","25"]. Inferred
# from sample annotation coordinates: kp0=high-x ⇒ right, kp1=low-x ⇒ left,
# kp2=low-y ⇒ top, kp3=high-y ⇒ bottom.
_REAR_NAME_MAP_B2_ANON: dict[str, str] = {
    "27": "right",
    "26": "left",
    "24": "top",
    "25": "bottom",
}

REAR_NAME_MAPS: dict[tuple[str, str], dict[str, str]] = {
    ("B2", "Cow back"): _REAR_NAME_MAP_B2_ANON,
    ("B3", "b3_rear"): _REAR_NAME_MAP_NAMED,
    ("B3", "Cattle Rear"): _REAR_NAME_MAP_NAMED,
    ("B4", "cattle_rear"): _REAR_NAME_MAP_NAMED,
}


# ----------------------------------------------------------------- filename grammars
#
# Findings from the dataset inspection (see docs/kaggle_dataset_notes.md):
#
# * B2 has TWO filename grammars that coexist:
#     - the rich grammar used by the actual image files on disk and in Pixel/
#       masks: ``<animal_id>_<r|s>_<weight>_<extra_decimal>_<M|F>.jpg`` where
#       ``animal_id`` is sometimes an integer (``450``) and sometimes a float
#       (``113.0``); ``extra_decimal`` is a mystery 1-10 score (likely BCS or
#       age — unconfirmed pending PDF inspection).
#     - the simplified grammar used by Vector/B2/.../COCO_*.json image
#       ``file_name`` entries: ``<seq_id>_<r|s>.jpg``. These sequential IDs do
#       NOT correspond to ``animal_id`` in the rich grammar, so B2 Vector
#       annotations cannot be joined to Pixel masks without a mapping table.
#       We accept this grammar so the parser doesn't crash on Vector entries,
#       but mark such samples as having no animal_id / no weight.
# * B3 grammar: ``<animal_id>_<r|s>_<weight>_<M|F>.jpg`` (no extra field).
# * B4 grammar: ``<animal_id>_b4-<sub>_<r|s>_<weight>_<M|F>.jpg`` where
#   ``sub`` is 1–4 (B4 has four sub-batches).
#
# Animal IDs that come in as floats (``113.0``) are normalized to integer
# strings (``"113"``) so ``animal_key`` is stable across views.

_RE_B2_RICH = re.compile(
    r"^(?P<animal_id>\d+(?:\.\d+)?)_(?P<view>[rs])_(?P<weight>\d+(?:\.\d+)?)_"
    r"(?P<extra>\d+(?:\.\d+)?)_(?P<sex>[MF])\.(?:jpe?g|png)$",
    re.I,
)
_RE_B2_SIMPLE = re.compile(
    r"^(?P<seq>\d+)_(?P<view>[rs])\.(?:jpe?g|png)$",
    re.I,
)
_RE_B3 = re.compile(
    r"^(?P<animal_id>\d+)_(?P<view>[rs])_(?P<weight>\d+(?:\.\d+)?)_(?P<sex>[MF])\.(?:jpe?g|png)$",
    re.I,
)
_RE_B4 = re.compile(
    r"^(?P<animal_id>\d+)_b4-(?P<sub>\d+)_(?P<view>[rs])_(?P<weight>\d+(?:\.\d+)?)_(?P<sex>[MF])\.(?:jpe?g|png)$",
    re.I,
)


def _normalize_animal_id(raw: str) -> str:
    """Collapse `"113.0"` and `"113"` to a single canonical string key."""
    if "." in raw:
        try:
            f = float(raw)
            if f.is_integer():
                return str(int(f))
        except ValueError:
            pass
    return raw


# ------------------------------------------------------------------ data classes


@dataclass
class FilenameMeta:
    """Fields recovered from a Kaggle image filename."""

    animal_id: str | None
    view: Literal["side", "rear"]
    weight_kg: float | None = None
    sex: Literal["M", "F"] | None = None
    batch_sub: str | None = None  # only populated for B4 (e.g., "1" from "b4-1")
    # B2 rich grammar's 5th decimal field — high-confidence inferred to be
    # **age in years** (PDF mentions age as a tracked attribute; observed
    # values 1.6-10.0 fit cattle lifespan, decimals rule out BCS and teeth
    # counts). Still stored opaquely as ``extra`` rather than ``age_years``
    # pending an independent verification; see docs/kaggle_dataset_notes.md.
    # ``None`` for B3/B4 and B2-simplified filenames.
    extra: float | None = None
    # ``True`` when the filename came from the B2 simplified grammar used by
    # Vector COCO files. Such samples have a sequential ``seq_id`` but no
    # recoverable animal_id, weight, or sex; downstream code should skip them
    # for weight-supervised training.
    is_b2_seq_only: bool = False


@dataclass
class KaggleSample:
    """A single image + its parsed annotations and filename-encoded metadata."""

    image_path: Path
    batch: Literal["B2", "B3", "B4"]
    view: Literal["side", "rear"]
    coco_image_id: int
    coco_category_id: int
    # Canonical-name → (x, y, visibility). Missing canonical keypoints map to None.
    keypoints: dict[str, tuple[float, float, int] | None]
    bbox: tuple[float, float, float, float] | None  # COCO (x, y, w, h)
    image_width: int
    image_height: int
    filename_meta: FilenameMeta
    # Filled in by the segmentation linker (Phase 3 follow-up); not by this parser.
    mask_path: Path | None = None

    @property
    def animal_id(self) -> str | None:
        """Batch-local animal id; combine with ``batch`` for a global key.

        ``None`` for B2 samples that come from the simplified Vector grammar
        (sequential IDs that don't map back to real animal IDs).
        """
        return self.filename_meta.animal_id

    @property
    def animal_key(self) -> tuple[str, str] | None:
        """``(batch, animal_id)`` — a globally unique animal identifier."""
        if self.animal_id is None:
            return None
        return (self.batch, self.animal_id)

    @property
    def weight_kg(self) -> float | None:
        return self.filename_meta.weight_kg


# ------------------------------------------------------------------------- errors


class UnknownSchemaError(KeyError):
    """Raised when a COCO category isn't in the per-batch name map."""


class FilenameParseError(ValueError):
    """Raised when a filename doesn't match its batch's grammar."""


# ------------------------------------------------------------------------- parser


def parse_filename(batch: str, filename: str) -> FilenameMeta:
    """Parse Kaggle image filenames per batch.

    B2 accepts both the rich grammar (typical Pixel/ image filenames) and the
    simplified Vector-COCO grammar; the latter sets ``is_b2_seq_only=True``
    and leaves animal_id / weight / sex unset.

    >>> parse_filename("B3", "9_r_144_M.jpg").animal_id
    '9'
    >>> parse_filename("B4", "100_b4-1_r_124_F.jpg").batch_sub
    '1'
    >>> parse_filename("B2", "450_s_172_3.6_F.jpg").weight_kg
    172.0
    >>> parse_filename("B2", "113.0_s_185_4.0_F.jpg").animal_id  # float -> int
    '113'
    >>> parse_filename("B2", "1_r.jpg").is_b2_seq_only
    True
    """
    if batch == "B2":
        m = _RE_B2_RICH.match(filename)
        if m:
            return FilenameMeta(
                animal_id=_normalize_animal_id(m["animal_id"]),
                view="rear" if m["view"].lower() == "r" else "side",
                weight_kg=float(m["weight"]),
                sex=m["sex"].upper(),  # type: ignore[arg-type]
                extra=float(m["extra"]),
            )
        m = _RE_B2_SIMPLE.match(filename)
        if m:
            return FilenameMeta(
                animal_id=None,
                view="rear" if m["view"].lower() == "r" else "side",
                is_b2_seq_only=True,
            )
        raise FilenameParseError(f"B2 filename did not match any known grammar: {filename!r}")

    if batch == "B3":
        m = _RE_B3.match(filename)
        if not m:
            raise FilenameParseError(f"B3 filename did not match: {filename!r}")
        return FilenameMeta(
            animal_id=_normalize_animal_id(m["animal_id"]),
            view="rear" if m["view"].lower() == "r" else "side",
            weight_kg=float(m["weight"]),
            sex=m["sex"].upper(),  # type: ignore[arg-type]
        )

    if batch == "B4":
        m = _RE_B4.match(filename)
        if not m:
            raise FilenameParseError(f"B4 filename did not match: {filename!r}")
        return FilenameMeta(
            animal_id=_normalize_animal_id(m["animal_id"]),
            view="rear" if m["view"].lower() == "r" else "side",
            weight_kg=float(m["weight"]),
            sex=m["sex"].upper(),  # type: ignore[arg-type]
            batch_sub=m["sub"],
        )

    raise ValueError(f"Unknown batch: {batch!r}")


def _build_remap(
    batch: str,
    view: Literal["side", "rear"],
    category: dict,
    *,
    warned: set[tuple[str, str]],
) -> tuple[list[str | None], tuple[str, ...]]:
    """Compute (array_index → canonical_name_or_None) for a COCO category.

    Returns a parallel list of canonical names (or ``None`` for unmapped slots)
    matching the layout of ``annotation["keypoints"]`` triplets, plus the
    canonical-name tuple that the parser will populate.
    """
    name_map_table = SIDE_NAME_MAPS if view == "side" else REAR_NAME_MAPS
    canonical = CANONICAL_SIDE_KEYPOINTS if view == "side" else CANONICAL_REAR_KEYPOINTS
    cat_name = category["name"]

    try:
        name_map = name_map_table[(batch, cat_name)]
    except KeyError as exc:
        raise UnknownSchemaError(
            f"No keypoint name map for ({batch!r}, category={cat_name!r}). "
            "Update SIDE_NAME_MAPS / REAR_NAME_MAPS in cattle_phenotyping/data/kaggle.py."
        ) from exc

    array_names: list[str] = list(category["keypoints"])
    remap: list[str | None] = []
    for raw_name in array_names:
        canonical_name = name_map.get(raw_name)
        if canonical_name is None and (batch, raw_name) not in warned:
            warned.add((batch, raw_name))
            log.warning(
                "Unmapped keypoint name %r in (%s, %s); dropping from canonical output.",
                raw_name, batch, cat_name,
            )
        remap.append(canonical_name)
    return remap, canonical


def _extract_keypoints(
    flat: list[float],
    remap: list[str | None],
    canonical: tuple[str, ...],
) -> dict[str, tuple[float, float, int] | None]:
    """Turn a flat ``[x,y,v,x,y,v,...]`` array into a canonical-name dict.

    Visibility codes follow COCO convention (0 = not labelled, 1 = labelled
    but not visible, 2 = labelled and visible). The dict has one entry per
    canonical keypoint; missing ones map to ``None``.
    """
    out: dict[str, tuple[float, float, int] | None] = {name: None for name in canonical}
    if len(flat) != 3 * len(remap):
        raise ValueError(
            f"Keypoint array has {len(flat)} entries but expected {3 * len(remap)} "
            f"(3× number of category keypoints)."
        )
    for idx, canonical_name in enumerate(remap):
        if canonical_name is None:
            continue
        x = float(flat[3 * idx])
        y = float(flat[3 * idx + 1])
        v = int(flat[3 * idx + 2])
        out[canonical_name] = (x, y, v)
    return out


def load_coco(json_path: Path) -> dict:
    """Load a COCO JSON; thin wrapper for clarity."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_mask_path(
    file_name: str,
    *,
    dataset_root: Path,
    batch: str,
    view: Literal["side", "rear"],
) -> Path | None:
    """Locate the segmentation mask PNG for a given image filename.

    Naming convention discovered from spot-check: masks live at
    ``Pixel/<batch>/<Side|Rear>/annotations/<image_filename>___fuse.png``.
    Joins cleanly for B3 and B4 because their COCO ``file_name`` entries match
    the Pixel-side filenames; **does not work for B2** because B2 Vector COCO
    uses sequential IDs that don't match the rich Pixel filenames. Returns
    ``None`` (with no warning) for B2; otherwise returns the path even if it
    doesn't exist on disk, so the caller can decide how to handle misses.
    """
    if batch == "B2":
        # Joining requires a separate Vector-seq-id ↔ rich-filename mapping
        # that the dataset doesn't ship. See docs/kaggle_dataset_notes.md.
        return None

    # B3 has masks under Pixel/B3/annotations/ (no Side/Rear split in the
    # spot-check); B4 keeps the Side/Rear split: Pixel/B4/Side/annotations/.
    if batch == "B3":
        candidate = dataset_root / "Pixel/B3/annotations" / f"{file_name}___fuse.png"
    else:  # B4
        view_dir = "Side" if view == "side" else "Rear"
        candidate = dataset_root / f"Pixel/B4/{view_dir}/annotations" / f"{file_name}___fuse.png"
    return candidate


def _resolve_image_path(coco_image: dict, json_path: Path) -> Path:
    """Best-effort resolution of the absolute image path for a COCO image record.

    COCO files in this dataset vary: B4 has an absolute ``path`` field
    (``/datasets/cattle_b4_rear/...``) that doesn't exist on Kaggle's filesystem,
    while B2/B3 just have ``file_name``. We always trust ``file_name`` and try a
    handful of likely image directories near the JSON.
    """
    fname = coco_image["file_name"]
    json_dir = json_path.parent
    candidates = [
        json_dir / "images" / fname,
        json_dir / fname,
        json_dir.parent / "images" / fname,
        json_dir.parent / fname,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fall back to the first candidate; the caller can check ``.exists()``.
    return candidates[0]


def parse_coco_file(
    json_path: Path,
    *,
    batch: str,
    view: Literal["side", "rear"],
    dataset_root: Path | None = None,
) -> Iterator[KaggleSample]:
    """Yield one :class:`KaggleSample` per annotation in a COCO file.

    Multiple annotations per image (rare but present, e.g. B2 Rear has 323
    annotations across 320 images) all yield separate samples; downstream
    consumers can group on ``coco_image_id`` if a single-annotation-per-image
    view is needed.
    """
    coco = load_coco(json_path)
    images_by_id = {img["id"]: img for img in coco["images"]}
    warned: set[tuple[str, str]] = set()

    # Pre-compute remaps per category to avoid recomputing per annotation.
    remaps: dict[int, tuple[list[str | None], tuple[str, ...]]] = {}
    for category in coco["categories"]:
        try:
            remaps[category["id"]] = _build_remap(batch, view, category, warned=warned)
        except UnknownSchemaError:
            log.error(
                "Skipping category id=%s name=%r in %s (no schema map).",
                category["id"], category["name"], json_path,
            )

    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in remaps:
            continue  # category we don't know how to map
        remap, canonical = remaps[cat_id]

        image_record = images_by_id.get(ann["image_id"])
        if image_record is None:
            log.warning("Annotation %s references unknown image_id=%s", ann["id"], ann["image_id"])
            continue

        try:
            filename_meta = parse_filename(batch, image_record["file_name"])
        except FilenameParseError as exc:
            log.warning("%s", exc)
            continue

        # Verify the filename's view tag matches the JSON's view bucket. If not,
        # the dataset has been mis-shelved — log and skip rather than crash.
        if filename_meta.view != view:
            log.warning(
                "View mismatch: %s says %s but JSON %s is %s; skipping.",
                image_record["file_name"], filename_meta.view, json_path, view,
            )
            continue

        keypoints = _extract_keypoints(ann["keypoints"], remap, canonical)

        bbox_raw = ann.get("bbox")
        # B4 sometimes has [0,0,0,0] degenerate bboxes — pass None upstream.
        if bbox_raw and any(v != 0 for v in bbox_raw):
            bbox = (
                float(bbox_raw[0]), float(bbox_raw[1]),
                float(bbox_raw[2]), float(bbox_raw[3]),
            )
        else:
            bbox = None

        # B4's image_record has both top-level width/height and an inner one;
        # COCO standard fields are width/height.
        width = int(image_record.get("width", 0))
        height = int(image_record.get("height", 0))

        mask_path = (
            _resolve_mask_path(
                image_record["file_name"],
                dataset_root=dataset_root,
                batch=batch,
                view=view,
            )
            if dataset_root is not None
            else None
        )

        yield KaggleSample(
            image_path=_resolve_image_path(image_record, json_path),
            batch=batch,  # type: ignore[arg-type]
            view=view,
            coco_image_id=int(ann["image_id"]),
            coco_category_id=int(cat_id),
            keypoints=keypoints,
            bbox=bbox,
            image_width=width,
            image_height=height,
            filename_meta=filename_meta,
            mask_path=mask_path,
        )


# --------------------------------------------------------- batch / view discovery


# Canonical relative paths inside the dataset root, keyed by (batch, view).
# Multiple JSONs per cell are supported (e.g. B2 has Rear + Rear_2).
COCO_PATHS: dict[tuple[str, Literal["side", "rear"]], tuple[str, ...]] = {
    ("B2", "side"): (
        "Vector/B2/Side/data/Side/COCO_Side.json",
        "Vector/B2/Side/data/Side_2/COCO_Side_2.json",
    ),
    ("B2", "rear"): (
        "Vector/B2/Rear/data/Rear/COCO_Rear.json",
        "Vector/B2/Rear/data/Rear_2/COCO_Rear_2.json",
    ),
    ("B3", "side"): ("Vector/B3/Side/data/COCO_Side.json",),
    ("B3", "rear"): ("Vector/B3/Rear/data/COCO_B3_rear.json",),
    ("B4", "side"): ("Vector/B4/Side/data/coco_b4_side.json",),
    ("B4", "rear"): ("Vector/B4/Rear/data/coco_b4_rear.json",),
}


# The dataset is delivered inside an extra ``www.acmeai.tech ...`` folder; the
# caller normally passes that as the root. If they pass one level up, this
# helper finds it.
_INNER_ROOT_NAME = "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV"


def resolve_dataset_root(path: str | Path) -> Path:
    """Return the directory that contains ``Vector/`` and ``Pixel/``."""
    p = Path(path)
    if (p / "Vector").is_dir():
        return p
    inner = p / _INNER_ROOT_NAME
    if (inner / "Vector").is_dir():
        return inner
    raise FileNotFoundError(
        f"Could not find Vector/ under {p} or {inner}. "
        "Pass the dataset root (the directory containing Vector/ and Pixel/)."
    )


def iter_samples(
    dataset_root: str | Path,
    *,
    batches: tuple[str, ...] = ("B3", "B4"),
    views: tuple[Literal["side", "rear"], ...] = ("side",),
) -> Iterator[KaggleSample]:
    """Iterate samples across selected ``(batch, view)`` cells.

    Defaults to ``B3+B4 / side`` — the subset with weight labels and the
    documented 9-keypoint anatomy. Pass other combinations explicitly when
    you want them.
    """
    root = resolve_dataset_root(dataset_root)
    for batch in batches:
        for view in views:
            key = (batch, view)
            if key not in COCO_PATHS:
                log.warning("No COCO path registered for %s; skipping", key)
                continue
            for rel in COCO_PATHS[key]:
                json_path = root / rel
                if not json_path.exists():
                    log.warning("Missing COCO file: %s", json_path)
                    continue
                yield from parse_coco_file(
                    json_path, batch=batch, view=view, dataset_root=root,
                )
