"""
generate_aruco.py

This program generates valid ArUco marker images using OpenCV's predefined
ArUco dictionaries. A marker can be created in three different ways:

1. --random
   Selects a random valid marker ID from the chosen predefined OpenCV ArUco
   dictionary and generates that marker.

2. --id
   Directly generates the specified predefined OpenCV ArUco marker ID.

3. --bits
   Interprets the provided bitstream as a direct payload grid in row-major
   order: top-to-bottom, left-to-right.

   For example, with a 4x4 dictionary size:
       --bits 1010010110100101

   means the payload cells are:

       1 0 1 0
       0 1 0 1
       1 0 1 0
       0 1 0 1

   where:
       1 = white cell
       0 = black cell

   The generated image includes a standard black border around the payload.

Notes:
- The --id and --random modes generate valid predefined ArUco markers.
- The --bits mode generates an image with the exact requested bit pattern,
  but it is NOT guaranteed to be a valid marker from a predefined ArUco
  dictionary, so OpenCV ArUco detection may not recognize it as such.

Examples:
python generate_aruco.py --random
python generate_aruco.py --dict DICT_4X4_250 --random --size 800 --out marker.png --show

python generate_aruco.py --id 23
python generate_aruco.py --dict DICT_4X4_250 --id 23 --size 800 --out marker.png --show

python generate_aruco.py --bits 1010010110100101
python generate_aruco.py --dict DICT_4X4_250 --bits 1010010110100101 --size 800 --out marker.png --show
"""

import argparse
import secrets
import sys

import cv2
import numpy as np

def get_aruco_dict(dict_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module not found. Install: pip install opencv-contrib-python")
    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        raise ValueError(f"Unknown dict '{dict_name}'. Example: DICT_4X4_250, DICT_6X6_250, ...")
    return aruco.getPredefinedDictionary(getattr(aruco, dict_name))

def dict_marker_size_from_name(dict_name: str) -> int | None:
    # Parses "DICT_4X4_250" -> 4
    # Returns None if not in that format.
    parts = dict_name.split("_")
    if len(parts) >= 2 and "X" in parts[1]:
        try:
            a, b = parts[1].split("X")
            if a == b:
                return int(a)
        except Exception:
            return None
    return None

def normalize_bits(bit_str: str) -> str:
    # Keep only 0/1; tolerate spaces/newlines/commas/slashes.
    return "".join(c for c in bit_str if c in "01")

def generate_marker_image(aruco_dict, marker_id: int, side_px: int, border_bits: int):
    # Compatibility across OpenCV builds:
    if hasattr(cv2.aruco, "drawMarker"):
        img = np.zeros((side_px, side_px), dtype=np.uint8)
        cv2.aruco.drawMarker(aruco_dict, marker_id, side_px, img, border_bits)
        return img

    # Newer API has generateImageMarker; borderBits kw support varies
    try:
        return cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px, borderBits=border_bits)
    except TypeError:
        return cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px)

def bits_to_grid(bits: str, msize: int) -> np.ndarray:
    # Row-major: top-to-bottom, left-to-right
    vals = [1 if c == "1" else 0 for c in bits]
    return np.array(vals, dtype=np.uint8).reshape((msize, msize))

def generate_custom_bits_marker(bits: str, msize: int, side_px: int, border_bits: int) -> np.ndarray:
    """
    Render a custom marker image from a payload bitstring.

    Conventions:
    - bits are read row-major: top-to-bottom, left-to-right
    - 1 = white payload cell
    - 0 = black payload cell
    - border is black
    """
    grid = bits_to_grid(bits, msize)

    total_modules = msize + 2 * border_bits
    module_px = side_px // total_modules
    actual_side = module_px * total_modules

    if module_px <= 0:
        raise ValueError("Image size too small for marker grid and border.")

    # Start with white image, then paint border/data
    img = np.full((actual_side, actual_side), 255, dtype=np.uint8)

    # Paint full marker area black first so border is black by default
    img[:, :] = 0

    # Paint payload cells
    for r in range(msize):
        for c in range(msize):
            y0 = (r + border_bits) * module_px
            y1 = y0 + module_px
            x0 = (c + border_bits) * module_px
            x1 = x0 + module_px

            if grid[r, c] == 1:
                img[y0:y1, x0:x1] = 255
            else:
                img[y0:y1, x0:x1] = 0

    # Resize to exact requested size if integer division trimmed a few pixels
    if actual_side != side_px:
        img = cv2.resize(img, (side_px, side_px), interpolation=cv2.INTER_NEAREST)

    return img

def print_bit_grid(bits: str, msize: int):
    print("payload_grid:")
    for r in range(msize):
        row = bits[r * msize:(r + 1) * msize]
        print(" ".join(row))

def rotate_grid_90_cw(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=3)

def render_marker_to_grid(aruco_dict, marker_id: int, msize: int, border_bits: int = 1) -> np.ndarray:
    """
    Render a predefined marker and sample its inner payload grid.

    Returns:
        msize x msize array of 0/1 where:
        1 = white cell
        0 = black cell
    """
    total_modules = msize + 2 * border_bits
    side_px = total_modules * 20  # large enough for clean sampling

    img = generate_marker_image(aruco_dict, marker_id, side_px, border_bits)

    module_px = side_px // total_modules
    grid = np.zeros((msize, msize), dtype=np.uint8)

    for r in range(msize):
        for c in range(msize):
            y0 = (r + border_bits) * module_px
            y1 = y0 + module_px
            x0 = (c + border_bits) * module_px
            x1 = x0 + module_px

            cell = img[y0:y1, x0:x1]
            grid[r, c] = 1 if np.mean(cell) > 127 else 0

    return grid

def find_matching_marker_id(bits: str, aruco_dict, dict_name: str, border_bits: int = 1):
    msize = dict_marker_size_from_name(dict_name)
    if msize is None:
        raise ValueError(f"Can't infer marker size from dict '{dict_name}'.")

    target = bits_to_grid(bits, msize)
    num_markers = int(aruco_dict.bytesList.shape[0])

    for marker_id in range(num_markers):
        marker_grid = render_marker_to_grid(aruco_dict, marker_id, msize, border_bits)

        test_grid = marker_grid.copy()
        for rot in range(4):
            if np.array_equal(target, test_grid):
                return marker_id, rot
            test_grid = rotate_grid_90_cw(test_grid)

    return None, None

def main():
    p = argparse.ArgumentParser(description="Generate an ArUco marker image from --id, --random, or --bits.")
    p.add_argument("--dict", type=str, default="DICT_4X4_250", help="ArUco dictionary name")
    p.add_argument("--size", type=int, default=800, help="Marker image size in pixels")
    p.add_argument("--border", type=int, default=1, help="Border thickness in bits/modules (1 is standard)")
    p.add_argument("--out", type=str, default=None, help="Output filename (png)")
    p.add_argument("--show", action="store_true", help="Show the generated marker in a window")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int, help="Explicit marker ID from predefined dictionary")
    g.add_argument("--random", action="store_true", help="Generate a random valid predefined marker ID")
    g.add_argument("--bits", type=str, help="Direct payload bitstream, interpreted top-to-bottom, left-to-right")

    args = p.parse_args()

    aruco_dict = get_aruco_dict(args.dict)
    num_markers = int(aruco_dict.bytesList.shape[0])

    marker = None
    mode = None

    if args.random:
        marker_id = secrets.randbelow(num_markers)
        marker = generate_marker_image(aruco_dict, marker_id, args.size, args.border)
        mode = "random"

    elif args.bits is not None:
        msize = dict_marker_size_from_name(args.dict)
        if msize is None:
            raise ValueError(f"Can't infer marker size from dict '{args.dict}'. Use a DICT_4X4_* dict for bitstreams.")
        bits = normalize_bits(args.bits)
        expected = msize * msize
        if len(bits) != expected:
            raise ValueError(f"--bits must contain exactly {expected} bits for {args.dict} (got {len(bits)}).")
        marker = generate_custom_bits_marker(bits, msize, args.size, args.border)
        mode = "bits"

        matched_id, matched_rot = find_matching_marker_id(bits, aruco_dict, args.dict, args.border)

        if matched_id is not None:
            print(f"Matching Id: {matched_id}")
            print(f"Matched Id Rotation: {matched_rot * 90}")
        else:
            print("No Matching Id")

    else:
        marker_id = int(args.id)
        if marker_id < 0 or marker_id >= num_markers:
            raise ValueError(f"--id must be in [0, {num_markers - 1}] for {args.dict}.")
        marker = generate_marker_image(aruco_dict, marker_id, args.size, args.border)
        mode = "id"

    out_name = args.out
    if out_name is None:
        if mode == "bits":
            out_name = f"marker_bits_{args.dict}.png"
        else:
            out_name = f"marker_{args.dict}_{marker_id}.png"

    ok = cv2.imwrite(out_name, marker)
    if not ok:
        raise RuntimeError(f"Failed to write {out_name}")

    print(f"mode={mode}")
    print(f"dict={args.dict} numMarkers={num_markers} borderBits={args.border}")

    if mode == "bits":
        bits = normalize_bits(args.bits)
        print(f"bits={bits}")
        print_bit_grid(bits, dict_marker_size_from_name(args.dict))
        print("note=bits mode renders the exact payload pattern; it is not guaranteed to be a valid predefined ArUco marker")
    else:
        print(f"marker_id={marker_id}")

    print(f"wrote={out_name}")

    if args.show:
        title = f"Custom Bits Marker" if mode == "bits" else f"ArUco {marker_id}"
        cv2.imshow(title, marker)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)