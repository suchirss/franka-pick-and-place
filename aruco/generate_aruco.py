"""
generate_aruco.py

This program generates ArUco marker images using OpenCV's predefined
ArUco dictionaries. All generated markers include a black detection border
and an additional white outer margin for improved visibility and printing.

Markers can be created in four different ways:

1. --random
   Selects a random valid marker ID from the chosen predefined OpenCV ArUco
   dictionary and generates that marker.

2. --id
   Directly generates the specified predefined OpenCV ArUco marker ID.

3. --range START END
   Generates all marker IDs in the inclusive range [START, END].

4. --bits
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

   The generated image includes a standard black border around the payload
   and a white outer margin.

Output:
- All generated images are saved in the "markers/" directory.
- Filenames:
    marker_<id>.png        (for id, random, and range modes)
    marker_bits.png       (for bits mode)

Notes:
- The --id, --random, and --range modes generate valid predefined ArUco markers.
- The --bits mode generates an image with the exact requested bit pattern,
  but it is NOT guaranteed to be a valid marker from a predefined ArUco
  dictionary, so OpenCV ArUco detection may not recognize it as such.

Examples:
python generate_aruco.py --random
python generate_aruco.py --dict DICT_4X4_250 --random --size 800 --show

python generate_aruco.py --id 23
python generate_aruco.py --dict DICT_4X4_250 --id 23 --size 800 --show

python generate_aruco.py --range 0 9
python generate_aruco.py --dict DICT_4X4_250 --range 0 9 --size 800

python generate_aruco.py --bits 1010010110100101
python generate_aruco.py --dict DICT_4X4_250 --bits 1010010110100101 --size 800 --show
"""

import os
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
    return "".join(c for c in bit_str if c in "01")

def generate_marker_image(aruco_dict, marker_id: int, side_px: int, border_bits: int):
    """
    Generate ArUco marker using OpenCV and add a white outer border.
    """
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px)
    border_px = max(10, side_px // 10)

    marker_with_border = cv2.copyMakeBorder(
        marker,
        border_px,
        border_px,
        border_px,
        border_px,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return marker_with_border


def bits_to_grid(bits: str, msize: int) -> np.ndarray:
    vals = [1 if c == "1" else 0 for c in bits]
    return np.array(vals, dtype=np.uint8).reshape((msize, msize))

def generate_custom_bits_marker(bits: str, msize: int, side_px: int, border_bits: int) -> np.ndarray:
    grid = bits_to_grid(bits, msize)

    total_modules = msize + 2 * border_bits
    module_px = side_px // total_modules
    actual_side = module_px * total_modules

    if module_px <= 0:
        raise ValueError("Image size too small for marker grid and border.")

    img = np.zeros((actual_side, actual_side), dtype=np.uint8)

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

    if actual_side != side_px:
        img = cv2.resize(img, (side_px, side_px), interpolation=cv2.INTER_NEAREST)

    border_px = max(10, side_px // 10)
    img = cv2.copyMakeBorder(
        img,
        border_px,
        border_px,
        border_px,
        border_px,
        cv2.BORDER_CONSTANT,
        value=255
    )

    return img

def print_bit_grid(bits: str, msize: int):
    print("payload_grid:")
    for r in range(msize):
        row = bits[r * msize:(r + 1) * msize]
        print(" ".join(row))

def rotate_grid_90_cw(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=3)

def render_marker_to_grid(aruco_dict, marker_id: int, msize: int, border_bits: int = 1) -> np.ndarray:
    total_modules = msize + 2 * border_bits
    side_px = total_modules * 20

    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px)

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
    g.add_argument("--bits", type=str, help="Direct payload bitstream")
    g.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="Generate a range of marker IDs (inclusive)")

    args = p.parse_args()
    aruco_dict = get_aruco_dict(args.dict)
    num_markers = int(aruco_dict.bytesList.shape[0])
    marker = None
    mode = None
    output_dir = "markers"
    os.makedirs(output_dir, exist_ok=True)

    if args.range is not None:
        start_id, end_id = args.range

        if start_id < 0 or end_id >= num_markers or start_id > end_id:
            raise ValueError(f"--range must be within [0, {num_markers - 1}] and START <= END")

        print(f"Generating markers from {start_id} to {end_id}")

        for marker_id in range(start_id, end_id + 1):
            marker = generate_marker_image(aruco_dict, marker_id, args.size, args.border)

            filename = os.path.join(output_dir, f"marker_{marker_id}.png")
            cv2.imwrite(filename, marker)
            print(f"wrote={filename}")

        print("mode=range")
        return

    elif args.random:
        marker_id = secrets.randbelow(num_markers)
        marker = generate_marker_image(aruco_dict, marker_id, args.size, args.border)
        mode = "random"

    elif args.bits is not None:
        msize = dict_marker_size_from_name(args.dict)
        bits = normalize_bits(args.bits)
        expected = msize * msize
        if len(bits) != expected:
            raise ValueError(f"--bits must contain exactly {expected} bits for {args.dict} (got {len(bits)})")
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
        marker = generate_marker_image(aruco_dict, marker_id, args.size, args.border)
        mode = "id"

    if mode == "bits":
        out_name = args.out or f"marker_bits.png"
    else:
        out_name = args.out or f"marker_{marker_id}.png"
    
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, marker)

    print(f"mode={mode}")
    print(f"wrote={out_path}")

    if args.show:
        cv2.imshow("Marker", marker)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()