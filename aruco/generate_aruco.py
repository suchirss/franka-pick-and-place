"""
generate_aruco.py

This program generates valid ArUco marker images using OpenCV's predefined
ArUco dictionaries. A marker can be created in three different ways:

1. --random
   Selects a random valid marker ID from the chosen dictionary and generates
   the corresponding marker image.

2. --bits
   Accepts a 4x4 bitstream (16 binary values) representing a conceptual
   payload grid. Because ArUco dictionaries only support specific marker
   patterns, the bitstream is deterministically mapped to a valid marker ID
   using a SHA-256 hash. The resulting integer is reduced modulo the number
   of markers in the selected dictionary to guarantee a valid ID.

       marker_id = sha256(bitstream) % dictionary_size

   This ensures the same bitstream always produces the same marker ID.

3. --id
   Directly generates the marker corresponding to the specified dictionary ID.

The script then renders the marker image using OpenCV's ArUco generation
functions and writes it to a PNG file. The marker includes the standard
ArUco black border (borderBits=1), meaning a DICT_4X4_* dictionary produces
a 4x4 data grid surrounded by a border, resulting in a 6x6 module marker.

This design guarantees that all generated markers are valid members of the
chosen dictionary and can be reliably detected by OpenCV's
cv2.aruco.detectMarkers() function.

Examples:
python generate_aruco.py --random
python generate_aruco.py --dict DICT_4X4_250 --random --size 800 --out marker.png --show

python generate_aruco.py --bits 0101110000111001
python generate_aruco.py --dict DICT_4X4_250 --bits 0101110000111001 --out marker.png --size 800 --show

python generate_aruco.py --id 23
python generate_aruco.py --dict DICT_4X4_250 --id 23 --size 800 --out marker.png --show
"""

import argparse
import hashlib
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

def bits_to_valid_id(bits: str, num_markers: int) -> int:
    digest = hashlib.sha256(bits.encode("ascii")).digest()
    x = int.from_bytes(digest, "big")
    return x % num_markers

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

def main():
    p = argparse.ArgumentParser(description="Generate an ArUco marker image from --id, --random, or --bits.")
    p.add_argument("--dict", type=str, default="DICT_4X4_250", help="ArUco dictionary name")
    p.add_argument("--size", type=int, default=800, help="Marker image size in pixels")
    p.add_argument("--border", type=int, default=1, help="Border thickness in bits/modules (1 is standard)")
    p.add_argument("--out", type=str, default=None, help="Output filename (png)")
    p.add_argument("--show", action="store_true", help="Show the generated marker in a window")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int, help="Explicit marker ID")
    g.add_argument("--random", action="store_true", help="Generate a random valid marker ID from the dictionary")
    g.add_argument("--bits", type=str, help="Bitstream (must match dictionary grid size; for DICT_4X4_* this is 16 bits)")

    args = p.parse_args()

    aruco_dict = get_aruco_dict(args.dict)
    num_markers = int(aruco_dict.bytesList.shape[0])

    marker_id: int
    mode: str

    if args.random:
        marker_id = secrets.randbelow(num_markers)
        mode = "random"
    elif args.bits is not None:
        msize = dict_marker_size_from_name(args.dict)
        if msize is None:
            raise ValueError(f"Can't infer marker size from dict '{args.dict}'. Use a DICT_4X4_* dict for bitstreams.")
        bits = normalize_bits(args.bits)
        expected = msize * msize
        if len(bits) != expected:
            raise ValueError(f"--bits must contain exactly {expected} bits for {args.dict} (got {len(bits)}).")
        marker_id = bits_to_valid_id(bits, num_markers)
        mode = "bits"
    else:
        marker_id = int(args.id)
        if marker_id < 0 or marker_id >= num_markers:
            raise ValueError(f"--id must be in [0, {num_markers - 1}] for {args.dict}.")
        mode = "id"

    marker = generate_marker_image(aruco_dict, marker_id, args.size, args.border)

    out_name = args.out or f"marker_{args.dict}_{marker_id}.png"
    ok = cv2.imwrite(out_name, marker)
    if not ok:
        raise RuntimeError(f"Failed to write {out_name}")

    print(f"mode={mode}")
    print(f"dict={args.dict} numMarkers={num_markers} borderBits={args.border}")
    if mode == "bits":
        print(f"bits={normalize_bits(args.bits)}")
    print(f"marker_id={marker_id}")
    print(f"wrote={out_name}")

    if args.show:
        cv2.imshow(f"ArUco {marker_id}", marker)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)