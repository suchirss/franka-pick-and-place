import cv2
import numpy as np


class WarpPlane:
    _instance = None

    # grid defaults
    _DEFAULT_GRID_ROWS = 6
    _DEFAULT_GRID_COLS = 5
    _DEFAULT_GRID_ID_START = 15
    _DEFAULT_ARUCO_DICT = cv2.aruco.DICT_4X4_250
    _DEFAULT_OUTPUT_PX_PER_CELL = 100

    def __new__(cls, grid_rows=None, grid_cols=None, cell_size=None, grid_id_start=None, output_px_per_cell=None):
        if cls._instance is not None:
            raise RuntimeError("WarpPlane already exists — only one instance allowed")

        cls._instance = super().__new__(cls)

        # initialize defaults
        cls._instance._grid_rows = grid_rows or cls._DEFAULT_GRID_ROWS
        cls._instance._grid_cols = grid_cols or cls._DEFAULT_GRID_COLS
        cls._instance._grid_id_start = grid_id_start if grid_id_start is not None else cls._DEFAULT_GRID_ID_START
        cls._instance._output_px_per_cell = output_px_per_cell or cls._DEFAULT_OUTPUT_PX_PER_CELL

        cls._instance._homography = None
        cls._instance._inverse_homography = None
        cls._instance._output_size = None

        cls._instance._build_grid_id_map()

        return cls._instance

    # map each grid marker ID to its row and column position
    def _build_grid_id_map(self):

        self._id_to_grid_pos = {}

        for row in range(self._grid_rows):
            for col in range(self._grid_cols):
                marker_id = self._grid_id_start + (row * self._grid_cols) + col
                self._id_to_grid_pos[marker_id] = (row, col)

        self._output_size = (
            self._grid_cols * self._output_px_per_cell,
            self._grid_rows * self._output_px_per_cell,
        )

    # perform the perspective homography from the detected ArUco marker
    def compute_homography(self, corners, ids):
        if ids is None or len(ids) == 0:
            return False

        src_points = []
        dst_points = []

        ids_flat = ids.flatten()

        for i, marker_id in enumerate(ids_flat):
            if marker_id not in self._id_to_grid_pos:
                continue

            row, col = self._id_to_grid_pos[marker_id]
            center_px = corners[i][0].mean(axis=0)
            src_points.append(center_px)

            dst_x = (col + 0.5) * self._output_px_per_cell
            dst_y = (row + 0.5) * self._output_px_per_cell
            dst_points.append([dst_x, dst_y])

        if len(src_points) < 4:
            return False

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)

        self._homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if self._homography is None:
            return False

        self._inverse_homography = np.linalg.inv(self._homography)
        return True

    @property  # allows is_calibrated to be called as an attribute instead of a method
    def is_calibrated(self):
        return self._homography is not None

    def pixel_to_grid(self, px_x, px_y):
        if self._homography is None:
            return None

        pt = np.array([[[px_x, px_y]]], dtype=np.float32)
        transformed_pt = cv2.perspectiveTransform(pt, self._homography)[0][0]

        col = transformed_pt[0] / self._output_px_per_cell
        row = transformed_pt[1] / self._output_px_per_cell

        # return a tuple
        return row, col

    def pixel_to_grid_cell(self, px_x, px_y):

        result = self.pixel_to_grid(px_x, px_y)
        if result is None:
            return None

        row, col = result
        row = max(0, min(int(row), self._grid_rows - 1))
        col = max(0, min(int(col), self._grid_cols - 1))

        return row, col

    def grid_to_pixel(self, row, col):
        if self._inverse_homography is None:
            return None

        dst_x = col * self._output_px_per_cell
        dst_y = row * self._output_px_per_cell
        pt = np.array([[[dst_x, dst_y]]], dtype=np.float32)
        transformed_pt = cv2.perspectiveTransform(pt, self._inverse_homography)[0][0]
        return float(transformed_pt[0]), float(transformed_pt[1])

    def warp_frame(self, frame):
        if self._homography is None:
            return None

        return cv2.warpPerspective(frame, self._homography, self._output_size)

    # verify normalization with projection
    def draw_grid_overlay(self, frame):
        if self._inverse_homography is None:
            return

        _GRID_COLOR = (0, 200, 0)
        _THICKNESS = 1

        # project column lines
        for c in range(self._grid_cols + 1):

            pt_top = self.grid_to_pixel(0, c)
            pt_bot = self.grid_to_pixel(self._grid_rows, c)

            if pt_top and pt_bot:
                cv2.line(frame, (int(pt_top[0]), int(pt_top[1])), (int(pt_bot[0]), int(pt_bot[1])),_GRID_COLOR,
                         _THICKNESS)

        # project row lines
        for r in range(self._grid_rows + 1):

            pt_left = self.grid_to_pixel(r, 0)
            pt_right = self.grid_to_pixel(r, self._grid_cols)

            if pt_left and pt_right:
                cv2.line(frame, (int(pt_left[0]), int(pt_left[1])), (int(pt_right[0]), int(pt_right[1])), _GRID_COLOR,
                         _THICKNESS)

    def draw_cell_label(self, frame, row, col, label, color=(0, 255, 255)):
        px = self.grid_to_pixel(row + 0.5, col + 0.5)

        if px is None:
            return
        cv2.putText(frame, label, (int(px[0]) - 10, int(px[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # release singleton class at EOL

    def destroy_warp_plane_instance(self):
        self._homography = None
        self._inverse_homography = None
        WarpPlane._instance = None














