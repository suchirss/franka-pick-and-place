import cv2
import asyncio


class WebcamAcquisition:
    _instance = None

    def __new__(cls, src=0, num_frames=10, wait_time_s=1):
        if cls._instance is not None:
            raise RuntimeError("WebcamAcquisition already exists — only one instance allowed")

        cls._instance = super().__new__(cls) # return first (and only) instance to _instance

        # make input parameters available to self
        cls._instance._num_frames = num_frames
        cls._instance._src = src
        cls._instance._wait_time_s = wait_time_s
        cls._instance._frame = None

        # initialize webcam when class instance is created
        cls._instance._initialize_webcam()

        return cls._instance

    def _initialize_webcam(self):
        self._cap = cv2.VideoCapture(self._src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Webcam source failed to open: {self._src}")

    def read_frame(self):
        ret, self._frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame from src: {self._src}")

        return self._frame

    def _display_frame(self):
        if self._frame is None:
            raise RuntimeError("No frame available. Call read_frame first")
        cv2.imshow("Output Video Capture", self._frame)
        cv2.waitKey(1)

    def _destroy_instance(self):
        self._cap.release()
        cv2.destroyAllWindows()
        WebcamAcquisition._instance = None

    async def _wait_async(self):
        print("...\n")
        await asyncio.sleep(self._wait_time_s)

    async def read_num_frames(self):
        for _ in range(self._num_frames):
            self.read_frame()
            self._display_frame()
            await self._wait_async()

        self._destroy_instance()


if __name__ == "__main__":
    webcam_instance = WebcamAcquisition()
    asyncio.run(webcam_instance.read_num_frames())
