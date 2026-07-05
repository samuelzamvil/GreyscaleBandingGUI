"""BurnBands PySide6 GUI front end. All banding logic lives in burnbands.core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import core

# Preview responsiveness knobs — tune together: a larger preview needs a
# longer debounce to stay smooth while dragging boundary values.
PREVIEW_MAX_DIM = 1024
DEBOUNCE_MS = 100

# Auto-assigned band overlay colors (high-contrast, 16 entries).
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (128, 0, 0), (170, 255, 195), (128, 128, 0),
]


def downscale(gray: np.ndarray, max_dim: int = PREVIEW_MAX_DIM) -> np.ndarray:
    """Nearest-neighbor downscale so preview tonal values match the source
    exactly (smooth resampling would invent intermediate grays and shift
    pixels across band boundaries)."""
    h, w = gray.shape
    scale = max(h, w) / max_dim
    if scale <= 1:
        return gray
    size = (max(1, round(w / scale)), max(1, round(h / scale)))
    return np.asarray(Image.fromarray(gray).resize(size, Image.NEAREST))


def array_to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    image = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BurnBands")
        self.resize(1100, 750)

        self.gray_full: np.ndarray | None = None
        self.gray_small: np.ndarray | None = None
        self.source_path: Path | None = None
        self.overlay_pixmap: QPixmap | None = None

        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(DEBOUNCE_MS)
        self.debounce.timeout.connect(self.update_preview)

        # --- preview (left) ---
        self.preview = QLabel("Open an image to begin")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(400, 400)

        # --- controls (right) ---
        self.open_button = QPushButton("Open Image…")
        self.open_button.clicked.connect(self.open_image)
        self.file_label = QLabel("No image loaded")
        self.file_label.setWordWrap(True)

        self.band_count = QSpinBox()
        self.band_count.setRange(2, 16)
        self.band_count.setValue(4)
        self.band_count.valueChanged.connect(self.schedule_update)

        self.export_button = QPushButton("Export Bands…")
        self.export_button.clicked.connect(self.export_bands)
        self.export_button.setEnabled(False)

        controls = QVBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.file_label)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Bands:"))
        controls.addWidget(self.band_count)
        controls.addStretch(1)
        controls.addWidget(self.export_button)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setFixedWidth(320)

        layout = QHBoxLayout()
        layout.addWidget(self.preview, stretch=1)
        layout.addWidget(controls_widget)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    # --- helpers -----------------------------------------------------

    def current_percentages(self) -> list[float]:
        return core.even_breakpoints(self.band_count.value())

    def band_colors(self, n: int) -> list[tuple[int, int, int]]:
        return [PALETTE[i % len(PALETTE)] for i in range(n)]

    # --- slots -------------------------------------------------------

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            self.gray_full = core.load_grayscale(path)
        except core.BandingError as exc:
            QMessageBox.critical(self, "BurnBands", str(exc))
            return
        self.gray_small = downscale(self.gray_full)
        self.source_path = Path(path)
        h, w = self.gray_full.shape
        self.file_label.setText(f"{self.source_path.name}  ({w}×{h})")
        self.export_button.setEnabled(True)
        self.schedule_update()

    def schedule_update(self) -> None:
        if self.gray_small is not None:
            self.debounce.start()

    def update_preview(self) -> None:
        if self.gray_small is None:
            return
        try:
            thresholds = core.validate_breakpoints(self.current_percentages())
        except core.BandingError:
            return
        masks = core.band_masks(self.gray_small, thresholds)
        overlay = core.make_overlay(
            self.gray_small, masks, self.band_colors(len(masks))
        )
        self.overlay_pixmap = array_to_pixmap(overlay)
        self.rescale_preview()

    def rescale_preview(self) -> None:
        if self.overlay_pixmap is None:
            return
        self.preview.setPixmap(
            self.overlay_pixmap.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.FastTransformation
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.rescale_preview()

    def export_bands(self) -> None:
        if self.gray_full is None:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        if not out_dir:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.export_button.setEnabled(False)
        try:
            manifest = core.export_bands(
                self.gray_full,
                self.current_percentages(),
                out_dir,
                source_name=self.source_path.name if self.source_path else "",
            )
        except core.BandingError as exc:
            QMessageBox.critical(self, "BurnBands", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.export_button.setEnabled(True)
        QMessageBox.information(
            self,
            "BurnBands",
            f"Exported {len(manifest.bands)} bands to:\n{out_dir}",
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
