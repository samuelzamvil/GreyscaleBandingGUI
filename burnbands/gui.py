"""BurnBands PySide6 GUI front end. All banding logic lives in burnbands.core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import core

# Preview responsiveness knobs — tune together: a larger preview needs a
# longer debounce to stay smooth while dragging boundary values.
PREVIEW_MAX_DIM = 1024
DEBOUNCE_MS = 100

# Minimum gap between adjacent breakpoints, in percent. 0.4% ≈ one
# luminance step, so clamped spinboxes can never trigger the core's
# collapsed-threshold validation error.
MIN_GAP_PCT = 0.4

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
        self.resize(1200, 800)

        self.gray_raw: np.ndarray | None = None  # as loaded, never inverted
        self._small_raw: np.ndarray | None = None  # downscaled copy of gray_raw
        self.source_path: Path | None = None
        self.overlay_pixmap: QPixmap | None = None
        self.custom_lowers: list[float] = []  # lower bound % per band
        self.color_overrides: dict[int, tuple[int, int, int]] = {}
        self._boundary_spinboxes: list[QDoubleSpinBox] = []

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

        self.invert_check = QCheckBox("Invert")
        self.invert_check.toggled.connect(self.on_invert_toggled)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(10, 2400)
        self.dpi_spin.setValue(300)

        self.white_bg_check = QCheckBox("White background (no transparency)")

        self.band_count = QSpinBox()
        self.band_count.setRange(2, 16)
        self.band_count.setValue(4)
        self.band_count.valueChanged.connect(self.on_band_count_changed)

        self.even_radio = QRadioButton("Even split")
        self.custom_radio = QRadioButton("Custom boundaries")
        self.even_radio.setChecked(True)
        self.even_radio.toggled.connect(self.on_mode_changed)

        self.band_table = QTableWidget(0, 3)
        self.band_table.setHorizontalHeaderLabels(["Color", "Lower %", "Coverage"])
        self.band_table.verticalHeader().setVisible(False)
        self.band_table.setSelectionMode(QTableWidget.NoSelection)
        header = self.band_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        self.export_button = QPushButton("Export Bands…")
        self.export_button.clicked.connect(self.export_bands)
        self.export_button.setEnabled(False)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Bands:", self.band_count)
        form.addRow("DPI:", self.dpi_spin)

        controls = QVBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.file_label)
        controls.addWidget(self.invert_check)
        controls.addLayout(form)
        controls.addWidget(self.even_radio)
        controls.addWidget(self.custom_radio)
        controls.addWidget(self.band_table, stretch=1)
        controls.addWidget(self.white_bg_check)
        controls.addWidget(self.export_button)
        controls.addWidget(self.status_label)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setFixedWidth(340)

        layout = QHBoxLayout()
        layout.addWidget(self.preview, stretch=1)
        layout.addWidget(controls_widget)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.rebuild_band_table()

    # --- derived state -------------------------------------------------

    def gray_full(self) -> np.ndarray | None:
        if self.gray_raw is None:
            return None
        return 255 - self.gray_raw if self.invert_check.isChecked() else self.gray_raw

    def gray_small(self) -> np.ndarray | None:
        if self._small_raw is None:
            return None
        return 255 - self._small_raw if self.invert_check.isChecked() else self._small_raw

    def current_percentages(self) -> list[float]:
        if self.even_radio.isChecked():
            return core.even_breakpoints(self.band_count.value())
        return [*self.custom_lowers, 100.0]

    def band_color(self, index: int) -> tuple[int, int, int]:
        return self.color_overrides.get(index, PALETTE[index % len(PALETTE)])

    # --- band table ------------------------------------------------------

    def rebuild_band_table(self) -> None:
        """Recreate table rows for the current band count and mode."""
        n = self.band_count.value()
        custom = self.custom_radio.isChecked()
        # (Re)derive lower bounds from an even split whenever the count
        # changes or the stored list is stale.
        if len(self.custom_lowers) != n:
            self.custom_lowers = core.even_breakpoints(n)[:-1]

        self._boundary_spinboxes = []
        self.band_table.setRowCount(n)
        for i in range(n):
            color_button = QPushButton()
            color_button.setFixedSize(40, 22)
            self._style_color_button(color_button, i)
            color_button.clicked.connect(
                lambda _=False, idx=i: self.pick_color(idx)
            )
            self.band_table.setCellWidget(i, 0, color_button)

            spin = QDoubleSpinBox()
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setSuffix(" %")
            spin.setValue(self.custom_lowers[i])
            spin.setEnabled(custom and i > 0)  # band 0 is locked at 0%
            spin.valueChanged.connect(
                lambda value, idx=i: self.on_boundary_changed(idx, value)
            )
            self._boundary_spinboxes.append(spin)
            self.band_table.setCellWidget(i, 1, spin)

            cov_item = QTableWidgetItem("—")
            cov_item.setFlags(Qt.ItemIsEnabled)
            cov_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.band_table.setItem(i, 2, cov_item)

        self._reclamp_boundaries()

    def _style_color_button(self, button: QPushButton, index: int) -> None:
        r, g, b = self.band_color(index)
        button.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #666;"
        )

    def _reclamp_boundaries(self) -> None:
        """Constrain each boundary spinbox between its neighbors so the
        breakpoint list is monotonic by construction — invalid states are
        unreachable rather than flagged after the fact."""
        lowers = self.custom_lowers
        for i, spin in enumerate(self._boundary_spinboxes):
            prev = lowers[i - 1] if i > 0 else 0.0
            nxt = lowers[i + 1] if i + 1 < len(lowers) else 100.0
            spin.blockSignals(True)
            spin.setRange(prev + MIN_GAP_PCT, nxt - MIN_GAP_PCT)
            if i == 0:
                spin.setRange(0.0, 0.0)  # locked
            spin.setValue(lowers[i])
            spin.blockSignals(False)

    # --- slots -------------------------------------------------------

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            self.gray_raw = core.load_grayscale(path)
        except core.BandingError as exc:
            QMessageBox.critical(self, "BurnBands", str(exc))
            return
        self._small_raw = downscale(self.gray_raw)
        self.source_path = Path(path)
        h, w = self.gray_raw.shape
        self.file_label.setText(f"{self.source_path.name}  ({w}×{h})")
        self.export_button.setEnabled(True)
        self.schedule_update()

    def on_band_count_changed(self) -> None:
        self.custom_lowers = []  # force even re-derivation for the new count
        self.rebuild_band_table()
        self.schedule_update()

    def on_mode_changed(self) -> None:
        custom = self.custom_radio.isChecked()
        for i, spin in enumerate(self._boundary_spinboxes):
            spin.setEnabled(custom and i > 0)
        if not custom:
            self.custom_lowers = core.even_breakpoints(self.band_count.value())[:-1]
            self._reclamp_boundaries()
        self.schedule_update()

    def on_boundary_changed(self, index: int, value: float) -> None:
        self.custom_lowers[index] = value
        self._reclamp_boundaries()
        self.schedule_update()

    def on_invert_toggled(self) -> None:
        self.schedule_update()

    def pick_color(self, index: int) -> None:
        current = QColor(*self.band_color(index))
        color = QColorDialog.getColor(current, self, f"Band {index} color")
        if not color.isValid():
            return
        self.color_overrides[index] = (color.red(), color.green(), color.blue())
        button = self.band_table.cellWidget(index, 0)
        if button is not None:
            self._style_color_button(button, index)
        self.schedule_update()

    # --- preview -------------------------------------------------------

    def schedule_update(self) -> None:
        if self.gray_raw is not None:
            self.debounce.start()

    def update_preview(self) -> None:
        small = self.gray_small()
        if small is None:
            return
        try:
            thresholds = core.validate_breakpoints(self.current_percentages())
        except core.BandingError as exc:
            self.status_label.setText(str(exc))
            return
        self.status_label.setText("")
        masks = core.band_masks(small, thresholds)
        colors = [self.band_color(i) for i in range(len(masks))]
        overlay = core.make_overlay(small, masks, colors)
        self.overlay_pixmap = array_to_pixmap(overlay)
        self.rescale_preview()

        for i, pct in enumerate(core.coverage(masks)):
            item = self.band_table.item(i, 2)
            if item is not None:
                item.setText(f"{pct:.1f} %")

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

    # --- export ----------------------------------------------------------

    def export_bands(self) -> None:
        gray = self.gray_full()
        if gray is None:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        if not out_dir:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.export_button.setEnabled(False)
        try:
            manifest = core.export_bands(
                gray,
                self.current_percentages(),
                out_dir,
                dpi=self.dpi_spin.value(),
                white_bg=self.white_bg_check.isChecked(),
                source_name=self.source_path.name if self.source_path else "",
                invert=self.invert_check.isChecked(),
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
