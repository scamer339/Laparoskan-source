"""Russian desktop interface for viewing, editing, and educational planning.

Medical calculations are delegated to the volume/segmentation/geometry modules.
This module only draws their results and turns user gestures into LPS points.
"""

from __future__ import annotations

import csv
import json
import math
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, QRectF, QThread, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressDialog, QPushButton, QScrollArea,
    QSlider, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkFiltersCore import vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkCellPicker, vtkPolyDataMapper, vtkRenderer
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderer  # noqa: F401 - load OpenGL backend

from .case import Case
from .demo import create_demo_case
from .dicom import discover_series, load_series
from .mpr import reslice_plane
from .persistence import load_case, save_case
from .planning import evaluate_training, plan_trajectory, suggest_trajectories
from .segmentation import (build_surface, create_segment, paint_voxel,
                           region_grow_segment, threshold_segment)


STYLE = """
QMainWindow, QWidget#root { background: #09121e; color: #edf4ff; }
QWidget { color: #e8f0fc; font-family: 'Segoe UI', 'Arial'; font-size: 13px; }
QFrame#header, QFrame#sidebar, QFrame#panel, QFrame#card, QFrame#footer {
    background: #101d2d; border: 1px solid #20364d; border-radius: 13px;
}
QFrame#sidebar { border-radius: 0; border-left: 0; border-top: 0; border-bottom: 0; }
QLabel#brand { font-size: 28px; font-weight: 700; color: #f7fbff; }
QLabel#headline { font-size: 22px; font-weight: 700; color: #f7fbff; }
QLabel#muted { color: #9eb4ca; }
QLabel#metric { font-size: 17px; font-weight: 650; color: #fff; }
QPushButton { background: #172b42; border: 1px solid #36516e; border-radius: 9px;
    padding: 10px 13px; text-align: left; min-height: 20px; }
QPushButton:hover { background: #203f60; border-color: #5799e9; }
QPushButton:pressed { background: #28517b; }
QPushButton:disabled { color: #6f8194; background: #142233; border-color: #263b50; }
QPushButton#primary { background: #1764d2; border-color: #4d9bff; font-weight: 700; text-align: center; }
QPushButton#primary:hover { background: #2678ed; }
QPushButton#nav { background: transparent; border: 0; border-radius: 0; font-size: 15px;
    padding: 15px 17px; }
QPushButton#nav:hover { background: #152b43; }
QPushButton#nav[active="true"] { color: #b9ddff; background: #163657;
    border-left: 3px solid #5aa8ff; font-weight: 700; }
QComboBox, QDoubleSpinBox { background: #132538; border: 1px solid #3a5572;
    border-radius: 7px; padding: 7px; min-height: 20px; }
QComboBox QAbstractItemView { background: #15273b; color: #edf4ff; selection-background-color: #2966a8; }
QSlider::groove:horizontal { height: 6px; background: #324860; border-radius: 3px; }
QSlider::handle:horizontal { background: #60aaff; width: 17px; margin: -6px 0; border-radius: 8px; }
QTableWidget { background: #0c1826; gridline-color: #263d54; border: 0; }
QHeaderView::section { background: #172e46; padding: 8px; border: 0; }
QScrollArea { border: 0; background: transparent; }
"""


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(10)
    return frame, layout


def _heading(text: str, muted: bool = False) -> QLabel:
    label = QLabel(text)
    label.setObjectName("muted" if muted else "headline")
    label.setWordWrap(True)
    return label


class _BackgroundJob(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, argument):
        super().__init__()
        self.operation = operation
        self.argument = argument

    @Slot()
    def run(self):
        try:
            self.finished.emit(self.operation(self.argument))
        except Exception as exc:
            self.failed.emit(str(exc))


class SliceCanvas(QWidget):
    """MPR image with a physically mapped click target and mask overlay."""

    point_clicked = Signal(str, object)
    scrolled = Signal(str, int)

    def __init__(self, plane: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.plane = plane
        self.slice = None
        self.case = None
        self.window = 400.0
        self.level = 40.0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start = None
        self._image: QImage | None = None
        self._overlay: QImage | None = None
        self._crosshair_pixel: tuple[float, float] | None = None
        self.setMinimumSize(180, 170)
        self.setMouseTracking(True)
        self.setToolTip("Щёлкните, чтобы переместить перекрестие. Колесо — соседний срез.")

    def set_slice(self, mpr_slice, case: Case, window: float, level: float,
                  crosshair_lps: tuple[float, float, float]):
        self.slice = mpr_slice
        self.case = case
        self.window = float(window)
        self.level = float(level)
        pixels = np.asarray(mpr_slice.pixels, dtype=np.float32)
        lo = self.level - self.window / 2
        image = np.clip((pixels - lo) * (255.0 / max(self.window, 1.0)), 0, 255).astype(np.uint8)
        if getattr(mpr_slice, "valid", None) is not None:
            image = image.copy()
            image[~np.asarray(mpr_slice.valid, dtype=bool)] = 0
        image = np.ascontiguousarray(image)
        height, width = image.shape
        self._image = QImage(image.data, width, height, width, QImage.Format_Grayscale8).copy()
        self._overlay = self._make_overlay(mpr_slice, case, height, width)
        delta = np.asarray(crosshair_lps, dtype=float) - np.asarray(mpr_slice.origin_lps, dtype=float)
        mm = float(mpr_slice.pixel_spacing_mm)
        self._crosshair_pixel = (
            float(np.dot(delta, mpr_slice.column_axis_lps) / mm),
            float(np.dot(delta, mpr_slice.row_axis_lps) / mm),
        )
        self.update()

    def _make_overlay(self, mpr_slice, case: Case, height: int, width: int) -> QImage | None:
        labelmap = case.segmentation
        if labelmap is None or not np.any(labelmap):
            return None
        geometry = case.geometry
        # Vectorised copy of the documented physical transform. Rendering may use
        # this sampled overlay; measurements always use the original labelmap.
        rows, cols = np.mgrid[0:height, 0:width]
        origin = np.asarray(mpr_slice.origin_lps, dtype=float)
        u = np.asarray(mpr_slice.column_axis_lps, dtype=float)
        v = np.asarray(mpr_slice.row_axis_lps, dtype=float)
        mm = float(mpr_slice.pixel_spacing_mm)
        points = origin + cols[..., None] * (u * mm) + rows[..., None] * (v * mm)
        direction = np.asarray(geometry.direction_lps, dtype=float).reshape(3, 3)
        voxel = (points - np.asarray(geometry.origin_lps, dtype=float)) @ np.linalg.inv(direction).T
        voxel /= np.asarray(geometry.spacing_xyz, dtype=float)
        ijk = np.rint(voxel).astype(np.int32)
        x, y, z = ijk[..., 0], ijk[..., 1], ijk[..., 2]
        inside = ((x >= 0) & (x < labelmap.shape[2]) & (y >= 0) &
                  (y < labelmap.shape[1]) & (z >= 0) & (z < labelmap.shape[0]))
        sampled = np.zeros((height, width), dtype=np.uint8)
        sampled[inside] = labelmap[z[inside], y[inside], x[inside]]
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for segment in case.segments:
            selected = sampled == segment.id
            rgba[selected, :3] = segment.color_rgb
            rgba[selected, 3] = 95
        rgba = np.ascontiguousarray(rgba)
        return QImage(rgba.data, width, height, 4 * width, QImage.Format_RGBA8888).copy()

    def _image_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        scale = min(self.width() / self._image.width(), self.height() / self._image.height()) * self.zoom
        w, h = self._image.width() * scale, self._image.height() * scale
        return QRectF((self.width() - w) / 2 + self.pan_x,
                      (self.height() - h) / 2 + self.pan_y, w, h)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#050c14"))
        if self._image is None:
            painter.setPen(QColor("#8da8bf"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Загрузите исследование")
            return
        rect = self._image_rect()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(rect, self._image)
        if self._overlay is not None:
            painter.drawImage(rect, self._overlay)
        if self._crosshair_pixel is not None:
            col, row = self._crosshair_pixel
            x = rect.left() + col / self._image.width() * rect.width()
            y = rect.top() + row / self._image.height() * rect.height()
            painter.setPen(QPen(QColor(63, 169, 242, 190), 1))
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            painter.setPen(QPen(QColor(103, 232, 160, 170), 1))
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_start = event.position()
            return
        if self.slice is None or self._image is None or event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        rect = self._image_rect()
        if not rect.contains(event.position()):
            return
        col = (event.position().x() - rect.left()) / rect.width() * self._image.width()
        row = (event.position().y() - rect.top()) / rect.height() * self._image.height()
        self.point_clicked.emit(self.plane, tuple(float(v) for v in self.slice.pixel_to_lps(col, row)))

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self._pan_start = event.position()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_start = None
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            self.zoom = float(np.clip(self.zoom * (1.12 if event.angleDelta().y() > 0 else 1 / 1.12), 0.5, 5))
            self.update()
        else:
            self.scrolled.emit(self.plane, 1 if event.angleDelta().y() > 0 else -1)
        event.accept()


class MainWindow(QMainWindow):
    """Single-workflow desktop shell. All displayed metrics come from NeedlePlan."""

    def __init__(self, case: Case | None = None):
        super().__init__()
        self.case = case if case is not None else create_demo_case()
        self.center_lps = tuple(float(v) for v in self.case.geometry.index_to_physical(
            tuple((n - 1) / 2 for n in self.case.geometry.size_xyz)))
        self.entry_lps: tuple[float, float, float] | None = None
        self.target_lps: tuple[float, float, float] | None = None
        self.plan = None
        self.tool = "navigate"
        self.training_started: float | None = None
        self.training_mode = "learning"
        self.training_attempts = 0
        self._last_case_path: str | None = None
        self._actors: dict[int, vtkActor] = {}
        self._visible: dict[int, bool] = {}
        self._slice_canvases: list[SliceCanvas] = []
        self._slice_title_labels: list[tuple[str, QLabel, str]] = []
        self._nav_buttons: list[QPushButton] = []
        self._scene_dirty = True
        self._background_threads: list[QThread] = []
        self._background_workers: list[_BackgroundJob] = []
        self._busy_dialog: QProgressDialog | None = None
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._candidates = ()
        self.setWindowTitle("Laparoskan — хирургическое планирование и тренажёр")
        self.resize(1536, 920)
        self.setMinimumSize(1100, 680)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._refresh_all()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        root.setAcceptDrops(True)
        self.setCentralWidget(root)
        self.setAcceptDrops(True)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(70)
        h = QHBoxLayout(header)
        h.setContentsMargins(22, 8, 20, 8)
        brand = QLabel("◢  Laparoskan")
        brand.setObjectName("brand")
        h.addWidget(brand)
        h.addSpacing(24)
        subtitle = QLabel("ХИРУРГИЧЕСКОЕ ПЛАНИРОВАНИЕ\nИ ВИРТУАЛЬНЫЙ ТРЕНАЖЁР")
        subtitle.setObjectName("muted")
        h.addWidget(subtitle)
        h.addStretch()
        self.case_badge = QLabel()
        h.addWidget(self.case_badge)
        outer.addWidget(header)

        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        outer.addLayout(middle, 1)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(225)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(0, 12, 0, 12)
        nav.setSpacing(3)
        labels = [
            ("▤  Исследование", 0, None), ("⬡  3D-модель", 1, 0),
            ("◎  Планирование", 1, 1), ("◇  Тренажёр", 1, 2),
            ("▥  Результаты", 2, None), ("▣  Кейсы", 3, None),
            ("⚙  Настройки", 4, None), ("ⓘ  О программе", 5, None),
        ]
        for index, (label, page, side) in enumerate(labels):
            button = QPushButton(label)
            button.setObjectName("nav")
            button.clicked.connect(lambda _checked=False, i=index, p=page, s=side: self._navigate(i, p, s))
            nav.addWidget(button)
            self._nav_buttons.append(button)
        nav.addStretch()
        notice = QLabel("Для исследовательского\nи учебного использования")
        notice.setObjectName("muted")
        notice.setWordWrap(True)
        notice.setContentsMargins(18, 8, 12, 8)
        nav.addWidget(notice)
        middle.addWidget(sidebar)

        self.pages = QStackedWidget()
        middle.addWidget(self.pages, 1)
        self.pages.addWidget(self._build_study_page())
        self.pages.addWidget(self._build_scene_page())
        self.pages.addWidget(self._build_results_page())
        self.pages.addWidget(self._build_cases_page())
        self.pages.addWidget(self._build_settings_page())
        self.pages.addWidget(self._build_about_page())

        footer = QFrame()
        footer.setObjectName("footer")
        footer.setFixedHeight(31)
        f = QHBoxLayout(footer)
        f.setContentsMargins(16, 3, 16, 3)
        self.status = QLabel("Готово")
        self.status.setObjectName("muted")
        f.addWidget(self.status)
        f.addStretch()
        f.addWidget(QLabel("Исследовательский и учебный инструмент • Не медицинское изделие"))
        outer.addWidget(footer)
        self._navigate(0, 0, None)

    def _navigate(self, nav_index: int, page: int, side: int | None):
        if self.training_started is not None and self.training_mode == "exam" and nav_index == 2:
            self.status.setText("Во время экзамена расчёт плана доступен после завершения упражнения.")
            return
        self.pages.setCurrentIndex(page)
        if side is not None:
            self.scene_side.setCurrentIndex(side)
            self._refresh_scene()
        if page == 2:
            self._refresh_results()
        for i, button in enumerate(self._nav_buttons):
            button.setProperty("active", "true" if i == nav_index else "false")
            button.style().unpolish(button)
            button.style().polish(button)

    def _build_study_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        top = QHBoxLayout()
        text = QVBoxLayout()
        text.addWidget(_heading("Исследование"))
        text.addWidget(_heading("Три синхронные плоскости • физические координаты пациента", True))
        top.addLayout(text)
        top.addStretch()
        open_button = QPushButton("Открыть DICOM")
        open_button.clicked.connect(self._open_dicom)
        top.addWidget(open_button)
        layout.addLayout(top)
        controls, controls_layout = _card()
        line = QHBoxLayout()
        controls_layout.addLayout(line)
        line.addWidget(QLabel("Окно"))
        self.preset = QComboBox()
        self.preset.addItems(["Мягкие ткани", "Кости", "Лёгкие", "Пользовательское"])
        self.preset.currentIndexChanged.connect(self._preset_changed)
        line.addWidget(self.preset)
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(1, 4000)
        self.window_spin.setValue(400)
        self.window_spin.setSuffix(" HU")
        self.window_spin.valueChanged.connect(self._window_changed)
        line.addWidget(self.window_spin)
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setRange(-1200, 3000)
        self.level_spin.setValue(40)
        self.level_spin.setSuffix(" HU")
        self.level_spin.valueChanged.connect(self._window_changed)
        line.addWidget(self.level_spin)
        line.addStretch()
        self.coord_label = QLabel()
        self.coord_label.setObjectName("muted")
        line.addWidget(self.coord_label)
        layout.addWidget(controls)
        mpr_grid = QGridLayout()
        mpr_grid.setSpacing(12)
        names = [("axial", "АКСИАЛЬНЫЙ"), ("sagittal", "САГИТТАЛЬНЫЙ"),
                 ("coronal", "КОРОНАЛЬНЫЙ")]
        for i, (plane, title) in enumerate(names):
            card, body = _card()
            slice_title = QLabel(title)
            body.addWidget(slice_title)
            self._slice_title_labels.append((plane, slice_title, title))
            canvas = self._new_slice_canvas(plane)
            body.addWidget(canvas, 1)
            mpr_grid.addWidget(card, 0 if i < 2 else 1, i if i < 2 else 0, 1, 1 if i < 2 else 2)
        layout.addLayout(mpr_grid, 1)
        seg_card, seg_layout = _card()
        seg_layout.addWidget(QLabel("Сегментация • выберите структуру и инструмент"))
        seg_line = QHBoxLayout()
        seg_layout.addLayout(seg_line)
        self.segment_combo = QComboBox()
        seg_line.addWidget(self.segment_combo, 2)
        for label, mode in [("Кисть", "brush"), ("Стереть", "erase"), ("Линейка", "measure")]:
            b = QPushButton(label)
            b.clicked.connect(lambda _checked=False, m=mode: self._set_tool(m))
            seg_line.addWidget(b)
        self.threshold_button = QPushButton("Порог по плотности")
        self.threshold_button.clicked.connect(self._apply_threshold)
        seg_line.addWidget(self.threshold_button)
        self.grow_button = QPushButton("Вырастить область")
        self.grow_button.setToolTip("Выберите допуск по HU, затем точку внутри структуры на срезе")
        self.grow_button.clicked.connect(self._select_region_grow)
        seg_line.addWidget(self.grow_button)
        undo = QPushButton("↶")
        undo.setToolTip("Отменить изменение разметки")
        undo.clicked.connect(self._undo_edit)
        seg_line.addWidget(undo)
        redo = QPushButton("↷")
        redo.setToolTip("Повторить изменение разметки")
        redo.clicked.connect(self._redo_edit)
        seg_line.addWidget(redo)
        add_segment = QPushButton("+ Структура")
        add_segment.clicked.connect(self._add_segment)
        seg_line.addWidget(add_segment)
        layout.addWidget(seg_card)
        return page

    def _new_slice_canvas(self, plane: str) -> SliceCanvas:
        canvas = SliceCanvas(plane)
        canvas.point_clicked.connect(self._on_slice_point)
        canvas.scrolled.connect(self._on_slice_scroll)
        self._slice_canvases.append(canvas)
        return canvas

    def _build_scene_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        title_row = QHBoxLayout()
        self.scene_title = _heading("3D-модель")
        title_row.addWidget(self.scene_title)
        title_row.addStretch()
        reset = QPushButton("Сбросить вид")
        reset.clicked.connect(self._reset_camera)
        title_row.addWidget(reset)
        rebuild = QPushButton("Обновить модель")
        rebuild.clicked.connect(lambda: self._refresh_scene(force=True))
        title_row.addWidget(rebuild)
        layout.addLayout(title_row)

        content = QHBoxLayout()
        content.setSpacing(12)
        layout.addLayout(content, 1)
        viewer_card, viewer_layout = _card()
        viewer_layout.setContentsMargins(4, 4, 4, 4)
        self.vtk_widget = QVTKRenderWindowInteractor(viewer_card)
        self.vtk_widget.setMinimumHeight(330)
        viewer_layout.addWidget(self.vtk_widget, 1)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.025, 0.055, 0.095)
        self.renderer.SetBackground2(0.09, 0.17, 0.26)
        self.renderer.GradientBackgroundOn()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.Initialize()
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_3d_click, 1.0)
        self.needle_actor: vtkActor | None = None
        content.addWidget(viewer_card, 1)

        self.scene_side = QStackedWidget()
        self.scene_side.setFixedWidth(310)
        self.scene_side.addWidget(self._build_anatomy_panel())
        for panel in (self._build_planning_panel(), self._build_training_panel()):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(panel)
            self.scene_side.addWidget(scroll)
        content.addWidget(self.scene_side)

        previews = QHBoxLayout()
        previews.setSpacing(10)
        for plane, name in [("axial", "Аксиальный срез"),
                            ("sagittal", "Сагиттальный срез"),
                            ("coronal", "Корональный срез")]:
            card, body = _card()
            body.setContentsMargins(8, 6, 8, 6)
            body.addWidget(QLabel(name))
            canvas = self._new_slice_canvas(plane)
            canvas.setMinimumHeight(130)
            canvas.setMaximumHeight(210)
            body.addWidget(canvas)
            previews.addWidget(card, 1)
        layout.addLayout(previews)
        return page

    def _build_anatomy_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        body = QVBoxLayout(panel)
        body.addWidget(_heading("Анатомия"))
        body.addWidget(_heading("Сегментированные структуры", True))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.anatomy_holder = QWidget()
        self.anatomy_layout = QVBoxLayout(self.anatomy_holder)
        self.anatomy_layout.setContentsMargins(0, 0, 0, 0)
        self.anatomy_layout.addStretch()
        scroll.setWidget(self.anatomy_holder)
        body.addWidget(scroll, 1)
        body.addWidget(QLabel("Прозрачность модели"))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(80)
        self.opacity.valueChanged.connect(self._change_opacity)
        body.addWidget(self.opacity)
        hint = QLabel("Вращение: левая кнопка • Масштаб: колесо • Смещение: средняя кнопка")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        body.addWidget(hint)
        return panel

    def _build_planning_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        body = QVBoxLayout(panel)
        body.addWidget(_heading("Траектория"))
        body.addWidget(_heading("Выберите точку входа и цель на модели или срезе", True))
        self.entry_button = QPushButton("◎  Выбрать точку входа")
        self.entry_button.clicked.connect(lambda: self._set_tool("entry"))
        body.addWidget(self.entry_button)
        self.entry_point_label = QLabel("Точка входа: не выбрана")
        self.entry_point_label.setObjectName("muted")
        self.entry_point_label.setWordWrap(True)
        body.addWidget(self.entry_point_label)
        self.entry_validation_label = QLabel("Положение входа относительно поверхности тела: —")
        self.entry_validation_label.setWordWrap(True)
        body.addWidget(self.entry_validation_label)
        self.target_button = QPushButton("◉  Выбрать цель")
        self.target_button.clicked.connect(lambda: self._set_tool("target"))
        body.addWidget(self.target_button)
        self.target_point_label = QLabel("Цель: не выбрана")
        self.target_point_label.setObjectName("muted")
        self.target_point_label.setWordWrap(True)
        body.addWidget(self.target_point_label)
        self.check_button = QPushButton("Проверить траекторию")
        self.check_button.setObjectName("primary")
        self.check_button.clicked.connect(self._compute_plan)
        body.addWidget(self.check_button)
        candidate_button = QPushButton("Расчёт вариантов")
        candidate_button.setToolTip("Экспериментальные геометрические варианты по размеченной поверхности тела")
        candidate_button.clicked.connect(self._generate_candidates)
        body.addWidget(candidate_button)
        self.candidate_combo = QComboBox()
        self.candidate_combo.currentIndexChanged.connect(self._candidate_changed)
        body.addWidget(self.candidate_combo)
        self.candidate_explanation = QLabel(
            "Выберите цель, затем рассчитайте варианты. Это учебная геометрическая оценка.")
        self.candidate_explanation.setWordWrap(True)
        self.candidate_explanation.setObjectName("muted")
        body.addWidget(self.candidate_explanation)
        apply_candidate = QPushButton("Применить вариант")
        apply_candidate.clicked.connect(self._apply_candidate)
        body.addWidget(apply_candidate)
        self.depth_label = QLabel("Глубина: —")
        self.depth_label.setObjectName("metric")
        body.addWidget(self.depth_label)
        self.hit_label = QLabel("Попадание в цель: —")
        body.addWidget(self.hit_label)
        self.angle_label = QLabel("Угол: —")
        body.addWidget(self.angle_label)
        self.clearance_label = QLabel("До критической структуры: —")
        self.clearance_label.setWordWrap(True)
        body.addWidget(self.clearance_label)
        self.clearance_details = QLabel("По отдельным структурам: —")
        self.clearance_details.setObjectName("muted")
        self.clearance_details.setWordWrap(True)
        body.addWidget(self.clearance_details)
        self.collision_label = QLabel("Выберите обе точки для расчёта")
        self.collision_label.setWordWrap(True)
        body.addWidget(self.collision_label)
        body.addStretch()
        warning = QLabel("Оценка учитывает только структуры, которые размечены в данном кейсе.")
        warning.setObjectName("muted")
        warning.setWordWrap(True)
        body.addWidget(warning)
        return panel

    def _build_training_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        body = QVBoxLayout(panel)
        body.addWidget(_heading("Тренажёр"))
        body.addWidget(_heading("Выберите вход и цель, затем завершите упражнение", True))
        body.addWidget(QLabel("Режим"))
        self.training_mode_box = QComboBox()
        self.training_mode_box.addItems(["Обучение", "Экзамен"])
        body.addWidget(self.training_mode_box)
        body.addWidget(QLabel("Участник (необязательно)"))
        self.trainee_name = QLineEdit()
        self.trainee_name.setPlaceholderText("Локальный псевдоним")
        body.addWidget(self.trainee_name)
        start = QPushButton("Начать упражнение")
        start.setObjectName("primary")
        start.clicked.connect(self._start_training)
        body.addWidget(start)
        choose_entry = QPushButton("Выбрать точку входа")
        choose_entry.clicked.connect(lambda: self._set_tool("entry"))
        body.addWidget(choose_entry)
        choose_target = QPushButton("Выбрать цель")
        choose_target.clicked.connect(lambda: self._set_tool("target"))
        body.addWidget(choose_target)
        self.training_state_label = QLabel("Упражнение не начато")
        self.training_state_label.setObjectName("muted")
        self.training_state_label.setWordWrap(True)
        body.addWidget(self.training_state_label)
        submit = QPushButton("Завершить и показать результат")
        submit.clicked.connect(self._finish_training)
        body.addWidget(submit)
        body.addStretch()
        body.addWidget(_heading("В экзамене подсказки скрыты до завершения.", True))
        return panel

    def _build_results_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 20, 22, 20)
        head = QHBoxLayout()
        head.addWidget(_heading("Результаты"))
        head.addStretch()
        for name, format_name in [("Экспорт CSV", "csv"), ("Экспорт JSON", "json")]:
            button = QPushButton(name)
            button.clicked.connect(lambda _checked=False, fmt=format_name: self._export_results(fmt))
            head.addWidget(button)
        layout.addLayout(head)
        self.results_summary = _heading("Результаты сохраняются вместе с кейсом.", True)
        layout.addWidget(self.results_summary)
        self.results_table = QTableWidget(0, 12)
        self.results_table.setHorizontalHeaderLabels(
            ["Дата", "Участник", "Режим", "Цель", "Вход", "Глубина", "Ошибка глубины",
             "Ошибка угла", "Просвет", "Пересечения", "Время", "Попытки"])
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table, 1)
        return page

    def _build_cases_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.addWidget(_heading("Кейсы"))
        layout.addWidget(_heading("Локальное хранение. Исходные DICOM не изменяются и никуда не отправляются.", True))
        actions, body = _card()
        for caption, callback in [
            ("Открыть исследование DICOM…", self._open_dicom),
            ("Открыть сохранённый кейс…", self._open_case),
            ("Сохранить текущий кейс…", self._save_case),
            ("Открыть синтетический демо-кейс", self._load_demo),
        ]:
            button = QPushButton(caption)
            button.clicked.connect(callback)
            body.addWidget(button)
        layout.addWidget(actions)
        self.case_details = QLabel()
        self.case_details.setObjectName("muted")
        self.case_details.setWordWrap(True)
        layout.addWidget(self.case_details)
        layout.addStretch()
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.addWidget(_heading("Настройки"))
        layout.addWidget(_heading("Отображение и данные", True))
        card, body = _card()
        body.addWidget(QLabel("Контраст снимков настраивается на экране «Исследование»."))
        body.addWidget(QLabel("Прозрачность 3D — на экране «3D-модель»."))
        body.addWidget(QLabel("Кейсы хранятся локально по выбранному вами пути."))
        layout.addWidget(card)
        future, future_body = _card()
        future_body.addWidget(QLabel("Будущие модули"))
        future_body.addWidget(_heading("Совмещение с УЗИ и аппаратное отслеживание сейчас недоступны.", True))
        layout.addWidget(future)
        layout.addStretch()
        return page

    def _build_about_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.addWidget(_heading("О программе"))
        card, body = _card()
        body.addWidget(QLabel("Laparoskan"))
        body.addWidget(_heading("Для исследовательского и учебного использования.", True))
        info = QLabel(
            "Приложение отображает DICOM и размеченные структуры, помогает изучать "
            "пространственную траекторию и выдаёт измерения по исходной геометрии "
            "и маскам. Оно не сертифицировано как медицинское изделие и не даёт "
            "клинической гарантии безопасности вмешательства."
        )
        info.setWordWrap(True)
        body.addWidget(info)
        layout.addWidget(card)
        layout.addStretch()
        return page

    def _set_case(self, case: Case):
        previous_case = self.case
        self.case = case
        self.center_lps = tuple(float(v) for v in case.geometry.index_to_physical(
            tuple((n - 1) / 2 for n in case.geometry.size_xyz)))
        self.entry_lps = None
        self.target_lps = None
        self.plan = None
        self._candidates = ()
        self.candidate_combo.clear()
        self.candidate_explanation.setText("Выберите цель, затем рассчитайте варианты.")
        self.entry_point_label.setText("Точка входа: не выбрана")
        self.target_point_label.setText("Цель: не выбрана")
        self.entry_validation_label.setText("Положение входа относительно поверхности тела: —")
        self.training_started = None
        self._nav_buttons[2].setEnabled(True)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._scene_dirty = True
        self._refresh_all()
        self._navigate(0, 0, None)
        if previous_case is not case:
            previous_case.close()

    def _refresh_all(self):
        self.case_badge.setText(f"Кейс: {self.case.name}    •    Данные локально")
        is_ct = str(self.case.metadata.get("modality", "CT")).upper() == "CT"
        self.grow_button.setEnabled(is_ct)
        self.threshold_button.setText("Порог по плотности" if is_ct else "Порог по интенсивности")
        self.window_spin.setRange(1, 4000 if is_ct else 100000)
        self.level_spin.setRange(-1200 if is_ct else -100000,
                                 3000 if is_ct else 100000)
        self.window_spin.setSuffix(" HU" if is_ct else "")
        self.level_spin.setSuffix(" HU" if is_ct else "")
        self.case_details.setText(
            f"Название: {self.case.name}\nID: {self.case.case_id}\n"
            f"Объём: {' × '.join(map(str, self.case.geometry.size_xyz))} пикселей\n"
            f"Шаг: {' × '.join(f'{v:.3f}' for v in self.case.geometry.spacing_xyz)} мм\n"
            f"Размечено структур: {sum(np.any(self.case.segmentation == s.id) for s in self.case.segments)}"
        )
        self.segment_combo.blockSignals(True)
        self.segment_combo.clear()
        for segment in self.case.segments:
            self.segment_combo.addItem(self._segment_display_name(segment.name), segment.id)
        self.segment_combo.blockSignals(False)
        self._refresh_anatomy()
        self._refresh_slices()
        self._scene_dirty = True
        if self.pages.currentIndex() == 1:
            self._refresh_scene(force=True)
        self._refresh_results()

    @staticmethod
    def _segment_display_name(name: str) -> str:
        return {
            "Abscess": "Абсцесс", "Vessels": "Сосуды", "Liver": "Печень",
            "Other organs": "Другие органы", "Body contour": "Кожа / контур тела",
            "Bones": "Кости",
        }.get(name, name)

    def _refresh_slices(self):
        center = tuple(float(v) for v in self.center_lps)
        self.coord_label.setText("LPS: " + " / ".join(f"{v:.1f}" for v in center) + " мм")
        xyz = self.case.geometry.physical_to_index(center)
        plane_axis = {"axial": 2, "sagittal": 0, "coronal": 1}
        for plane, title_label, base_title in self._slice_title_labels:
            axis = plane_axis[plane]
            index = min(max(int(round(xyz[axis])) + 1, 1), self.case.geometry.size_xyz[axis])
            title_label.setText(f"{base_title}  ·  срез {index} / {self.case.geometry.size_xyz[axis]}")
        slices = {}
        for canvas in self._slice_canvases:
            try:
                if canvas.plane not in slices:
                    slices[canvas.plane] = reslice_plane(self.case, canvas.plane, center)
                mpr = slices[canvas.plane]
                canvas.set_slice(mpr, self.case, self.window_spin.value(),
                                 self.level_spin.value(), center)
            except Exception as exc:
                self.status.setText(f"Не удалось построить срез: {exc}")

    def _refresh_anatomy(self):
        while self.anatomy_layout.count():
            item = self.anatomy_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for segment in self.case.segments:
            checkbox = QCheckBox(self._segment_display_name(segment.name))
            checkbox.setChecked(self._visible.get(segment.id, True))
            color = QColor(*segment.color_rgb).name()
            checkbox.setStyleSheet(f"QCheckBox {{ color: {color}; font-size: 14px; padding: 6px; }}")
            checkbox.toggled.connect(lambda checked, sid=segment.id: self._set_visibility(sid, checked))
            self.anatomy_layout.addWidget(checkbox)
        self.anatomy_layout.addStretch()

    def _refresh_scene(self, force: bool = False):
        if not force and not self._scene_dirty:
            self.vtk_widget.GetRenderWindow().Render()
            return
        self._scene_dirty = False
        for actor in self._actors.values():
            self.renderer.RemoveActor(actor)
        self._actors.clear()
        labels = self.case.segmentation
        if labels is None or not np.any(labels):
            self.status.setText("Сегментация пуста: отметьте структуру для 3D-модели.")
            return
        # The mesh is display-only; every plan/measurement uses the labelmap.
        self.status.setText("Подготовка 3D-модели…")
        QApplication.processEvents()
        for segment in self.case.segments:
            if not np.any(labels == segment.id):
                continue
            try:
                polydata = build_surface(self.case, segment.id)
                if polydata is None or polydata.GetNumberOfPoints() == 0:
                    continue
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*(value / 255 for value in segment.color_rgb))
                actor.GetProperty().SetAmbient(0.20)
                actor.GetProperty().SetDiffuse(0.75)
                actor.GetProperty().SetSpecular(0.35)
                actor.GetProperty().SetSpecularPower(18)
                actor.GetProperty().SetOpacity(
                    min(0.3, self.opacity.value() / 100) if segment.name == "Body contour"
                    else self.opacity.value() / 100)
                actor.SetVisibility(self._visible.get(segment.id, True))
                self.renderer.AddActor(actor)
                self._actors[segment.id] = actor
            except Exception as exc:
                self.status.setText(f"3D-модель «{self._segment_display_name(segment.name)}»: {exc}")
        self._draw_needle()
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.status.setText(f"3D-модель: {len(self._actors)} структур")

    def _set_visibility(self, segment_id: int, visible: bool):
        self._visible[segment_id] = visible
        actor = self._actors.get(segment_id)
        if actor:
            actor.SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()

    def _change_opacity(self, value: int):
        for segment in self.case.segments:
            actor = self._actors.get(segment.id)
            if actor:
                actor.GetProperty().SetOpacity(
                    min(0.3, value / 100) if segment.name == "Body contour" else value / 100)
        self.vtk_widget.GetRenderWindow().Render()

    def _reset_camera(self):
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _preset_changed(self, index: int):
        presets = [(400, 40), (2000, 400), (1500, -600)]
        if index < len(presets):
            width, level = presets[index]
            self.window_spin.blockSignals(True)
            self.level_spin.blockSignals(True)
            self.window_spin.setValue(width)
            self.level_spin.setValue(level)
            self.window_spin.blockSignals(False)
            self.level_spin.blockSignals(False)
            self._refresh_slices()

    def _window_changed(self, _value: float):
        if self.preset.currentIndex() != 3:
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(3)
            self.preset.blockSignals(False)
        self._refresh_slices()

    def _set_tool(self, tool: str):
        self.tool = tool
        descriptions = {
            "navigate": "Щёлкните по срезу, чтобы переместить перекрестие.",
            "entry": "Выберите точку входа на 3D-модели или срезе.",
            "target": "Выберите цель внутри размеченного абсцесса.",
            "brush": "Кисть: щёлкните по срезу для разметки выбранной структуры.",
            "erase": "Ластик: щёлкните по срезу для удаления разметки.",
            "measure": "Линейка: укажите две точки на срезах.",
            "region_grow": "Рост области: укажите точку внутри структуры на срезе.",
        }
        self.status.setText(descriptions.get(tool, ""))

    def _on_slice_scroll(self, plane: str, steps: int):
        current = next((c.slice for c in self._slice_canvases if c.plane == plane and c.slice), None)
        if current is None:
            return
        step_mm = min(self.case.geometry.spacing_xyz)
        center = np.asarray(self.center_lps) + np.asarray(current.normal_lps) * step_mm * steps
        # Clamp to the imaged grid to avoid apparently blank scrolling.
        xyz = np.clip(self.case.geometry.physical_to_index(center), 0,
                      np.asarray(self.case.geometry.size_xyz) - 1)
        self.center_lps = tuple(float(v) for v in self.case.geometry.index_to_physical(xyz))
        self._refresh_slices()

    def _on_slice_point(self, plane: str, point_lps: tuple[float, float, float]):
        if self.tool == "region_grow":
            label_id = self.segment_combo.currentData()
            if label_id is None:
                return
            previous = zlib.compress(self.case.segmentation.tobytes(), 1)
            try:
                changed = region_grow_segment(
                    self.case, int(label_id), point_lps,
                    tolerance_hu=self._region_grow_tolerance_hu)
            except Exception as exc:
                self._error("Не удалось вырастить область", exc)
                return
            if changed:
                self._push_edit({
                    "kind": "full", "shape": self.case.segmentation.shape,
                    "before": previous,
                    "after": zlib.compress(self.case.segmentation.tobytes(), 1),
                })
                self._after_mask_change()
                self.status.setText(f"Добавлено {changed:,} вокселей. Проверьте маску во всех плоскостях.")
            else:
                self.status.setText("Рост области не добавил новых вокселей.")
            self.tool = "navigate"
            return
        if self.tool in {"brush", "erase"}:
            label_id = int(self.segment_combo.currentData() or 1)
            xyz = self.case.geometry.physical_to_index(point_lps)
            radius = np.ceil(4 / np.asarray(self.case.geometry.spacing_xyz)).astype(int) + 2
            lo = np.maximum(np.floor(xyz).astype(int) - radius, 0)
            hi = np.minimum(np.ceil(xyz).astype(int) + radius + 1,
                            np.asarray(self.case.geometry.size_xyz))
            region = (slice(int(lo[2]), int(hi[2])),
                      slice(int(lo[1]), int(hi[1])),
                      slice(int(lo[0]), int(hi[0])))
            previous = self.case.segmentation[region].copy()
            try:
                changed = paint_voxel(
                    self.case, point_lps, label_id, radius_mm=4, erase=self.tool == "erase")
            except Exception as exc:
                self._error("Не удалось изменить разметку", exc)
                return
            if changed:
                self._push_edit({
                    "kind": "patch", "region": region, "before": previous,
                    "after": self.case.segmentation[region].copy(),
                })
                self._after_mask_change()
            return
        if self.tool == "measure":
            previous = getattr(self, "_measure_start", None)
            if previous is None:
                self._measure_start = point_lps
                self.status.setText("Линейка: выберите вторую точку.")
            else:
                distance = float(np.linalg.norm(np.asarray(point_lps) - np.asarray(previous)))
                self.status.setText(f"Расстояние между точками: {distance:.1f} мм")
                self._measure_start = None
            return
        if self.tool in {"entry", "target"}:
            self._accept_plan_point(self.tool, point_lps)
            self.tool = "navigate"
        self.center_lps = point_lps
        self._refresh_slices()

    def _on_3d_click(self, interactor, _event):
        if self.tool not in {"entry", "target"}:
            return
        x, y = interactor.GetEventPosition()
        picker = vtkCellPicker()
        picker.SetTolerance(0.003)
        if picker.Pick(x, y, 0, self.renderer):
            point = tuple(float(v) for v in picker.GetPickPosition())
            self._accept_plan_point(self.tool, point)
            self.tool = "navigate"
        else:
            self.status.setText("Точка не попала на отображаемую структуру. Выберите её на срезе.")

    def _accept_plan_point(self, kind: str, point: tuple[float, float, float]):
        if kind == "entry":
            self.entry_lps = point
            self.entry_point_label.setText("Точка входа: " + " / ".join(f"{v:.1f}" for v in point) + " мм")
        else:
            self.target_lps = point
            self.target_point_label.setText("Цель: " + " / ".join(f"{v:.1f}" for v in point) + " мм")
            self._candidates = ()
            self.candidate_combo.clear()
            self.candidate_explanation.setText("Цель изменена; рассчитайте варианты заново.")
        self.plan = None
        self.entry_validation_label.setText("Положение входа относительно поверхности тела: требуется расчёт")
        if self.entry_lps is not None and self.target_lps is not None:
            if self.training_started is not None:
                self.training_attempts += 1
            self._draw_needle()
            if self.training_started is None or self.training_mode == "learning":
                self._compute_plan(silent=True)
        self.vtk_widget.GetRenderWindow().Render()

    def _draw_needle(self):
        if self.needle_actor is not None:
            self.renderer.RemoveActor(self.needle_actor)
            self.needle_actor = None
        if self.entry_lps is None or self.target_lps is None:
            return
        line = vtkLineSource()
        line.SetPoint1(*self.entry_lps)
        line.SetPoint2(*self.target_lps)
        line.Update()
        tube = vtkTubeFilter()
        tube.SetInputConnection(line.GetOutputPort())
        tube.SetRadius(0.7)
        tube.SetNumberOfSides(12)
        tube.CappingOn()
        tube.Update()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.38, 0.86, 1.0)
        actor.GetProperty().SetAmbient(0.4)
        actor.GetProperty().SetSpecular(0.8)
        self.renderer.AddActor(actor)
        self.needle_actor = actor

    def _compute_plan(self, _checked: bool = False, silent: bool = False):
        if self.training_started is not None and self.training_mode == "exam":
            return
        if self.entry_lps is None or self.target_lps is None:
            if not silent:
                QMessageBox.information(self, "Нужны две точки",
                                        "Выберите точку входа и цель на модели или срезах.")
            return
        try:
            self.plan = plan_trajectory(self.case, self.entry_lps, self.target_lps)
        except Exception as exc:
            self._error("Не удалось рассчитать траекторию", exc)
            return
        plan = self.plan
        if plan.entry_on_body_surface is True:
            self.entry_validation_label.setText("Вход вблизи размеченной поверхности тела (допуск — диагональ вокселя).")
            self.entry_validation_label.setStyleSheet("color: #8de1b3;")
        elif plan.entry_on_body_surface is False:
            self.entry_validation_label.setText(
                "Положение входа на поверхности тела не подтверждено. Глубина указана между выбранными точками.")
            self.entry_validation_label.setStyleSheet("color: #ffca8b; font-weight: 700;")
        else:
            self.entry_validation_label.setText(
                "Поверхность тела не размечена: положение входа относительно кожи не подтверждено.")
            self.entry_validation_label.setStyleSheet("color: #ffca8b;")
        self.depth_label.setText(f"Глубина: {plan.depth_mm:.1f} мм")
        self.hit_label.setText(
            "Попадание в размеченную цель: да" if plan.target_hit
            else "Попадание в размеченную цель: нет")
        self.angle_label.setText(f"Угол: {plan.angle_degrees:.1f}°")
        if plan.closest_clearance_mm is None:
            self.clearance_label.setText("До критической структуры: нет размеченных критических масок")
        else:
            self.clearance_label.setText(
                f"До ближайшей размеченной структуры: не менее {plan.closest_clearance_mm:.1f} мм "
                "(консервативная оценка)" +
                (". Разметка/поиск неполные." if not plan.clearance_complete else ""))
        critical_segments = [segment for segment in self.case.segments
                             if segment.name.casefold() not in {
                                 "abscess", "body contour", "skin", "body surface", "skin surface"}]
        details = []
        for segment in critical_segments:
            name = self._segment_display_name(segment.name)
            if segment.name in plan.clearances_mm:
                details.append(f"{name}: ≥ {plan.clearances_mm[segment.name]:.1f} мм")
        self.clearance_details.setText(
            "По отдельным структурам: " + ("; ".join(details) if details else "нет размеченных критических масок"))
        if plan.unmarked_critical_labels:
            self.clearance_details.setText(
                self.clearance_details.text() + "\nНет разметки: " +
                ", ".join(self._segment_display_name(v) for v in plan.unmarked_critical_labels))
        if plan.collided:
            names = ", ".join(self._segment_display_name(n) for n in plan.collision_labels)
            self.collision_label.setText(f"Обнаружено пересечение: {names}")
            self.collision_label.setStyleSheet("color: #ff9c9c; font-weight: 700;")
        else:
            self.collision_label.setText(
                "Пересечений с отмеченными критическими структурами не обнаружено.")
            self.collision_label.setStyleSheet("color: #8de1b3; font-weight: 700;")
        self.status.setText("Траектория рассчитана по физической геометрии и разметке.")

    def _generate_candidates(self):
        if self.training_started is not None and self.training_mode == "exam":
            return
        if self.target_lps is None:
            QMessageBox.information(self, "Нужна цель", "Сначала выберите цель на модели или срезе.")
            return
        try:
            target_segment = self.case.segment_by_name("Abscess")
            xyz = np.rint(self.case.geometry.physical_to_index(self.target_lps)).astype(int)
            size = self.case.geometry.size_xyz
            target_hit = (all(0 <= int(xyz[i]) < size[i] for i in range(3))
                          and int(self.case.segmentation[xyz[2], xyz[1], xyz[0]]) == target_segment.id)
        except (KeyError, ValueError):
            target_hit = False
        if not target_hit:
            QMessageBox.information(
                self, "Цель вне разметки",
                "Для расчёта вариантов выберите точку внутри размеченного абсцесса.")
            return
        target = self.target_lps
        self._start_background_job(
            lambda point: suggest_trajectories(
                self.case, point, max_candidates=4, max_entry_points=16),
            target, "Расчёт геометрических вариантов…", self._candidates_ready,
            "Не удалось рассчитать варианты")

    def _candidates_ready(self, candidates):
        self._candidates = tuple(candidates)
        self.candidate_combo.clear()
        for candidate in self._candidates:
            plan = candidate.plan
            self.candidate_combo.addItem(
                f"Вариант {candidate.rank} · {plan.depth_mm:.1f} мм · "
                + ("пересечение" if plan.collided else "без обнаруженных пересечений"))
        if not candidates:
            self.candidate_explanation.setText(
                "Варианты недоступны: разметьте поверхность тела и цель-абсцесс.")
        else:
            self._candidate_changed(0)

    def _candidate_changed(self, index: int):
        if index < 0 or index >= len(self._candidates):
            return
        candidate = self._candidates[index]
        plan = candidate.plan
        clearance = (f"не менее {candidate.clearance_lower_bound_mm:.1f} мм"
                     if candidate.clearance_lower_bound_mm is not None else "неизвестен")
        if not plan.clearance_complete:
            clearance += " (поиск ограничен)"
        collisions = ", ".join(self._segment_display_name(v) for v in plan.collision_labels)
        self.candidate_explanation.setText(
            f"Попадание в размеченную цель: {'да' if plan.target_hit else 'нет'}; "
            f"глубина {plan.depth_mm:.1f} мм; пересечения: {collisions or 'не обнаружены'}; "
            f"просвет до размеченных структур: {clearance}. "
            "Порядок: меньше пересечений, больше просвет, короче путь. "
            "Оценка экспериментальная и не является клиническим оптимумом.")

    def _apply_candidate(self):
        index = self.candidate_combo.currentIndex()
        if index < 0 or index >= len(self._candidates):
            return
        candidate = self._candidates[index]
        plan = candidate.plan
        self._accept_plan_point("entry", plan.entry_lps)
        self.center_lps = plan.target_lps
        self._refresh_slices()
        self.status.setText(f"Применён учебный вариант {candidate.rank}; проверьте его вручную.")

    def _apply_threshold(self):
        label_id = self.segment_combo.currentData()
        if label_id is None:
            return
        is_ct = str(self.case.metadata.get("modality", "CT")).upper() == "CT"
        unit = "HU" if is_ct else "интенсивность"
        minimum, maximum = (-2000, 4000) if is_ct else (-100000, 100000)
        lower, accepted = QInputDialog.getDouble(
            self, "Порог по интенсивности", f"Нижняя граница ({unit})", -100,
            minimum, maximum, 1)
        if not accepted:
            return
        upper, accepted = QInputDialog.getDouble(
            self, "Порог по интенсивности", f"Верхняя граница ({unit})", 100,
            minimum, maximum, 1)
        if not accepted:
            return
        if lower > upper:
            QMessageBox.warning(self, "Неверный диапазон", "Нижняя граница должна быть меньше верхней.")
            return
        previous = zlib.compress(self.case.segmentation.tobytes(), 1)
        try:
            changed = threshold_segment(self.case, int(label_id), lower, upper)
        except Exception as exc:
            self._error("Не удалось применить порог", exc)
            return
        if changed:
            self._push_edit({
                "kind": "full", "shape": self.case.segmentation.shape,
                "before": previous, "after": zlib.compress(self.case.segmentation.tobytes(), 1),
            })
            self._after_mask_change()
        else:
            self.status.setText("Порог не изменил разметку.")

    def _select_region_grow(self):
        if str(self.case.metadata.get("modality", "CT")).upper() != "CT":
            self.status.setText("Рост области по HU доступен только для КТ.")
            return
        if self.segment_combo.currentData() is None:
            self.status.setText("Сначала создайте и выберите структуру.")
            return
        tolerance, accepted = QInputDialog.getDouble(
            self, "Рост области", "Допуск от плотности точки (± HU)",
            35, 0, 500, 1)
        if not accepted:
            return
        self._region_grow_tolerance_hu = tolerance
        self._set_tool("region_grow")

    def _after_mask_change(self):
        """Never leave a collision or clearance conclusion from an older mask."""
        self._scene_dirty = True
        self.plan = None
        self._candidates = ()
        self.candidate_combo.clear()
        self.candidate_explanation.setText("Разметка изменена; рассчитайте варианты заново.")
        self.entry_validation_label.setText("Положение входа относительно поверхности тела: —")
        self.depth_label.setText("Глубина: требуется пересчёт")
        self.hit_label.setText("Попадание в цель: требуется пересчёт")
        self.angle_label.setText("Угол: требуется пересчёт")
        self.clearance_label.setText("До критической структуры: требуется пересчёт")
        self.clearance_details.setText("По отдельным структурам: требуется пересчёт")
        self.collision_label.setText("Пересечения: требуется пересчёт")
        self.collision_label.setStyleSheet("color: #ffca8b; font-weight: 700;")
        self._refresh_slices()
        if self.entry_lps is not None and self.target_lps is not None:
            self._compute_plan(silent=True)
        if self.plan is None:
            self.status.setText("Разметка изменена. Выводы о траектории ожидают пересчёта.")
        else:
            self.status.setText("Разметка изменена; траектория пересчитана по новой маске.")

    def _push_edit(self, action: dict):
        self._undo_stack.append(action)
        self._undo_stack = self._undo_stack[-20:]
        self._redo_stack.clear()

    def _apply_edit(self, action: dict, key: str):
        if action["kind"] == "patch":
            self.case.segmentation[action["region"]] = action[key]
        else:
            raw = zlib.decompress(action[key])
            self.case.segmentation[:] = np.frombuffer(raw, dtype=np.uint8).reshape(action["shape"])
        self.case.touch_segmentation()
        self._after_mask_change()

    def _undo_edit(self):
        if not self._undo_stack:
            self.status.setText("Нет изменений для отмены.")
            return
        action = self._undo_stack.pop()
        self._apply_edit(action, "before")
        self._redo_stack.append(action)

    def _redo_edit(self):
        if not self._redo_stack:
            self.status.setText("Нет изменений для повтора.")
            return
        action = self._redo_stack.pop()
        self._apply_edit(action, "after")
        self._undo_stack.append(action)

    def _add_segment(self):
        name, accepted = QInputDialog.getText(self, "Новая структура", "Название структуры")
        name = name.strip()
        if not accepted or not name:
            return
        try:
            segment = create_segment(self.case, name, (100, 197, 218))
        except Exception as exc:
            self._error("Не удалось создать структуру", exc)
            return
        self.segment_combo.addItem(self._segment_display_name(segment.name), segment.id)
        self.segment_combo.setCurrentIndex(self.segment_combo.count() - 1)
        self._refresh_anatomy()
        self.status.setText(f"Создана структура «{name}». Выберите кисть для разметки.")

    def _open_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка с DICOM")
        if folder:
            self._import_dicom_folder(folder)

    def _import_dicom_folder(self, folder: str):
        self._start_background_job(
            discover_series, folder, "Поиск серий DICOM…",
            self._choose_dicom_series, "Не удалось прочитать папку DICOM")

    def _choose_dicom_series(self, series):
        if not series:
            QMessageBox.information(self, "Серии не найдены",
                                    "В выбранной папке не найдены пригодные серии DICOM.")
            return
        labels = []
        for i, item in enumerate(series):
            state = "" if item.is_loadable else f" • НЕДОСТУПНА: {item.rejection_reason}"
            labels.append(f"{i + 1}. {item.modality}  {item.description}  •  "
                          f"{item.slice_count} срезов{state}")
        selected, accepted = QInputDialog.getItem(
            self, "Выбор серии", "Выберите последовательную серию", labels, 0, False)
        if not accepted:
            return
        index = labels.index(selected)
        if not series[index].is_loadable:
            QMessageBox.warning(self, "Непригодная серия",
                                f"Эта серия не будет реконструирована:\n{series[index].rejection_reason}")
            return
        self._start_background_job(
            load_series, series[index], "Загрузка объёма и проверка геометрии…",
            self._dicom_loaded, "Не удалось загрузить серию")

    def _dicom_loaded(self, case: Case):
        self._set_case(case)
        self.status.setText("Серия загружена. Проверьте срезы и разметьте интересующие структуры.")

    def _start_background_job(self, operation, argument, message, on_success, error_title):
        if self._busy_dialog is not None:
            return
        dialog = QProgressDialog(message, "", 0, 0, self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.show()
        self._busy_dialog = dialog
        thread = QThread(self)
        worker = _BackgroundJob(operation, argument)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def complete(value):
            dialog.close()
            self._busy_dialog = None
            on_success(value)

        def failed(message_text):
            dialog.close()
            self._busy_dialog = None
            self._error(error_title, ValueError(message_text))

        worker.finished.connect(complete)
        worker.failed.connect(failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._background_threads.remove(thread)
                                if thread in self._background_threads else None)
        thread.finished.connect(lambda: self._background_workers.remove(worker)
                                if worker in self._background_workers else None)
        self._background_threads.append(thread)
        self._background_workers.append(worker)
        thread.start()

    def _open_case(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть кейс", "", "Кейс Laparoskan (*.lapcase *.zip);;Все файлы (*)")
        if not path:
            return
        try:
            case = load_case(path)
        except Exception as exc:
            self._error("Не удалось открыть кейс", exc)
            return
        self._last_case_path = path
        self._set_case(case)
        self.status.setText("Кейс открыт без изменения исходного DICOM.")

    def _save_case(self):
        default = self._last_case_path or f"{self.case.name.replace(' ', '_')}.lapcase"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить кейс", default, "Кейс Laparoskan (*.lapcase)")
        if not path:
            return
        if not path.lower().endswith(".lapcase"):
            path += ".lapcase"
        try:
            save_case(self.case, path)
        except Exception as exc:
            self._error("Не удалось сохранить кейс", exc)
            return
        self._last_case_path = path
        self.status.setText(f"Кейс сохранён: {Path(path).name}")

    def _load_demo(self):
        self._set_case(create_demo_case())
        self.status.setText("Открыт синтетический демо-кейс. Это не данные пациента.")

    def _start_training(self):
        self.training_mode = "exam" if self.training_mode_box.currentIndex() == 1 else "learning"
        self.training_started = time.monotonic()
        self.training_attempts = 0
        self.entry_lps = None
        self.target_lps = None
        self.plan = None
        self.entry_point_label.setText("Точка входа: не выбрана")
        self.target_point_label.setText("Цель: не выбрана")
        self.entry_validation_label.setText("Положение входа относительно поверхности тела: —")
        self._nav_buttons[2].setEnabled(self.training_mode != "exam")
        self.depth_label.setText("Глубина: —")
        self.hit_label.setText("Попадание в цель: —")
        self.angle_label.setText("Угол: —")
        self.clearance_label.setText("До критической структуры: —")
        self.clearance_details.setText("По отдельным структурам: —")
        self.collision_label.setText("Пересечения: —")
        self._draw_needle()
        self.vtk_widget.GetRenderWindow().Render()
        self.training_state_label.setText(
            "Экзамен начат: подсказки скрыты." if self.training_mode == "exam"
            else "Обучение начато: выберите точку входа и цель.")
        self.status.setText("Упражнение начато.")

    def _finish_training(self):
        if self.training_started is None:
            QMessageBox.information(self, "Упражнение не начато", "Сначала нажмите «Начать упражнение».")
            return
        if self.entry_lps is None or self.target_lps is None:
            QMessageBox.information(self, "Нет решения", "Выберите точку входа и цель.")
            return
        self.training_attempts = max(self.training_attempts, 1)
        try:
            plan = plan_trajectory(self.case, self.entry_lps, self.target_lps)
        except Exception as exc:
            self._error("Не удалось оценить упражнение", exc)
            return
        self.plan = plan
        duration_s = round(time.monotonic() - self.training_started, 1)
        reference_entry = self.case.metadata.get("reference_entry_lps")
        reference_target = self.case.metadata.get("reference_target_lps")
        facts = evaluate_training(self.case, plan, reference_entry, reference_target)
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": self.case.case_id,
            "case_name": self.case.name,
            "user": self.trainee_name.text().strip(),
            "mode": self.training_mode,
            "entry_lps": list(self.entry_lps),
            "target_lps": list(self.target_lps),
            "target_hit": plan.target_hit,
            "entry_on_body_surface": facts["entry_on_body_surface"],
            "unmarked_critical_labels": facts["unmarked_critical_labels"],
            "entry_and_target_verified": bool(plan.target_hit and plan.entry_on_body_surface is True),
            "depth_mm": round(plan.depth_mm, 2),
            "angle_degrees": round(plan.angle_degrees, 2),
            "closest_clearance_mm": (
                round(plan.closest_clearance_mm, 2)
                if plan.closest_clearance_mm is not None else None),
            "collision_labels": list(plan.collision_labels),
            "duration_s": duration_s,
            "attempts": self.training_attempts,
            "depth_error_mm": facts.get("depth_error_mm"),
            "angle_error_degrees": facts.get("angular_error_degrees"),
            "reference_errors_available": facts["reference_errors_available"],
            "clearance_complete": facts["clearance_complete"],
            "clearance_limit_mm": facts["clearance_limit_mm"],
        }
        self.case.metadata.setdefault("results", []).append(result)
        self.training_started = None
        self._nav_buttons[2].setEnabled(True)
        self.training_state_label.setText(
            f"Упражнение завершено за {duration_s:.0f} с. "
            + ("Вход у размеченной поверхности подтверждён. "
               if plan.entry_on_body_surface is True
               else "Положение входа у поверхности не подтверждено; план неполный. ")
            + "Результат сохранён в текущем кейсе.")
        self._compute_plan(silent=True)
        self._refresh_results()
        self._navigate(4, 2, None)

    def _refresh_results(self):
        if not hasattr(self, "results_table"):
            return
        results = self.case.metadata.get("results", [])
        self.results_table.setRowCount(len(results))
        for row, result in enumerate(results):
            hit = result.get("target_hit")
            entry_ok = result.get("entry_on_body_surface")
            values = [
                result.get("timestamp", "")[:19].replace("T", " "),
                result.get("user", "") or "—",
                "Экзамен" if result.get("mode") == "exam" else "Обучение",
                "Да" if hit is True else "Нет" if hit is False else "Не определено",
                "У поверхности" if entry_ok is True else "Вне поверхности" if entry_ok is False else "Не проверен",
                f"{result.get('depth_mm', 0):.1f} мм",
                (f"{result['depth_error_mm']:.1f} мм"
                 if result.get("depth_error_mm") is not None else "Нет эталона"),
                (f"{result['angle_error_degrees']:.1f}°"
                 if result.get("angle_error_degrees") is not None else "Нет эталона"),
                (f"{result['closest_clearance_mm']:.1f} мм"
                 if result.get("closest_clearance_mm") is not None else "Нет разметки"),
                ", ".join(result.get("collision_labels", [])) or "Не обнаружены",
                f"{result.get('duration_s', 0):.0f} с",
                str(result.get("attempts", 1)),
            ]
            for column, value in enumerate(values):
                self.results_table.setItem(row, column, QTableWidgetItem(str(value)))
        trend = ""
        if len(results) >= 2:
            first = results[0].get("depth_error_mm")
            last = results[-1].get("depth_error_mm")
            if first is not None and last is not None:
                trend = f" Ошибка глубины: {first:.1f} → {last:.1f} мм."
        self.results_summary.setText(
            f"Прохождений: {len(results)}.{trend} Сохраните кейс для хранения результатов на диске.")

    def _export_results(self, format_name: str):
        results = self.case.metadata.get("results", [])
        if not results:
            QMessageBox.information(self, "Нет результатов", "Сначала завершите упражнение.")
            return
        suffix = f".{format_name}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт результатов", f"laparoskan-results{suffix}",
            "CSV (*.csv)" if format_name == "csv" else "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(suffix):
            path += suffix
        try:
            with open(path, "w", encoding="utf-8-sig" if format_name == "csv" else "utf-8",
                      newline="") as stream:
                if format_name == "json":
                    json.dump(results, stream, ensure_ascii=False, indent=2)
                else:
                    keys = list(dict.fromkeys(key for item in results for key in item))
                    writer = csv.DictWriter(stream, fieldnames=keys)
                    writer.writeheader()
                    for item in results:
                        writer.writerow({
                            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                            else value for key, value in item.items()
                        })
        except Exception as exc:
            self._error("Не удалось экспортировать результаты", exc)
            return
        self.status.setText(f"Результаты экспортированы: {Path(path).name}")

    def _error(self, title: str, error: Exception):
        QMessageBox.warning(self, title, f"{title}.\n\n{error}")
        self.status.setText(title)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.is_dir():
            self._import_dicom_folder(str(path))
            event.acceptProposedAction()
        elif path.suffix.lower() in {".lapcase", ".zip"}:
            try:
                self._set_case(load_case(str(path)))
                event.acceptProposedAction()
            except Exception as exc:
                self._error("Не удалось открыть кейс", exc)

    def closeEvent(self, event):
        if any(thread.isRunning() for thread in self._background_threads):
            self.status.setText("Подождите завершения загрузки перед закрытием приложения.")
            event.ignore()
            return
        self.case.close()
        super().closeEvent(event)
