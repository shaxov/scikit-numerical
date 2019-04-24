from numerical.area.bounds import _base
import matplotlib.pyplot as plt


class _CartesianCoordinates:
    COORDS_TYPE = "cartesian"


_Iterable = _base.BoundaryIterable
_Integrable = _base.BaseIntegrableBoundary
_Cartesian = _CartesianCoordinates


class LineBoundary1D(_Iterable, _Integrable, _Cartesian):
    def __init__(self, x_start, x_end):
        super().__init__(
            [
                _base.LineBoundary(x_start, x_end),
            ]
        )
        self._set_described_rect()

    def _set_described_rect(self):
        self._described_rect = self._bounds

    def get_described_rect(self):
        return self._described_rect

    def plot(self, save_path=None):
        self._bounds[0].plot(save_path)


class LineBoundary2D(_Iterable, _Integrable, _Cartesian):
    def __init__(self,
                 x_start, x_end,
                 y_start, y_end):
        super().__init__(
            [
                _base.LineBoundary(x_start, x_end),
                _base.LineBoundary(y_start, y_end),
            ]
        )
        self._set_described_rect()

    def _set_described_rect(self):
        self._described_rect = self._bounds

    def get_described_rect(self):
        return self._described_rect

    def plot(self, save_path=None):
        x_start, x_end = self._bounds[0].start, self._bounds[0].end
        y_start, y_end = self._bounds[1].start, self._bounds[1].end
        x_start2, x_end2 = self._described_rect[0].start, self._described_rect[0].end
        y_start2, y_end2 = self._described_rect[1].start, self._described_rect[1].end

        plt.hlines([y_start, y_end], x_start, x_end, colors='black', linestyle='-', linewidth=3)
        plt.hlines([y_start2, y_end2], x_start2, x_end2, colors='red', linestyle='--', linewidth=1)
        plt.vlines([x_start, x_end], y_start, y_end, colors='black', linestyle='-', linewidth=3)
        plt.vlines([x_start2, x_end2], y_start2, y_end2, colors='red', linestyle='--', linewidth=1)

        plt.legend(('Defined boundary', 'Described rect'), loc='upper right')

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class LineBoundary3D(_Iterable, _Integrable, _Cartesian):
    def __init__(self,
                 x_start, x_end,
                 y_start, y_end,
                 z_start, z_end):
        super().__init__(
            [
                _base.LineBoundary(x_start, x_end),
                _base.LineBoundary(y_start, y_end),
                _base.LineBoundary(z_start, z_end),
            ]
        )
        self._set_described_rect()

    def _set_described_rect(self):
        self._described_rect = self._bounds

    def get_described_rect(self):
        return self._described_rect

    def plot(self, *args):
        raise NotImplementedError
