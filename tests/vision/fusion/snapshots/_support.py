"""Support."""


def points(x_min, y_min, x_max, y_max):
    return [{"x": x_min, "y": y_min}, {"x": x_max, "y": y_min}, {"x": x_max, "y": y_max}, {"x": x_min, "y": y_max}]
