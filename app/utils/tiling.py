from __future__ import annotations


def tile_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    origins = list(range(0, length - tile_size + 1, stride))
    if origins[-1] != length - tile_size:
        origins.append(length - tile_size)
    return origins


def clip_box_to_tile(box, tile, min_visibility) -> tuple[float, float, float, float] | None:
    box_x1, box_y1, box_x2, box_y2 = box
    tile_x1, tile_y1, tile_x2, tile_y2 = tile

    inter_x1 = max(box_x1, tile_x1)
    inter_y1 = max(box_y1, tile_y1)
    inter_x2 = min(box_x2, tile_x2)
    inter_y2 = min(box_y2, tile_y2)

    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1
    if inter_width <= 0 or inter_height <= 0:
        return None

    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)
    if box_area <= 0:
        return None
    if (inter_width * inter_height) / box_area < min_visibility:
        return None

    return inter_x1 - tile_x1, inter_y1 - tile_y1, inter_x2 - tile_x1, inter_y2 - tile_y1
