"""
IFC Parser - ממיר קבצי IFC לפורמט ConTech Metadata
"""

import numpy as np
from typing import List, Optional
from contech_metadata import Wall, ContechMetadata
import hashlib


def parse_ifc_to_metadata(
    ifc_path: str,
    pixels_per_meter: float = 100.0,
    target_width: int = 2000,
    target_height: int = 1500,
) -> Optional[ContechMetadata]:
    """
    פרסור קובץ IFC לפורמט metadata

    Args:
        ifc_path: נתיב לקובץ IFC
        pixels_per_meter: כיול (ברירת מחדל 100 פיקסלים = מטר)
        target_width: רוחב תמונה וירטואלי
        target_height: גובה תמונה וירטואלי

    Returns:
        אובייקט ContechMetadata או None אם נכשל
    """
    try:
        import ifcopenshell
        import ifcopenshell.geom
    except ImportError:
        return None

    # פתיחת קובץ IFC
    try:
        ifc_file = ifcopenshell.open(ifc_path)
    except Exception as e:
        print(f"Error opening IFC: {e}")
        return None

    # חישוב checksum
    with open(ifc_path, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    # יצירת metadata
    import os

    filename = os.path.basename(ifc_path)
    metadata = ContechMetadata(filename, checksum)

    metadata.pixels_per_meter = pixels_per_meter
    metadata.scale_text = "IFC Import"
    metadata.image_width = target_width
    metadata.image_height = target_height
    metadata.plan_type = "walls"
    metadata.is_scale_locked = True  # IFC הוא מדויק!

    # הגדרות עבור ifcopenshell
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    # חילוץ קירות
    walls_found = ifc_file.by_type("IfcWall")

    # מצא את הגבולות (bounding box)
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    wall_geometries = []

    for ifc_wall in walls_found:
        try:
            shape = ifcopenshell.geom.create_shape(settings, ifc_wall)
            verts = shape.geometry.verts

            # המרה ל-numpy array של נקודות (x, y, z)
            points_3d = np.array(verts).reshape(-1, 3)

            # שמירת גבולות
            min_x = min(min_x, points_3d[:, 0].min())
            min_y = min(min_y, points_3d[:, 1].min())
            max_x = max(max_x, points_3d[:, 0].max())
            max_y = max(max_y, points_3d[:, 1].max())

            wall_geometries.append((ifc_wall, points_3d))

        except Exception as e:
            print(f"Error processing wall {ifc_wall.GlobalId}: {e}")
            continue

    if not wall_geometries:
        return None

    # חישוב גורם קנה מידה
    real_width = max_x - min_x
    real_height = max_y - min_y

    scale_x = target_width / real_width if real_width > 0 else 1
    scale_y = target_height / real_height if real_height > 0 else 1
    scale = min(scale_x, scale_y) * 0.9  # 90% כדי להשאיר שוליים

    # המרת קירות לפיקסלים
    for idx, (ifc_wall, points_3d) in enumerate(wall_geometries):
        # המרה ל-2D (הקרנה על מישור xy)
        points_2d = points_3d[:, :2]  # רק x, y

        # נרמול למרכז התמונה
        points_2d[:, 0] = (points_2d[:, 0] - min_x) * scale + (
            target_width - real_width * scale
        ) / 2
        points_2d[:, 1] = (points_2d[:, 1] - min_y) * scale + (
            target_height - real_height * scale
        ) / 2

        # המרה ל-int
        points_pixels = [(int(p[0]), int(p[1])) for p in points_2d]

        # סינון נקודות כפולות
        unique_points = []
        for p in points_pixels:
            if not unique_points or p != unique_points[-1]:
                unique_points.append(p)

        # יצירת קיר
        wall = Wall(
            id=f"ifc_wall_{idx:04d}",
            points=unique_points,
            wall_type="concrete",
            thickness=0,  # ניתן לחלץ מה-IFC
        )

        metadata.add_wall(wall)

    metadata.notes = f"Imported from IFC: {len(walls_found)} walls found"

    return metadata


def ifc_supported() -> bool:
    """בדוק אם ספריית ifcopenshell זמינה"""
    try:
        import ifcopenshell

        return True
    except ImportError:
        return False
