"""
Inject GPS EXIF metadata into pothole images for testing the Damage Map feature.

Usage:
    python scripts/inject_gps_exif.py <image_path> <latitude> <longitude>
    python scripts/inject_gps_exif.py <image_path>   # uses default coords (Surabaya)

Examples:
    python scripts/inject_gps_exif.py data/test_pothole.jpg -7.2575 112.7521
    python scripts/inject_gps_exif.py data/test_pothole.jpg               
"""
import sys
from pathlib import Path

try:
    from PIL import Image
    import piexif
except ImportError:
    print("Missing dependency. Install with:")
    print("  pip install piexif Pillow")
    sys.exit(1)


def decimal_to_dms(decimal_deg: float):
    """Convert decimal degrees to EXIF-compatible (degrees, minutes, seconds) tuples."""
    d = int(abs(decimal_deg))
    m = int((abs(decimal_deg) - d) * 60)
    s = int(((abs(decimal_deg) - d) * 60 - m) * 60 * 10000)
    return ((d, 1), (m, 1), (s, 10000))


def inject_gps(image_path: str, lat: float, lon: float):
    """Inject GPS coordinates into the EXIF data of an image."""
    path = Path(image_path)
    if not path.exists():
        print(f"Error: File not found — {path}")
        return

    img = Image.open(path)

    # Build GPS EXIF IFD
    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef:  b"S" if lat < 0 else b"N",
        piexif.GPSIFD.GPSLatitude:     decimal_to_dms(lat),
        piexif.GPSIFD.GPSLongitudeRef: b"W" if lon < 0 else b"E",
        piexif.GPSIFD.GPSLongitude:    decimal_to_dms(lon),
    }

    # Try to preserve existing EXIF, or create new
    try:
        exif_dict = piexif.load(img.info.get("exif", b""))
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

    exif_dict["GPS"] = gps_ifd
    exif_bytes = piexif.dump(exif_dict)

    # Save (overwrite)
    img.save(str(path), exif=exif_bytes)
    print(f"✓ GPS injected into {path.name}")
    print(f"  Latitude:  {lat}")
    print(f"  Longitude: {lon}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    img_path = sys.argv[1]

    # Default: Surabaya, East Java
    latitude  = float(sys.argv[2]) if len(sys.argv) > 2 else -7.2575
    longitude = float(sys.argv[3]) if len(sys.argv) > 3 else 112.7521

    inject_gps(img_path, latitude, longitude)
