import MinkowskiEngine as ME

ME_VERSION = ME.__version__  # "0.5.4" or "0.4.3"

__all__ = ['IS_OLD_ME']


def is_old():
    mid_ver = int(ME_VERSION.split('.')[1])
    if mid_ver <= 4:
        return True
    else:
        return False


IS_OLD_ME = is_old()
