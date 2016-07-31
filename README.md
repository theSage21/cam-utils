Cam-Utils
=========

Laptop camera utilities.

*NOTE*: Hogs CPU so you might want to customize which scripts to run on each frame

Usage
-----

- `run.py` is the main entry point. Use `python run.py` to run the scripts.
- all other scripts must implement a `run` function and be explicitly imported in `run.py`

Dependencies
------------

- global
    - OpenCV (tested with OpenCV 3.1.0 and Python 3.5)
- attendance
    - scikit-learn


Subscripts
----------

- `Face_Distance`: Distance of your face from the screen. Warns you if you get too close
- `screen brightness`: Adjust screen brightness based on luminance of camera image
- `attendance`: labels faces seen in the image

Todo
----

- Better subscript usage system. Imports are too hacky
- Some way to bring down CPU usage
