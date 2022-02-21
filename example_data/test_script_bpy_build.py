"""
Requires a blender python module build to run it!
See https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule

Opens the BLENDER_START_FILE and reads the SUPPORTS_XML_TEST_FILE using the
addon (ensure it can be found by specifying its location in the PYTHONPATH
environment variable) and writes the results to BLENDER_OUTPUT_FILE
"""

import pathlib

import bpy

import io_import_supports

BLENDER_START_FILE = 'empty.blend'
SUPPORTS_XML_TEST_FILE = 'simple structure_supports.xml'
BLENDER_OUTPUT_FILE = 'test_script_bpy_build_output.blend'


def main(
        supports_file_path: str, blender_start_file_path: str,
        blender_output_file_path: str
):
    supports_file_path = pathlib.Path(supports_file_path).absolute()

    start_file = str(pathlib.Path(blender_start_file_path).absolute())
    bpy.ops.wm.open_mainfile(filepath=start_file)
    output_file = str(
        pathlib.Path(blender_output_file_path).absolute()
    )

    io_import_supports.register()
    io_import_supports.import_supports(supports_file_path)

    bpy.ops.wm.save_mainfile(filepath=output_file)


if __name__ == '__main__':
    main(SUPPORTS_XML_TEST_FILE, BLENDER_START_FILE, BLENDER_OUTPUT_FILE)
