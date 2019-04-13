# import_valkyria: A Blender Add-on for Valkyria Chronicles models

`import_valkyria` is a Blender 2.74+ add-on for importing MLX, HMD, ABR,
and MXE models from Valkyria Chronicles. The model files used by the
PlayStation 3 and PC versions of Valkyria Chronicles are the same, so both
versions are supported.

To use `import_valkyria`, you have to install it (once) and activate it (each
time you start Blender, or in your startup file).

## Installing and activating

To install, open Blender's User Preferences panel, click the Add-ons tab,
click the Install from File button, select import_valkyria-X.X.zip, and click
Install.

To activate, open Blender's User Preferences panel, click the Addons tab,
find "Import-Export: Valkyria Chronicles (.MLX, .HMD, .ABR, .MXE)", and click
its checkbox.

## Importing Models

Once `import_valkyria` is installed and activated, you can import a model by
clicking File, Import, Valkyria Chronicles (.MLX, .HMD, .ABR, .MXE), or by
pressing space, typing valk, and choosing it from the list.

## Finding Models

If you have the Steam version of Valkyria Chronicles, you can open model files
directly from subfolders of its data folder. One example is `valcA02aD_h.mlx`,
which is in the `data\resource\mx` folder.

If you have the Playstation 3 version of Valkyria Chronicles, you have to
split up `DATA.CVM` to get usable models files. Use chrrox's `quickbms`
script, which you can download here:

http://forum.xentax.com/viewtopic.php?p=76717#p76717

## Known Issues

Bones are a little bit weird because this was the best way I could think of to
get bones to correspond to the right vertex groups. Also, armature
relationships that exist in some models (such as between the body armature and
the head armature) are not supported.

Poses are disabled for now because I never finished this feature, but I wanted
to release a new version anyway.

## Changelog

* Set default colorspace to sRGB to fix dark textures.
* Merged [#2](https://github.com/gomtuu/import_valkyria/pull/2) from [angavrilov](https://github.com/angavrilov).
  * Fix bad object positioning in environment maps and some characters.
  * Support opening DLC files in the PC version.
  * Use the custom normal layer introduced in 2.74 to save game normals.

### 2018-09-16: Version 0.7

* Added support for the Steam version's directory structure.
* Added preliminary support for armature poses/animations.
    (Temporarily disabled.)
* Uploaded to github and added README.

### 2013-05-25: Version 0.6

* Added preliminary support for ABR models.
* Added preliminary support for MXE models.
* Added support for double textures. Normal map support is now broken.

### 2013-05-11: Version 0.5

* Renamed from hmdl_import to import_valkyria.
* Heavily refactored for encapsulation and extensibility.
* Handles bones differently, allowing fingertip articulation.
* Supports shape keys for facial expressions, etc.

### 2013-04-26: Version 0.4

* Preview release. Poorly coded and poorly documented.
* First release that works in Blender 2.63+
* Has preliminary support for bones.
* Has better support for materials, including normal maps and transparency.
