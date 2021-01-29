#!BPY
# -*- coding: utf-8 -*-

bl_info = {
        "name": "Valkyria Chronicles (.MLX, .HMD, .ABR, .MXE)",
        "description": "Imports model files from Valkyria Chronicles (PS3)",
        "author": "Chrrox, Gomtuu",
        "version": (0, 8),
        "blender": (2, 83, 0),
        "location": "File > Import",
        "warning": "",
        "category": "Import-Export",
        }

if 'bpy' in locals():
    import importlib
    importlib.reload(valkyria.files)
    importlib.reload(materials)

import os.path
from math import radians, isfinite
from collections import defaultdict
import bpy, mathutils
from mathutils import Vector, Matrix, Quaternion, Euler
from bpy_extras.io_utils import ImportHelper
from . import valkyria, materials

DEBUG = False

ROTATE_SCENE_MATRIX = Matrix.Rotation(radians(90), 4, 'X')

def make_transform_matrix(loc,rot,scale):
    mat_loc = Matrix.Translation(loc)
    mat_rot = Quaternion(rot).to_matrix().to_4x4()
    mat_scale = Matrix.Diagonal([*scale, 1])
    return mat_loc @ mat_rot @ mat_scale

def make_transform_matrix_euler(loc,rot,scale,order='XYZ'):
    mat_loc = Matrix.Translation(loc)
    mat_rot = Euler(rot, order).to_matrix().to_4x4()
    mat_scale = Matrix.Diagonal([*scale, 1])
    return mat_loc @ mat_rot @ mat_scale

def matrix_difference(mat1, mat2):
    return max(max(abs(a - b) for a, b in zip(ra, rb)) for ra, rb in zip(mat1, mat2))


class Texture_Pack:
    def __init__(self):
        self.htsf_images = []
        self.blender_built = False

    def add_image(self, htsf, filename):
        image = HTSF_Image(htsf)
        image.filename = filename + ".dds"
        self.htsf_images.append(image)
        return image

    def build_blender(self, vscene):
        for image in self.htsf_images:
            image.build_blender(vscene)
        self.blender_built = True


class HTEX_Pack:
    def __init__(self, source_file, htex_id):
        self.F = source_file
        self.htex_id = htex_id
        self.htsf_images = []
        self.blender_built = False

    def add_image(self, htsf):
        htsf_id = len(self.htsf_images)
        image = HTSF_Image(htsf)
        image.filename = "HTEX-{:03}-HTSF-{:03}.dds".format(self.htex_id, htsf_id)
        self.htsf_images.append(image)
        return image

    def read_data(self, vscene):
        for htsf in self.F.HTSF:
            image = self.add_image(htsf)
            image.read_data(vscene)

    def build_blender(self, vscene):
        for image in self.htsf_images:
            image.build_blender(vscene)
        self.blender_built = True

    def build_raw_texture_planes(self):
        obj = None

        for i, image in enumerate(self.htsf_images):
            material = bpy.data.materials.new(name=image.filename)
            material.use_nodes = True
            material.blend_method = 'CLIP'
            material.shadow_method = 'CLIP'
            node_tree = material.node_tree

            for n in list(node_tree.nodes):
                node_tree.nodes.remove(n)

            nodes = {}
            nodes['Principled BSDF'] = node = node_tree.nodes.new('ShaderNodeBsdfPrincipled')
            node.name = 'Principled BSDF'
            node.location = Vector((170, 300))
            node.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
            node.inputs['Metallic'].default_value = 0.0
            node.inputs['Specular'].default_value = 0.0
            nodes['Image Texture'] = node = node_tree.nodes.new('ShaderNodeTexImage')
            node.name = 'Image Texture'
            node.location = Vector((-140, -76))
            node.image = image.image
            nodes['Material Output'] = node = node_tree.nodes.new('ShaderNodeOutputMaterial')
            node.name = 'Material Output'
            node.location = Vector((500, 300))
            node_tree.links.new(nodes['Image Texture'].outputs['Alpha'],nodes['Principled BSDF'].inputs['Alpha'])
            node_tree.links.new(nodes['Principled BSDF'].outputs['BSDF'],nodes['Material Output'].inputs['Surface'])
            node_tree.links.new(nodes['Image Texture'].outputs['Color'],nodes['Principled BSDF'].inputs['Emission'])

            if obj is None:
                bpy.ops.mesh.primitive_plane_add()
                bpy.ops.mesh.uv_texture_add()
                bpy.ops.object.material_slot_add()

                obj = bpy.context.active_object
                obj.name = image.filename
            else:
                obj = bpy.data.objects.new(object_data=obj.data, name=image.filename)
                bpy.context.collection.objects.link(obj)

            obj.location = i * Vector((2,0,0))
            mslot = obj.material_slots[0]
            mslot.link = 'OBJECT'
            mslot.material = material


class HTSF_Image:
    def __init__(self, source_file):
        self.F = source_file
        assert len(self.F.DDS) == 1
        self.dds = self.F.DDS[0]
        self.image_ref = None

    @property
    def image(self):
        if not self.image_ref:
            # Lazy import on demand
            self.vscene.image_manager.load_image(self)
        return self.image_ref

    def build_blender(self, vscene):
        self.vscene = vscene

    def read_data(self, vscene):
        self.dds.read_data()


class MXTL_List:
    def __init__(self, source_file):
        self.F = source_file
        self.texture_packs = []

    def read_data(self, vscene):
        self.F.read_data()
        self.texture_lists = self.F.texture_lists


class IZCA_Model:
    def __init__(self, source_file):
        self.F = source_file
        self.texture_packs = []
        self.shape_key_sets = []
        self.hmdl_models = []

    def add_hshp(self, hshp):
        hshp_id = len(self.shape_key_sets)
        shape_key_set = HSHP_Key_Set(hshp, hshp_id)
        self.shape_key_sets.append(shape_key_set)
        return shape_key_set

    def add_htex(self, htex):
        htex_id = len(self.texture_packs)
        htex_pack = HTEX_Pack(htex, htex_id)
        self.texture_packs.append(htex_pack)
        return htex_pack

    def add_model(self, hmdl):
        model_id = len(self.hmdl_models)
        model = HMDL_Model(hmdl, model_id)
        self.hmdl_models.append(model)
        return model

    def read_data(self, vscene):
        if hasattr(self.F, 'HSHP'):
            for hshp in self.F.HSHP:
                shape_key_set = self.add_hshp(hshp)
                shape_key_set.read_data(vscene)
        if getattr(self.F, 'MXTL', False):
            # read HMDL/HTSF associations from MXTL
            mxtl = MXTL_List(self.F.MXTL[0])
            mxtl.read_data(vscene)
            for model_i, texture_list in enumerate(mxtl.texture_lists):
                texture_pack = Texture_Pack()
                for htsf_i, filename in texture_list:
                    htsf = texture_pack.add_image(self.F.HTSF[htsf_i], filename)
                    htsf.read_data(vscene)
                self.texture_packs.append(texture_pack)
                model = self.add_model(self.F.HMDL[model_i])
                model.read_data(vscene)
        else:
            # deduce HMDL/HTEX associations
            for hmd, htx in zip(self.F.HMDL, self.F.HTEX):
                htex_pack = self.add_htex(htx)
                htex_pack.read_data(vscene)
                model = self.add_model(hmd)
                model.read_data(vscene)

    def build_blender(self, vscene):
        for texture_pack, model in zip(self.texture_packs, self.hmdl_models):
            texture_pack.build_blender(vscene)
            model.build_blender(vscene)
            model.assign_materials(texture_pack.htsf_images)
        for shape_key_set in self.shape_key_sets:
            # TODO: Is there a smarter way to determine which models use shape keys?
            self.hmdl_models[1].build_shape_keys(shape_key_set)

    def finalize_blender(self):
        for model in self.hmdl_models:
            model.finalize_blender()


class IZCA_Poses:
    def __init__(self, source_file):
        self.F = source_file
        self.poses = []

    def read_data(self, vscene):
        for hmot in self.F.HMOT:
            hmot.read_data()
            self.poses.append(hmot.bones)

    def pose_from_armature(self, arm, posed_arm):
        # Needs to run multiple times within script window?
        # Even then, bones are rolled incorrectly.
        for bone in arm.pose.bones:
            posed_bone = posed_arm.data.bones[bone.name]
            #posed_vector = posed_bone.tail - posed_bone.head
            #vector = bone.tail - bone.head
            #bone.rotation_quaternion = vector.rotation_difference(posed_vector)
            #bone.rotation_mode = 'QUATERNION'
            bone.matrix = posed_bone.matrix_local

    def pose_model(self, izca_model):
        hmdl = izca_model.hmdl_models[0]
        kfmd = hmdl.kfmd_models[0]
        #i = 0
        for bone in kfmd.bones:
            bone["orig_location"] = bone["location"]
            bone["orig_rotation"] = bone["rotation"]
            bone["orig_scale"] = bone.get("scale", (1, 1, 1))
        for pose_bones in self.poses:
            #j = 0
            for bone, pose_bone in zip(kfmd.bones, pose_bones):
                #print("{:02x} Orig:".format(i), bone["location"], bone["rotation"])
                if "location" not in pose_bone:
                    pose_bone["location"] = None
                if "rotation" not in pose_bone:
                    pose_bone["rotation"] = None
                if "scale" not in pose_bone:
                    pose_bone["scale"] = None
                if pose_bone["location"]:
                    bone["location"] = tuple(map(sum, zip(pose_bone["location"], pose_bone["location_frames"][0])))
                if pose_bone["rotation"]:
                    bone["rotation"] = tuple(map(sum, zip(pose_bone["rotation"], pose_bone["rotation_frames"][0])))
                if pose_bone["scale"]:
                    bone["scale"] = tuple(map(sum, zip(pose_bone["scale"], pose_bone["scale_frames"][0])))
                #print("Pose {:02x} Bone {:02x}:".format(i, j), bone["location"], bone["rotation"], bone["scale"])
                #j += 1
            armature = hmdl.kfmd_models[0].build_armature()
            #i += 1
            #armature.location = ((i % 16) * 10, int(i / 16) * -20, 0)
            #self.pose_from_armature(kfmd.armature, armature)
            for bone in kfmd.bones:
                bone["location"] = bone["orig_location"]
                bone["rotation"] = bone["orig_rotation"]
                bone["scale"] = bone.get("orig_scale", (1, 1, 1))
            break


class ABRS_Model:
    def __init__(self, source_file):
        self.F = source_file
        self.texture_packs = []
        self.hmdl_models = []

    def add_model(self, hmdl):
        model_id = len(self.hmdl_models)
        model = HMDL_Model(hmdl, model_id)
        self.hmdl_models.append(model)
        return model

    def read_data(self, vscene):
        texture_pack = None
        htex_count = 0
        self.first_texture_pack = None
        for inner_file in self.F.inner_files:
            if inner_file.ftype == 'HMDL':
                model = self.add_model(inner_file)
                model.read_data(vscene)
                texture_pack = Texture_Pack()
                self.texture_packs.append(texture_pack)
            elif inner_file.ftype == 'HTEX':
                if not texture_pack:
                    texture_pack = self.first_texture_pack = Texture_Pack()
                htex_pack = HTEX_Pack(inner_file, htex_count)
                htex_pack.read_data(vscene)
                for htsf in htex_pack.htsf_images:
                    texture_pack.htsf_images.append(htsf)
                htex_count += 1
        assert len(self.texture_packs) == len(self.hmdl_models)

    def build_blender(self, vscene):
        if self.first_texture_pack:
            self.first_texture_pack.build_blender(vscene)
        for texture_pack, model in zip(self.texture_packs, self.hmdl_models):
            texture_pack.build_blender(vscene)
            model.build_blender(vscene)
            model.assign_materials(texture_pack.htsf_images)

    def finalize_blender(self):
        for model in self.hmdl_models:
            model.finalize_blender()


class MXEN_Model:
    # TODO: EV_OBJ_026.MXE causes vertex group error
    def __init__(self, source_file):
        self.F = source_file
        self.texture_packs = []
        self.hmdl_models = []
        self.instances = []

    def add_htex(self, htex):
        htex_id = len(self.texture_packs)
        htex_pack = HTEX_Pack(htex, htex_id)
        self.texture_packs.append(htex_pack)
        return htex_pack

    def add_model(self, hmdl, filename):
        model_id = len(self.hmdl_models)
        model = HMDL_Model(hmdl, model_id, model_name=filename, collection_path=['shared_models'])
        self.hmdl_models.append(model)
        return model

    def open_file(self, filename):
        path = os.path.dirname(self.F.filename)
        possible_files = []
        possible_files.append(os.path.join(path, filename))
        possible_files.append(os.path.join(path, filename.lower()))
        possible_files.append(os.path.join(path, filename.upper()))
        possible_files.append(os.path.join(path, '..', 'resource', 'mx', filename.lower()))
        possible_files.append(os.path.join(path, '..', 'resource', 'mx', filename.upper()))
        opened_file = None
        for model_filepath in possible_files:
            try:
                opened_file = valkyria.files.valk_open(model_filepath)[0]
            except FileNotFoundError:
                pass
        if opened_file is None:
            raise FileNotFoundError(filename)
        return opened_file

    def read_data(self, vscene):
        mxec = self.F.MXEC[0]
        mxec.read_data()
        if hasattr(mxec, "mmf_file"):
            mmf = self.open_file(mxec.mmf_file["filename"])
            mmf.find_inner_files()
            mmf.read_data()
        if hasattr(mxec, "htr_file"):
            htr = self.open_file(mxec.htr_file["filename"])
            htr.read_data()
        if hasattr(mxec, "merge_htx_file"):
            merge_htx = self.open_file(mxec.merge_htx_file["filename"])
            merge_htx.find_inner_files()
        model_cache = {}
        texture_cache = {}
        self.instance_ids = []
        self.instance_models = []
        self.instance_textures = []
        for i, mxec_model in enumerate(mxec.models):
            if not "model_file" in mxec_model:
                continue
            self.instance_ids.append(i)
            model_file_desc = mxec_model["model_file"]
            model_filename = model_file_desc["filename"]
            model = model_cache.get(model_filename, None)
            if model is None:
                print("Reading", model_filename)
                if model_file_desc["is_inside"] == 0:
                    hmd = self.open_file(model_filename)
                    hmd.find_inner_files()
                    model = self.add_model(hmd, model_filename)
                    model.read_data(vscene)
                elif model_file_desc["is_inside"] == 0x200:
                    hmd = mmf.named_models[model_filename]
                    model = self.add_model(hmd, model_filename)
                    model.read_data(vscene)
                model_cache[model_filename] = model
            else:
                model.as_collection = True
            self.instance_models.append(model)
            self.instances.append((
                (mxec_model["location_x"], mxec_model["location_y"], mxec_model["location_z"]),
                (radians(mxec_model["rotation_x"]), radians(mxec_model["rotation_y"]), radians(mxec_model["rotation_z"])),
                (mxec_model["scale_x"], mxec_model["scale_y"], mxec_model["scale_z"])
                ))
            texture_file_desc = mxec_model["texture_file"]
            texture_pack_filename = texture_file_desc["filename"]
            texture_pack = texture_cache.get(texture_pack_filename)
            if texture_pack is None:
                if texture_file_desc["is_inside"] == 0:
                    htx = self.open_file(texture_pack_filename)
                    htx.find_inner_files()
                    texture_pack = self.add_htex(htx)
                    texture_pack.read_data(vscene)
                elif texture_file_desc["is_inside"] == 0x100:
                    texture_pack = Texture_Pack()
                    for htsf_i in htr.texture_packs[texture_file_desc["htr_index"]]["htsf_ids"]:
                        texture_filename = "{}-{:03d}".format(texture_pack_filename, htsf_i)
                        htsf = texture_pack.add_image(merge_htx.HTSF[htsf_i], texture_filename)
                        htsf.read_data(vscene)
                    self.texture_packs.append(texture_pack)
                texture_pack.mxec_filename = texture_pack_filename
                texture_cache[texture_pack_filename] = texture_pack
            self.instance_textures.append(texture_pack)

    def build_blender(self, vscene):
        # Build models
        for model in self.hmdl_models:
            model.build_blender(vscene)
            model.mxec_texture_pack = None
        # Build texture packs
        for texture_pack in self.texture_packs:
            if not texture_pack.blender_built:
                texture_pack.build_blender(vscene)
        # Combine and instantiate
        for i, texture_pack, model, (loc,rot,scale) in zip(self.instance_ids, self.instance_textures, self.instance_models, self.instances):
            if model.mxec_texture_pack is None:
                model.mxec_texture_pack = texture_pack
                model.assign_materials(texture_pack.htsf_images)
            elif model.mxec_texture_pack != texture_pack:
                print(
                    "Texture pack mismatch for ", model.model_name, ": ",
                    model.mxec_texture_pack.mxec_filename, " vs ", texture_pack.mxec_filename
                )
            matrix = make_transform_matrix_euler(loc, rot, scale, order='XYZ')
            if model.as_collection:
                instance = bpy.data.objects.new(model.model_name, None)
                instance.instance_type = 'COLLECTION'
                instance.instance_collection = model.collection
                vscene.collection.objects.link(instance)
                if vscene.rotate_scene:
                    matrix @= ROTATE_SCENE_MATRIX.inverted()
            else:
                instance = model.empty
            instance.matrix_world = matrix
            if DEBUG:
                instance['id'] = i
                instance['model_path'] = '{}:{:x}'.format(':'.join(model.F.container_path()), model.F.container_offset())

    def finalize_blender(self):
        for model in self.hmdl_models:
            model.finalize_blender()
            if model.as_collection:
                model.layer_collection.exclude = True


class HSHP_Key_Set:
    def __init__(self, source_file, shape_key_set_id):
        self.F = source_file
        self.shape_key_set_id = shape_key_set_id

    def read_data(self, vscene):
        self.F.read_data()
        self.shape_keys = self.F.shape_keys


class HMDL_Model:
    def __init__(self, source_file, model_id, *, model_name=None, collection_path=[], as_collection=False):
        self.F = source_file
        self.model_id = model_id
        self.model_name = model_name
        self.kfmd_models = []
        self.as_collection = as_collection
        self.collection_path = collection_path

    def add_model(self, kfmd):
        model_id = len(self.kfmd_models)
        model = KFMD_Model(kfmd, model_id)
        self.kfmd_models.append(model)
        return model

    def read_data(self, vscene):
        for kfmd in self.F.KFMD:
            model = self.add_model(kfmd)
            model.read_data(vscene)

            # Heuristic: assume the remaining models are LOD meshes
            if vscene.lod_filter:
                break

    def build_blender(self, vscene):
        name = self.model_name or "HMDL-{:03d}".format(self.model_id)
        if self.as_collection:
            self.collection, self.layer_collection = vscene.get_sub_collection(*self.collection_path, name)
            for model in self.kfmd_models:
                model.build_blender(vscene, self.collection)
                if vscene.rotate_scene:
                    model.empty.matrix_world = ROTATE_SCENE_MATRIX
        else:
            self.empty = bpy.data.objects.new(name, None)
            vscene.collection.objects.link(self.empty)
            for model in self.kfmd_models:
                model.build_blender(vscene)
                model.empty.parent = self.empty

    def assign_materials(self, texture_pack):
        for model in self.kfmd_models:
            model.assign_uv_maps()
            model.assign_vertex_colors()
            model.build_materials(texture_pack)
            model.assign_materials()

    def build_shape_keys(self, shape_key_set):
        # TODO: Is there a smarter way to determine which models use shape keys?
        self.kfmd_models[0].build_shape_keys(shape_key_set)

    def finalize_blender(self):
        for model in self.kfmd_models:
            model.finalize_blender()


class KFMD_Model:
    def __init__(self, source_file, model_id):
        self.F = source_file
        self.model_id = model_id
        self.kfms = self.F.KFMS[0]
        self.kfmg = self.F.KFMG[0]
        self.empty = None
        self.oneside = None

    def build_armature(self, vscene):
        armature = bpy.data.objects.new("Armature",
            bpy.data.armatures.new("ArmatureData"))
        self.collection.objects.link(armature)
        vscene.view_layer.objects.active = armature
        armature.select_set(True)
        armature.data.display_type = 'STICK'
        bpy.ops.object.mode_set(mode = 'EDIT')
        common_bias_matrix = None
        for i, bone in enumerate(self.bones):
            if 'deform_id' in bone:
                bone['name'] = "Bone-{:02x}".format(bone['deform_id'])
            else:
                bone['name'] = "Bone-{:02x}".format(bone['id'])
            # Root bone location breaks:
            #   VC1: valcA08cX_h (body)
            #   VC4: omap_ground_00_01_a (grass,flowers)
            # Root bone rotation required:
            #   VC1: valcA08bX_h (body)
            bone_loc = bone["location"] if i != 0 else (0,0,0)
            bone["matrix"] = make_transform_matrix(bone_loc, bone["rotation"], bone["scale"])
            if bone["parent"]:
                bone["accum_matrix"] = bone["parent"]["accum_matrix"]
                bone["head"] = bone["accum_matrix"] @ Vector(bone_loc)
                bone["accum_matrix"] = bone["accum_matrix"] @ bone["matrix"]
            else:
                bone["accum_matrix"] = bone["matrix"]
                bone["head"] = Vector(bone_loc)
            # Check that deform bone rest matrices are consistent
            if i == 0:
                common_bias_matrix = bone["accum_matrix"]
            if 'matrix_raw' in bone:
                bias_matrix = bone["accum_matrix"] @ Matrix(bone["matrix_raw"]).transposed()
                if common_bias_matrix is None:
                    common_bias_matrix = bias_matrix
                elif matrix_difference(common_bias_matrix, bias_matrix) > 1e-4:
                    print('Varying armature matrix bias: bone ', bone['id'], ' deform ', bone['deform_id'], '\n',
                          'common: ', common_bias_matrix, '\nbone: ', bias_matrix)
        for bone in self.bones:
            # Default bone orientation and size
            if bone["parent"]:
                length = min(1, (bone["parent"]["tail"] - bone["parent"]["head"]).length)
            else:
                length = 5
            bone["tail"] = bone["head"] + bone["accum_matrix"].col[0].normalized().resized(3) * length
            # Point the bone at the average position of its children if they are in a common direction
            child_vecs = [ child["head"] - bone["head"] for child in bone["children"] if (child["head"] - bone["head"]).length > 1e-4 ]
            if len(child_vecs) > 0:
                avg_vec = sum(child_vecs, Vector((0,0,0))) / len(child_vecs)
                avg_dir = avg_vec.normalized()
                if all(vec.normalized().dot(avg_dir) > 0.8 for vec in child_vecs):
                    bone["tail"] = bone["head"] + avg_vec
            # Create and place the bone
            bone["edit_bpy"] = armature.data.edit_bones.new(bone["name"])
            bone["edit_bpy"].use_connect = False
            if bone["parent"]:
                bone["edit_bpy"].parent = bone["parent"]["edit_bpy"]
            bone["edit_bpy"].head = bone["head"]
            bone["edit_bpy"].tail = bone["tail"]
            if DEBUG:
                bone["edit_bpy"]["id"] = bone["id"]
            #print(bone["edit_bpy"], {k:v for k,v in bone.items() if k not in {'parent', 'children', 'matrix', 'accum_matrix', 'edit_bpy'}})
        bpy.ops.object.mode_set(mode = 'OBJECT')
        return armature

    def check_single_vgroup(self, mesh):
        vgroups = mesh["vertex_groups"]

        # There is only one bone
        bones = list(vgroups.keys())
        if len(bones) != 1:
            return None

        # The bone affects all vertices
        vdefs = vgroups[bones[0]]
        if len(vdefs) != len(mesh["vertices"]):
            return None

        return "Bone-{:02x}".format(bones[0])

    def assign_parent_bone(self, mesh_dict, parent_bone, rest_matrix):
        bone = self.armature.data.bones[parent_bone]
        bone_matrix = bone.matrix_local @ Matrix.Translation((0,bone.length,0))
        mesh_dict["bpy"].parent_type = 'BONE'
        mesh_dict["bpy"].parent_bone = parent_bone
        mesh_dict["bpy"].matrix_parent_inverse = bone_matrix.inverted() @ rest_matrix

    def build_meshes(self, vscene):
        for i, mesh_dict in enumerate(self.meshes):
            # Create mesh object
            mesh = bpy.data.meshes.new("MeshData-{:03d}".format(i))
            mesh_dict["bpy"] = bpy.data.objects.new("Mesh-{:03d}".format(i), mesh)
            self.collection.objects.link(mesh_dict['bpy'])
            mesh_dict["bpy"].parent = self.armature
            # Create mesh data
            vertices = [vertex["location"] for vertex in mesh_dict["vertices"]]
            mesh.from_pydata(vertices, [], mesh_dict["faces"])
            for p in mesh.polygons:
                p.use_smooth = True
            # Move accessories to proper places
            parent_bone_id = mesh_dict["object"]["parent_bone_id"]
            parent_bone = self.bones[parent_bone_id]
            # Parent meshes with vertex groups to the armature, and others to bones or the object
            if mesh_dict["object"]["parent_is_armature"]:
                # Armature deformation is expensive, try to downgrade to bone parenting if only one bone
                vgroup_parent = self.check_single_vgroup(mesh_dict)
                if vgroup_parent and vgroup_parent in self.armature.data.bones:
                    self.assign_parent_bone(mesh_dict, vgroup_parent, parent_bone["accum_matrix"])
                else:
                    mesh_dict["bpy"].parent_type = 'ARMATURE'
                    mesh_dict["bpy"].matrix_parent_inverse = parent_bone["accum_matrix"]
            elif parent_bone["name"] in self.armature.data.bones:
                self.assign_parent_bone(mesh_dict, parent_bone["name"], parent_bone["accum_matrix"])
            else:
                mesh_dict["bpy"].parent_type = 'OBJECT'
                mesh_dict["bpy"].matrix_parent_inverse = parent_bone["accum_matrix"]

    def assign_vertex_groups(self):
        for mesh in self.meshes:
            for global_id, vertex_list in mesh["vertex_groups"].items():
                vgroup_name = "Bone-{:02x}".format(global_id)
                if vgroup_name in mesh["bpy"].vertex_groups:
                    vgroup = mesh["bpy"].vertex_groups[vgroup_name]
                else:
                    vgroup = mesh["bpy"].vertex_groups.new(name=vgroup_name)
                for vertex_id, weight in vertex_list:
                    vgroup.add([vertex_id], weight, 'ADD')

    def build_blender(self, vscene, collection=None):
        self.collection = collection or vscene.collection
        self.vscene = vscene
        self.empty = bpy.data.objects.new("KFMD-{:03d}".format(self.model_id), None)
        self.collection.objects.link(self.empty)
        self.armature = self.build_armature(vscene)
        self.armature.parent = self.empty
        self.build_meshes(vscene)
        self.assign_vertex_groups()

    def index_vertex_groups(self):
        # TODO: This function and assign_vertex_groups might be a little
        # excessive. Consider doing this all directly when building the mesh.
        group_names = [("vertex_group_1", "vertex_group_weight_1"),
                       ("vertex_group_2", "vertex_group_weight_2"),
                       ("vertex_group_3", "vertex_group_weight_3"),
                       ("vertex_group_4", "vertex_group_weight_4")]
        for mesh in self.meshes:
            id_map = mesh["vertex_group_map"]
            vertex_groups = defaultdict(list)

            for i, vertex in enumerate(mesh["vertices"]):
                total = 0.0
                weights = defaultdict(float) # duplicate detection

                for id_name, weight_name in group_names:
                    if id_name not in vertex:
                        break

                    group = vertex[id_name]

                    # Allow implicit last weight
                    if weight_name in vertex:
                        weight = vertex[weight_name]
                    else:
                        weight = 1.0 - total

                    if isfinite(weight) and weight > 0:
                        total += weight
                        weights[group] += weight

                for group, weight in weights.items():
                    vertex_groups[group].append([i, weight])

            mesh["vertex_groups"] = { id_map[k]: v for k,v in vertex_groups.items() }

    def read_data(self, vscene):
        self.F.read_data()
        self.bones = self.F.bones
        self.materials = self.F.materials
        self.meshes = self.F.meshes
        self.textures = self.F.textures
        self.index_vertex_groups()

    def create_oneside(self):
        self.oneside = bpy.data.textures.new("OneSide", type='BLEND')
        self.oneside.use_color_ramp = True
        self.oneside.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        self.oneside.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        element0 = self.oneside.color_ramp.elements.new(0.5)
        element0.color = (0.0, 0.0, 0.0, 1.0)
        element1 = self.oneside.color_ramp.elements.new(0.501)
        element1.color = (1.0, 1.0, 1.0, 1.0)

    def build_materials(self, texture_pack):
        builder = self.vscene.material_builder

        for ptr, material_dict in self.materials.items():
            name = "Material-{:04x}".format(ptr)

            matbpy = material_dict['bpy'] = {}
            for vcolors in material_dict['needs_vertex_colors']:
                matbpy[vcolors] = builder.build_material(name, material_dict, texture_pack, vcolors)

    def assign_materials(self):
        for mesh in self.meshes:
            vertices = mesh["vertices"]
            if len(vertices) == 0:
                continue
            material_dict = self.materials[mesh["object"]["material_ptr"]]
            material = material_dict['bpy'][mesh['has_vertex_colors']]
            mesh["bpy"].data.materials.append(material)

            disp_node = material.node_tree.nodes.get("Displacement")
            if disp_node and isinstance(disp_node, bpy.types.ShaderNodeGroup):
                group_name = disp_node.node_tree.get('valkyria_special','')
                if group_name.startswith('DisplaceWorldY'):
                    self.add_displacement(mesh['bpy'], material, disp_node, group_name)

    def add_displacement(self, obj, material, disp_node, group_name):
        tex_node = disp_node.inputs['Image'].links[0].from_node
        assert isinstance(tex_node, bpy.types.ShaderNodeTexImage)
        uv_node = tex_node.inputs[0].links[0].from_node
        assert isinstance(uv_node, bpy.types.ShaderNodeUVMap)

        tex = bpy.data.textures.new(tex_node.image.name, 'IMAGE')
        tex.image = tex_node.image
        tex.factor_red = 0
        tex.factor_blue = 0
        tex.saturation = 0

        mod = obj.modifiers.new('Displace', 'DISPLACE')
        mod.texture = tex
        mod.texture_coords = 'UV'
        mod.uv_layer = uv_node.uv_map
        mod.space = 'GLOBAL'
        mod.direction = 'Z' if self.vscene.rotate_scene else 'Y'
        mod.mid_level = 0.5
        mod.strength = disp_node.inputs['Scale'].default_value

    def assign_uv_maps(self):
        uv_names = ["uv", "uv2", "uv3", "uv4", "uv5"]
        for mesh in self.meshes:
            vertices = mesh["vertices"]
            if len(vertices) == 0:
                continue
            for slot_i, field in enumerate(uv_names):
                if field not in vertices[0]:
                    break
                for vert in vertices:
                    if vert[field] != (0,0):
                        break
                else:
                    continue
                uv_name = "UVMap-{}".format(slot_i)
                layer = mesh["bpy"].data.uv_layers.new(name=uv_name)
                for data, loop in zip(layer.data, mesh["bpy"].data.loops):
                    u, v = vertices[loop.vertex_index][field]
                    data.uv = (u, 1 - v)

    def assign_vertex_colors(self):
        color_names = ['color', 'color2']
        for mesh in self.meshes:
            vcolors = set()
            vertices = mesh["vertices"]
            if len(vertices) == 0:
                continue
            for color_i, field in enumerate(color_names):
                if field not in vertices[0]:
                    break
                for vert in vertices:
                    if vert[field] != (1,1,1,1):
                        break
                else:
                    continue
                vcolors.add(color_i)
                color_name = "Color-{}".format(color_i)
                layer = mesh["bpy"].data.vertex_colors.new(name=color_name)
                for data, loop in zip(layer.data, mesh["bpy"].data.loops):
                    data.color = vertices[loop.vertex_index][field]
            mesh['has_vertex_colors'] = vcolors = frozenset(vcolors)
            material = self.materials[mesh["object"]["material_ptr"]]
            material.setdefault('needs_vertex_colors', set()).add(vcolors)

    def build_shape_keys(self, shape_key_set):
        for mesh, shape_key in zip(self.meshes, shape_key_set.shape_keys):
            if shape_key['vc_game'] == 1:
                shape_vertices = shape_key["vertices"]
                vertex_shift = len(mesh["bpy"].data.vertices) - len(shape_vertices)
            elif shape_key['vc_game'] == 4:
                slice_start = mesh["first_vertex"]
                slice_end = slice_start + mesh["vertex_count"]
                shape_vertices = shape_key["vertices"][slice_start:slice_end]
                vertex_shift = 0
            if not mesh["bpy"].data.shape_keys:
                mesh["bpy"].shape_key_add(name='Basis')
            sk_name = "HSHP-{:02d}".format(shape_key_set.shape_key_set_id)
            sk = mesh["bpy"].shape_key_add(name=sk_name)
            for i, vertex in enumerate(shape_vertices):
                if "translate" not in vertex:
                    continue
                sk.data[i + vertex_shift].co += Vector(vertex["translate"])

    def finalize_blender(self):
        for mesh in self.meshes:
            mesh["bpy"].data.update()
            mesh["bpy"].data.use_auto_smooth = True
            normals = [dict_vertex["normal"] for dict_vertex in mesh["vertices"]]
            mesh["bpy"].data.normals_split_custom_set_from_vertices(normals)


class DummyScene:
    def __init__(self):
        self.image_manager = materials.ImageManager()


class ValkyriaScene:
    def __init__(self, context, source_file, name, *, rotate_scene, lod_filter):
        self.context = context
        self.source_file = source_file
        self.name = os.path.basename(name)
        self.filename = name
        self.rotate_scene = rotate_scene
        self.lod_filter = lod_filter
        self.image_manager = materials.ImageManager()
        self.material_builder = materials.MaterialBuilder(self, rotate_scene=rotate_scene)
        self.extra_objects = []

    def create_scene(self, name):
        self.scene = bpy.data.scenes.new(name)
        self.context.window.scene = self.scene
        self.init_scene()

    def reuse_scene(self):
        self.scene = self.context.scene
        self.init_scene()

    def init_scene(self):
        self.view_layer = self.scene.view_layers[0]
        self.root_layer_collection = self.view_layer.layer_collection
        self.root_collection = self.root_layer_collection.collection
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.clip_start = 1
                            space.clip_end = 22000
                            space.shading.show_backface_culling = True
        self.scene.display_settings.display_device = 'sRGB'
        self.scene.view_settings.view_transform = 'Standard'

    def create_collection(self, name):
        self.collection = bpy.data.collections.new(name)
        self.root_collection.children.link(self.collection)
        self.layer_collection = self.root_layer_collection.children[self.collection.name]
        self.layer_collection_map = {}

    def get_sub_collection(self, *names):
        layer_collection = self.layer_collection
        for i in range(len(names)):
            key = tuple(names[0:i+1])
            child = self.layer_collection_map.get(key)
            if not child:
                collection = bpy.data.collections.new(names[i])
                layer_collection.collection.children.link(collection)
                child = layer_collection.children[collection.name]
                self.layer_collection_map[key] = child
            layer_collection = child
        return layer_collection.collection, layer_collection

    def create_lamp(self):
        lamp_data = bpy.data.lights.new("Default Lamp", 'SUN')
        lamp = bpy.data.objects.new("Default Lamp", lamp_data)
        lamp.location = (0.0, 20.0, 15.0)
        lamp.rotation_mode = 'AXIS_ANGLE'
        lamp.rotation_axis_angle = (radians(-22.0), 1.0, 0.0, 0.0)
        self.root_collection.objects.link(lamp)
        self.extra_objects.append(lamp)

    def read_data(self):
        self.source_file.read_data(self)
        if isinstance(self.source_file, HMDL_Model):
            possible_files = []
            possible_files.append(self.filename[0:-4] + '.htx')
            possible_files.append(self.filename[0:-4] + '.HTX')
            htex = None
            for possible_file in possible_files:
                try:
                    htex = valkyria.files.valk_open(possible_file)[0]
                except FileNotFoundError:
                    continue
            if htex is not None:
                htex.find_inner_files()
                self.hmdl_htex_pack = HTEX_Pack(htex, 0)
                self.hmdl_htex_pack.read_data(self)

    def build_blender(self, create_scene):
        if create_scene:
            self.create_scene(self.name)
            self.create_lamp()
        else:
            self.reuse_scene()
        self.create_collection(self.name)
        self.source_file.build_blender(self)
        self.source_file.finalize_blender()
        if isinstance(self.source_file, HMDL_Model) and hasattr(self, 'hmdl_htex_pack'):
            self.hmdl_htex_pack.build_blender(self)
            self.source_file.assign_materials(self.hmdl_htex_pack.htsf_images)
        if self.rotate_scene:
            self.fix_rotation()

    def fix_rotation(self):
        self.view_layer.update()

        for obj in [*self.collection.objects, *self.extra_objects]:
            if not obj.parent:
                obj.matrix_world = ROTATE_SCENE_MATRIX @ obj.matrix_world

    def pose_blender(self, pose_filename):
        poses = IZCA_Poses(valkyria.files.valk_open(pose_filename)[0])
        poses.F.find_inner_files()
        poses.read_data(self)
        poses.pose_model(self.source_file)


class ImportValkyria(bpy.types.Operator, ImportHelper):
    bl_idname = 'import_scene.import_valkyria'
    bl_label = 'Valkyria Chronicles (.MLX, .HMD, .ABR, .MXE)'
    bl_options = {'UNDO'}

    filename_ext = "*.mlx"
    filter_glob: bpy.props.StringProperty(
            default = "*.mlx;*.hmd;*.abr;*.mxe",
            options = {'HIDDEN'},
            )

    create_scene: bpy.props.BoolProperty(
            default=True, name="Create Scene",
            description="Create a new scene"
            )

    rotate_scene: bpy.props.BoolProperty(
            default=True, name="Rotate Scene",
            description="Rotate the imported scene to match the Blender 'Z Up' convention"
            )

    lod_filter: bpy.props.BoolProperty(
            default=False, name="Skip LOD Meshes",
            description="Skip loading reduced level of detail meshes (heuristic hack)"
            )

    def import_file(self, context, filename):
        vfile = valkyria.files.valk_open(filename)[0]
        vfile.find_inner_files()
        if vfile.ftype == 'IZCA':
            model = IZCA_Model(vfile)
        elif vfile.ftype == 'HMDL':
            model = HMDL_Model(vfile, 0)
        elif vfile.ftype == 'ABRS':
            model = ABRS_Model(vfile)
        elif vfile.ftype == 'MXEN':
            model = MXEN_Model(vfile)
        elif vfile.ftype == 'HTEX':
            pack = HTEX_Pack(vfile, 0)
            vscene = DummyScene()
            pack.read_data(vscene)
            pack.build_blender(vscene)
            pack.build_raw_texture_planes()
            return
        else:
            self.report({'ERROR'}, "Unknown module file type: "+vfile.ftype)
            return
        self.valk_scene = ValkyriaScene(
            context, model, filename,
            rotate_scene=self.rotate_scene,
            lod_filter=self.lod_filter,
        )
        try:
            self.valk_scene.read_data()
        except FileNotFoundError as e:
            message = 'This model requires a separate file which could not be found:\n'
            message += '    ' + str(e)
            message += '\nTry finding the file manually and copying it into the same folder as the model you attempted to open.'
            self.report({'ERROR'}, message)
        self.valk_scene.build_blender(self.create_scene)
        #pose_filename = os.path.join(os.path.dirname(filename), "VALCA02AD.MLX")
        #self.valk_scene.pose_blender(pose_filename)

    def execute(self, context):
        self.import_file(context, os.path.realpath(self.filepath))
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(ImportValkyria.bl_idname)

def register():
    bpy.utils.register_class(ImportValkyria)
    bpy.types.TOPBAR_MT_file_import.append(menu_func)

def unregister():
    bpy.utils.unregister_class(ImportValkyria)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func)
