import bpy
import os
import math
import platform
import subprocess
import traceback
import hashlib

from itertools import count
from mathutils import Vector
from bpy_extras.image_utils import load_image

DEBUG = False


def sha1hash(data):
    m = hashlib.sha1()
    m.update(data)
    return m.hexdigest()

def remove_file_safe(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


class ImageManager:
    """Imports texture images with duplicate removal based on SHA1 hash of image data."""
    def __init__(self):
        self.cache = {}
        self.index_images()

    def index_images(self):
        for img in bpy.data.images:
            if 'valkyria_hash' in img:
                self.cache[img['valkyria_hash']] = img

    def convert_dds_to_png(self, dds_path):
        texconv = os.path.join(os.path.dirname(__file__), 'texconv.exe')
        if not os.path.isfile(texconv):
            return None

        is_windows = platform.system().lower().startswith('win')
        dds_dir, dds_file = os.path.split(dds_path)
        png_path = dds_path[0:-4] + '.PNG'

        if is_windows:
            if subprocess.call([texconv, '-ft', 'png', dds_file], cwd=dds_dir) != 0:
                print('Could not convert {} to PNG'.format(dds_path))
                return None
        else:
            # Saving PNG is broken in Wine, so go in two steps via TGA
            tga_path = dds_path[0:-4] + '.TGA'
            try:
                if subprocess.call(['wine', texconv, '-ft', 'tga', dds_file], cwd=dds_dir) != 0:
                    print('Could not convert {} to TGA'.format(dds_path))
                    return None

                if subprocess.call(['convert', tga_path, png_path]) != 0:
                    print('Could not convert {} to PNG'.format(tga_path))
                    return None
            finally:
                remove_file_safe(tga_path)

        return png_path

    def load_image_data(self, filename, dds_data):
        dds_path = os.path.join(bpy.app.tempdir, filename)
        try:
            with open(dds_path, 'wb') as f:
                f.write(dds_data)

            image = load_image(dds_path)

            if image.size[0] == 0 and image.size[1] == 0:
                # Use texconv to convert unsupported BC7 compressed files to PNG.
                png_path = self.convert_dds_to_png(dds_path)

                if png_path and os.path.isfile(png_path):
                    bpy.data.images.remove(image)
                    image = load_image(png_path)

            image.pack()
            return image
        finally:
            remove_file_safe(dds_path)

    def detect_normal_map(self, image):
        pixels = image.pixels

        if image.channels != 4:
            return False

        # Test 100 points on the image
        num_pixels = int(len(pixels) / 4)
        num_points = min(num_pixels, 100)

        midvec = Vector((0.5, 0.5, 0.5))
        sumvec = Vector((0, 0, 0))

        for i in range(0, num_pixels, int(num_pixels / num_points)):
            color = Vector((pixels[i*4], pixels[i*4 + 1], pixels[i*4 + 2]))
            normal = (color - midvec) * 2
            # Require a valid normalized vector (allowing for precision issues e.g. in evmap04_02)
            if normal.z <= 0 or abs(normal.length - 1.0) > 0.3:
                return False
            sumvec += normal

        # Require the average to point in the right direction
        return sumvec.normalized().z > 0.8

    def detect_alpha(self, image):
        if image.channels != 4:
            return False

        if image.depth in (8, 24):
            return False

        if image.depth not in (16, 32):
            print('Unexpected image bit depth: ', image.depth)

        return True

    def load_image(self, htsf_image):
        data = htsf_image.dds.data
        key = sha1hash(data)

        if key in self.cache:
            htsf_image.image_ref = image = self.cache[key]
            htsf_image.is_normal_map = image['valkyria_is_normal']
        else:
            image = self.load_image_data(htsf_image.filename, data)
            is_normal = self.detect_normal_map(image)

            image['valkyria_hash'] = key
            image['valkyria_is_normal'] = is_normal
            htsf_image.image_ref = self.cache[key] = image
            htsf_image.is_normal_map = is_normal

            # Keep the alpha channel separate (otherwise e.g. eyes break)
            image.alpha_mode = 'CHANNEL_PACKED' if self.detect_alpha(image) else 'NONE'


class NodeTreeBuilder:
    def __init__(self, owner, tree):
        self.owner = owner
        self.tree = tree

    def mknode(self, type, x, y, **options):
        node = self.tree.nodes.new(type=type)
        node.location = (x, y)
        for k,v in options.items():
            setattr(node, k, v)
        return node

    def link(self, src, src_field, dst, dst_field):
        return self.tree.links.new(src.outputs[src_field], dst.inputs[dst_field])

    def link_in(self, dst, dst_field, src_socket):
        if src_socket is not None:
            if isinstance(src_socket, bpy.types.NodeSocket):
                return self.tree.links.new(src_socket, dst.inputs[dst_field])
            else:
                dst_socket = dst.inputs[dst_field]

                if isinstance(dst_socket, (bpy.types.NodeSocketColor, bpy.types.NodeSocketVector)):
                    if not isinstance(src_socket, tuple):
                        src_socket = (src_socket, src_socket, src_socket, 1)
                    src_socket = tuple([*src_socket, 1][0:len(dst_socket.default_value)])
                elif isinstance(dst_socket, (bpy.types.NodeSocketFloat, bpy.types.NodeSocketFloatFactor)):
                    if isinstance(src_socket, tuple):
                        src_socket = sum(src_socket[0:3])/3

                dst_socket.default_value = src_socket

    def link_out(self, src, src_field, dst_socket):
        if dst_socket is not None:
            return self.tree.links.new(src.outputs[src_field], dst_socket)

    def copy_link_in(self, dst, dst_field, src, src_field):
        src_socket = src.inputs[src_field]
        if src_socket.links:
            self.link_in(dst, dst_field, src_socket.links[0].from_socket)
        else:
            dst.inputs[dst_field].default_value = src_socket.default_value

    def make_group_node(self, group_name, x, y):
        return self.mknode(self.group_node_type, x, y, node_tree=self.owner.get_group(group_name))

    def deselect_all(self):
        for n in self.tree.nodes:
            n.select = False


class MaterialTreeBuilder(NodeTreeBuilder):
    def __init__(self, owner, matinfo, texture_pack, vcolors, mat):
        mat.use_nodes = True
        super().__init__(owner, mat.node_tree)

        self.info = matinfo
        self.texture_pack = texture_pack
        self.vertex_colors = vcolors
        self.material = mat

    group_node_type = 'ShaderNodeGroup'

    colsize = 200
    rowsize = 200
    colsize_bsdf = 300
    colsize_mid = 300
    colsize_tex = 400
    rowsize_tex = 300

    def clear_tree(self):
        nodes = self.tree.nodes
        self.node_out = None

        for n in list(nodes):
            if n.bl_idname == 'ShaderNodeOutputMaterial':
                self.node_out = n
            else:
                nodes.remove(n)

        if not self.node_out:
            self.node_out = nodes.new(type='ShaderNodeOutputMaterial')

        self.out_x, self.out_y = self.node_out.location
        self.main_x = self.out_x - self.colsize_bsdf
        self.node_main = self.mknode('ShaderNodeBsdfPrincipled', self.main_x, self.out_y)

        self.link(self.node_main, 0, self.node_out, 0)

    def create_textures(self):
        self.texture_nodes = {}
        self.texture_images = {}

        self.tex_x = self.out_x - 7 * self.colsize
        tex_y = self.out_y

        for i in range(5):
            texture = self.info.get('texture%d' % (i), None)

            if texture and 0 <= texture['image'] < len(self.texture_pack):
                self.texture_images[i] = imginfo = self.texture_pack[texture['image']]

                if DEBUG:
                    field_names = ['unk0', 'unk2', 'unk3a', 'unk7c']
                    tail = ''.join(' %s=%r'%(k,texture[k]) for k in field_names if k in texture)
                else:
                    tail = ''

                self.texture_nodes[i] = node_tex = self.mknode(
                    'ShaderNodeTexImage',
                    self.tex_x - self.colsize_tex, tex_y,
                    label='texture%d%s' % (i, tail),
                    image=imginfo.image,
                    width=self.colsize_tex - 20,
                )

                uv_id = 0 if 'all_uv0' in self.info.traits else i

                node_uv = self.mknode(
                    'ShaderNodeUVMap',
                    self.tex_x - self.colsize_tex - self.colsize_mid * 2, tex_y,
                    uv_map='UVMap-{}'.format(uv_id)
                )

                self.link(node_uv, 0, node_tex, 0)
                tex_y -= self.rowsize_tex

    def build_color(self):
        cur_x, cur_y = self.node_main.location
        cur_x -= self.colsize
        cur_socket = self.node_main.inputs['Base Color']

        # Multiply by the main vertex color if present
        self.node_mix_vcolor = None

        if 0 in self.vertex_colors:
            self.node_mix_vcolor = node_mix = self.mknode(
                'ShaderNodeMixRGB', cur_x, cur_y, blend_type='MULTIPLY'
            )
            node_color = self.mknode(
                'ShaderNodeVertexColor',
                cur_x - self.colsize, cur_y + self.rowsize,
                layer_name='Color-0'
            )
            self.link(node_color, 0, node_mix, 2)
            self.link_out(node_mix, 0, cur_socket)
            self.link_in(node_mix, 0, 1.0)
            cur_socket = node_mix.inputs[1]
            cur_x -= self.colsize

        # Create nodes for the vertex color used for blending textures
        use_mul_blend_all = 'texblend_mul' in self.info['traits']
        use_no_blend_alpha_all = 'texblend_noalpha' in self.info['traits']
        use_vc_blend = not use_mul_blend_all and 'texblend_vcolor1' in self.info['traits']

        if use_vc_blend and 1 in self.vertex_colors:
            vc_x = self.tex_x - self.colsize
            vc_y = self.out_y + self.colsize
            node_vc_blend_split = self.mknode('ShaderNodeSeparateRGB', vc_x, vc_y)
            node_vc_blend_color = self.mknode('ShaderNodeVertexColor', vc_x - self.colsize, vc_y, layer_name='Color-1')
            self.link(node_vc_blend_color, 0, node_vc_blend_split, 0)
            vc_blend_factors = node_vc_blend_split.outputs
        else:
            vc_blend_factors = (1,1,1)

        # Link the extra textures
        for i in range(4,0,-1):
            si = str(i)
            if i not in self.texture_nodes:
                continue

            if ('texture'+si) in self.info['traits']:
                use_mul_blend = use_mul_blend_all or ('texblend_mul'+si) in self.info['traits']
                use_no_blend_alpha = use_no_blend_alpha_all or ('texblend_noalpha'+si) in self.info['traits']
                node_mix = self.mknode(
                    'ShaderNodeMixRGB', cur_x, cur_y,
                    blend_type='MULTIPLY' if use_mul_blend else 'MIX'
                )
                self.link_out(node_mix, 0, cur_socket)
                if use_no_blend_alpha:
                    self.link_in(node_mix, 0, 1)
                elif use_vc_blend:
                    self.link_in(node_mix, 0, vc_blend_factors[i])
                else:
                    self.link(self.texture_nodes[i], 1, node_mix, 0)
                self.link(self.texture_nodes[i], 0, node_mix, 2)
                cur_socket = node_mix.inputs[1]
                cur_x -= self.colsize

            if ('specular'+si) in self.info['traits']:
                self.link(self.texture_nodes[i], 0, self.node_main, "Specular")

            if ('specular_alpha'+si) in self.info['traits']:
                self.link(self.texture_nodes[i], 1, self.node_main, "Specular")

        # Link the primary texture
        if 0 in self.texture_nodes:
            self.link_out(self.texture_nodes[0], 0, cur_socket)

            if 'alpha' in self.info['traits'] and self.texture_nodes[0].image.alpha_mode != 'NONE':
                self.link(self.texture_nodes[0], 1, self.node_main, 'Alpha')
                self.material.shadow_method = 'CLIP'
                if 'alphablend' in self.info['traits']:
                    self.material.blend_method = 'BLEND'
                else:
                    self.material.blend_method = 'CLIP'

    def build_normal(self):
        self.node_normal = None

        for i, texnode in self.texture_nodes.items():
            if self.texture_images[i].is_normal_map:
                if i != 1:
                    print('Material {}: texture {} is normal'.format(self.material.name, i))
                elif 'normal1' not in self.info['traits']:
                    print('Material {}: normal not expected'.format(self.material.name))

        if 'normal1' in self.info['traits'] and 1 in self.texture_nodes:
            self.texture_nodes[1].image.colorspace_settings.name = 'Non-Color'
            self.node_normal = node_normal = self.mknode(
                'ShaderNodeNormalMap',
                texnode.location.x + self.colsize_tex, texnode.location.y,
                uv_map='UVMap-{}'.format(1)
            )
            self.link(texnode, 0, node_normal, 'Color')
            self.link(node_normal, 0, self.node_main, 'Normal')

    def build_backface_cull(self):
        if self.info['use_backface_culling']:
            self.material.use_backface_culling = True

            old_out = self.node_out
            self.node_out = self.make_group_node('BackfaceCullOutput', self.out_x, self.out_y)
            self.copy_link_in(self.node_out, 0, old_out, 0)
            self.tree.nodes.remove(old_out)

    def switch_to_emission(self, base_color):
        self.copy_link_in(self.node_main, 'Emission', self.node_main, 'Base Color')
        self.tree.links.remove(self.node_main.inputs['Base Color'].links[0])
        self.link_in(self.node_main, 'Base Color', base_color)

    def build_water(self):
        self.link_in(self.node_main, 'Roughness', 0)
        self.link_in(self.node_main, 'Specular', 0.5)

        if self.node_normal:
            self.link_in(self.node_normal, 'Strength', 0.1)

        if 'specular' in self.info['traits']:
            self.link_in(self.node_main, 'Transmission', 1)
            self.link_in(self.node_main, 'IOR', 1.33)

            self.switch_to_emission((0.8,0.8,0.8))

            if self.node_mix_vcolor:
                self.node_mix_vcolor.blend_type = 'MIX'
                self.link_in(self.node_mix_vcolor, 0, 0.5)

            self.material.use_screen_refraction = True
            self.owner.vscene.scene.eevee.use_ssr = True
            self.owner.vscene.scene.eevee.use_ssr_refraction = True

        else:
            if self.node_mix_vcolor:
                self.node_mix_vcolor.blend_type = 'MIX'
                self.link_in(self.node_mix_vcolor, 0, 0)


    def build(self):
        self.clear_tree()
        self.create_textures()

        if 'specular' not in self.info['traits']:
            self.link_in(self.node_main, 'Specular', 0)

        self.build_color()
        self.build_normal()
        self.build_backface_cull()

        if 'water' in self.info['traits']:
            self.build_water()
        else:
            if 'unlit' in self.info['traits']:
                self.switch_to_emission((0,0,0))

        self.deselect_all()


class MaterialBuilder:
    def __init__(self, vscene):
        self.vscene = vscene
        self.known_groups = {}

        self.index_groups()

    def index_groups(self):
        for group in bpy.data.node_groups:
            if 'valkyria_special' in group:
                self.known_groups[group['valkyria_special']] = group

    def get_group(self, group_name):
        builder = getattr(self, 'build_group_'+group_name)

        group = self.known_groups.get(group_name, None)
        if not group:
            group = self.known_groups[group_name] = builder()
            group['valkyria_special'] = group_name

        return group


    def build_group_BackfaceCullOutput(self):
        node_tree = bpy.data.node_groups.new('BackfaceCullOutput', 'ShaderNodeTree')
        socket = node_tree.inputs.new('NodeSocketShader','Surface')

        nodes = {}
        nodes['Material Output.001'] = node = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node.name = 'Material Output.001'
        node.location = Vector((186.5911865234375, -71.14187622070312))
        node.target = 'EEVEE'
        node.inputs['Displacement'].default_value = (0.0, 0.0, 0.0)
        nodes['Transparent BSDF'] = node = node_tree.nodes.new('ShaderNodeBsdfTransparent')
        node.name = 'Transparent BSDF'
        node.location = Vector((-189.2015380859375, -157.58596801757812))
        node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        nodes['Material Output'] = node = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node.name = 'Material Output'
        node.location = Vector((189.20147705078125, 72.20443725585938))
        node.target = 'CYCLES'
        node.inputs['Displacement'].default_value = (0.0, 0.0, 0.0)
        nodes['Mix Shader'] = node = node_tree.nodes.new('ShaderNodeMixShader')
        node.name = 'Mix Shader'
        node.location = Vector((17.9478759765625, 32.75042724609375))
        nodes['Reroute'] = node = node_tree.nodes.new('NodeReroute')
        node.name = 'Reroute'
        node.location = Vector((-51.013336181640625, -128.97991943359375))
        nodes['Group Output'] = node = node_tree.nodes.new('NodeGroupOutput')
        node.name = 'Group Output'
        node.location = Vector((389.20147705078125, -0.0))
        nodes['Geometry'] = node = node_tree.nodes.new('ShaderNodeNewGeometry')
        node.name = 'Geometry'
        node.location = Vector((-188.635498046875, 157.58599853515625))
        nodes['Group Input'] = node = node_tree.nodes.new('NodeGroupInput')
        node.name = 'Group Input'
        node.location = Vector((-392.9754638671875, -93.25386810302734))

        node_tree.links.new(nodes['Reroute'].outputs['Output'],nodes['Mix Shader'].inputs[1])
        node_tree.links.new(nodes['Mix Shader'].outputs['Shader'],nodes['Material Output'].inputs['Surface'])
        node_tree.links.new(nodes['Transparent BSDF'].outputs['BSDF'],nodes['Mix Shader'].inputs[2])
        node_tree.links.new(nodes['Geometry'].outputs['Backfacing'],nodes['Mix Shader'].inputs['Fac'])
        node_tree.links.new(nodes['Group Input'].outputs['Surface'],nodes['Reroute'].inputs['Input'])
        node_tree.links.new(nodes['Reroute'].outputs['Output'],nodes['Material Output.001'].inputs['Surface'])
        return node_tree


    def build_material(self, name, matdata, texture_pack, vcolors):
        if vcolors:
            name += '-Color' + ''.join(str(i) for i in vcolors)

        mat = bpy.data.materials.new(name=name)
        MaterialTreeBuilder(self, matdata, texture_pack, vcolors, mat).build()

        if DEBUG:
            blocklist = {
                'id', 'ptr', 'use_backface_culling', 'num_textures', 'num_parameters', 'shader_hash',
                'texture0_ptr', 'texture1_ptr', 'texture2_ptr', 'texture3_ptr', 'texture4_ptr',
                'parameter_ptr',
            }

            for key, val in matdata.items():
                if key not in blocklist and isinstance(val, (int, float, tuple, str)):
                    try:
                        mat[key] = val
                    except OverflowError:
                        mat[key] = repr(val)

            for param in matdata.get('parameters',[]):
                mat[param.name] = param.data

            mat['traits'] = ','.join(sorted(matdata.get('traits',[])))

        return mat
