import bpy
import os
import math
import platform
import subprocess
import traceback
import hashlib

from itertools import count
from mathutils import Vector, Euler
from bpy_extras.image_utils import load_image

# Add unknown material structure fields as custom properties
DEBUG = False

IS_WINDOWS = platform.system().lower().startswith('win')
TEXCONV_PATH = os.path.join(os.path.dirname(__file__), 'texconv.exe')


def sha1hash(data):
    m = hashlib.sha1()
    m.update(data)
    return m.hexdigest()

def srgb_to_linear(color):
    return tuple(x/12.92 if x <= 0.04045 else pow((x+0.055)/1.055,2.4) for x in color)

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

        self.work_dir = bpy.app.tempdir

    def index_images(self):
        for img in bpy.data.images:
            if 'valkyria_hash' in img:
                self.cache[img['valkyria_hash']] = img

    def full_path(self, filename):
        return os.path.join(self.work_dir, filename)

    def load_image_file(self, filename):
        return load_image(os.path.join(self.work_dir, filename))

    def remove_file_safe(self, filename):
        if filename:
            remove_file_safe(os.path.join(self.work_dir, filename))

    def invoke_texconv(self, dds_file, fmt):
        if not os.path.isfile(TEXCONV_PATH):
            return None, ''

        assert fmt in ('PNG', 'TGA', 'HDR')

        prefix = [] if IS_WINDOWS else ['wine']
        command = [*prefix, TEXCONV_PATH, '-ft', fmt, dds_file]
        result_file = os.path.splitext(dds_file)[0] + '.' + fmt

        success = False
        self.remove_file_safe(result_file)

        try:
            output = subprocess.check_output(command, cwd=self.work_dir, stderr=subprocess.STDOUT)
            success = True
        except subprocess.CalledProcessError as e:
            output = e.output

        success = success and os.path.isfile(self.full_path(result_file))

        filtered_output = '\n'.join(
            line for line in output.decode("latin-1").split('\n')
            if line.startswith('reading')
            or 'FAILED' in line
            or 'ERROR' in line
        )
        print(filtered_output)

        return (result_file if success else None), filtered_output

    def convert_dds(self, dds_file):
        # PNG file writing is broken on Wine, so use TGA
        res_file, output = self.invoke_texconv(dds_file, 'PNG' if IS_WINDOWS else 'TGA')

        # Use HDR for floating point textures
        if 'R32G32B32A32_FLOAT' in output:
            self.remove_file_safe(res_file)

            res_file, output = self.invoke_texconv(dds_file, 'HDR')

        # Convert TGA to PNG
        if res_file and res_file.endswith('TGA'):
            tga_file = res_file
            try:
                res_file = os.path.splitext(dds_file)[0] + '.PNG'

                if subprocess.call(['convert', tga_file, res_file], cwd=self.work_dir) != 0:
                    print('Could not convert {} to PNG'.format(tga_file))
                    return None
            finally:
                self.remove_file_safe(tga_file)

        return res_file

    def load_image_data(self, filename, dds_data):
        conv_file = None
        try:
            with open(self.full_path(filename), 'wb') as f:
                f.write(dds_data)

            image = self.load_image_file(filename)

            if image.size[0] == 0 and image.size[1] == 0:
                # Use texconv to convert unsupported BC7 compressed files to PNG.
                conv_file = self.convert_dds(filename)

                if conv_file and os.path.isfile(self.full_path(conv_file)):
                    bpy.data.images.remove(image)
                    image = self.load_image_file(conv_file)

            image.pack()
            return image
        finally:
            self.remove_file_safe(filename)
            self.remove_file_safe(conv_file)

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

        if image.depth in (8, 24, 96):
            return False

        if image.depth not in (16, 32, 128):
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

    def unlink_socket(self, socket):
        for link in list(socket.links):
            self.tree.links.remove(link)

    def link_in(self, dst, dst_field, src_socket):
        if src_socket is not None:
            dst_socket = dst.inputs[dst_field]

            self.unlink_socket(dst_socket)

            if isinstance(src_socket, bpy.types.NodeSocket):
                return self.tree.links.new(src_socket, dst_socket)
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
        self.texture_info = {}
        self.texture_images = {}
        self.texture_uvmaps = {}

        self.tex_x = self.out_x - 7 * self.colsize
        tex_y = self.out_y

        for i in range(5):
            si = str(i)
            self.texture_info[i] = texture = self.info.get('texture%d' % (i), None)

            if texture and 0 <= texture['image'] < len(self.texture_pack):
                self.texture_images[i] = imginfo = self.texture_pack[texture['image']]

                if DEBUG:
                    field_names = ['unk0', 'blend_factor', 'unk3a', 'unk7c']
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

                if i == 0 and 'ice' in self.info.traits:
                    params = self.info.parameters.get('cIceParallaxParam',(0,0.1,0,0))
                    node_uv = self.make_group_node(
                        'ParallaxUVMap0',
                        self.tex_x - self.colsize_tex - self.colsize_mid * 2, tex_y,
                    )
                    self.link_in(node_uv, 'UV Factor', texture.get('blend_factor',1))
                    self.link_in(node_uv, 'Parallax Scale', params[1] if 'parallax' in self.info.traits else 0)
                else:
                    uv_id = i

                    prefix = 'tex'+si+'_uv'
                    matched = [ t for t in self.info.traits if t.startswith(prefix) ]

                    if matched:
                        uv_id = int(matched[0][len(prefix):])
                    elif 'all_uv0' in self.info.traits:
                        uv_id = 0

                    self.texture_uvmaps[i] = uv_id

                    node_uv = self.mknode(
                        'ShaderNodeUVMap',
                        self.tex_x - self.colsize_tex - self.colsize_mid * 2, tex_y,
                        uv_map='UVMap-{}'.format(uv_id)
                    )

                self.link(node_uv, 0, node_tex, 0)
                tex_y -= self.rowsize_tex

        if 'warp_uv0_tex1' in self.info['traits'] and 0 in self.texture_nodes and 1 in self.texture_nodes:
            self.texture_nodes[1].image.colorspace_settings.name = 'Non-Color'
            node_tex0 = self.texture_nodes[0]
            node_warp = self.make_group_node('WarpUV', node_tex0.location.x-self.colsize, node_tex0.location.y)
            self.copy_link_in(node_warp, 'UV', node_tex0, 0)
            self.link_in(node_warp, 'Scale', self.texture_info[1].get('blend_factor', 1))
            self.link(self.texture_nodes[1], 0, node_warp, 'Warp Map')
            self.link(node_warp, 0, node_tex0, 0)

    def build_color(self):
        cur_x, cur_y = self.node_main.location
        cur_x -= self.colsize
        cur_socket = self.node_main.inputs['Base Color']

        '''
        This is wrong, but keeping as an example in case a field is identified.
        matcolor = self.info.get('unk5a', (1,1,1,1))[0:3]
        if matcolor != (1,1,1):
            self.node_mix_matcolor = node_mix = self.mknode(
                'ShaderNodeMixRGB', cur_x, cur_y, blend_type='MULTIPLY'
            )
            self.link_out(node_mix, 0, cur_socket)
            self.link_in(node_mix, 0, 1.0)
            self.link_in(node_mix, 2, srgb_to_linear(matcolor))
            cur_socket = node_mix.inputs[1]
            cur_x -= self.colsize
        '''

        # Multiply by the main vertex color if present
        self.node_mix_vcolor = None
        node_vcolor0 = None

        if 0 in self.vertex_colors and 'texblend_footprint' not in self.info['traits']:
            self.node_mix_vcolor = node_mix = self.mknode(
                'ShaderNodeMixRGB', cur_x, cur_y, blend_type='MULTIPLY'
            )
            node_vcolor0 = self.mknode(
                'ShaderNodeVertexColor',
                cur_x - self.colsize, cur_y + self.rowsize,
                layer_name='Color-0'
            )
            self.link(node_vcolor0, 0, node_mix, 2)
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
                use_add_blend = ('texblend_add'+si) in self.info['traits']
                use_mul_blend = ('texblend_mul'+si) in self.info['traits']

                if ('texblend_lm'+si) in self.info['traits']:
                    if self.info.parameters.get('lightmap_compo_type'):
                        use_add_blend = True
                    else:
                        use_mul_blend = True

                if use_add_blend:
                    blend_type = 'ADD'
                elif use_mul_blend or use_mul_blend_all:
                    blend_type = 'MULTIPLY'
                else:
                    blend_type = 'MIX'

                use_no_blend_alpha = use_no_blend_alpha_all or ('texblend_noalpha'+si) in self.info['traits']
                node_mix = self.mknode('ShaderNodeMixRGB', cur_x, cur_y, blend_type=blend_type)
                self.link_out(node_mix, 0, cur_socket)

                use_footprint = (i == 1) and 'texblend_footprint' in self.info['traits']

                # Multiplication by the constant blend factor
                blend_factor = self.texture_info[i].get('blend_factor', 1)
                if blend_factor != 1 and not use_footprint:
                    node_blend_fac = self.mknode('ShaderNodeMath', cur_x-self.colsize, cur_y+self.rowsize, operation='MULTIPLY')
                    self.link_in(node_blend_fac, 1, blend_factor)
                    self.link(node_blend_fac, 0, node_mix, 0)
                else:
                    node_blend_fac = node_mix

                # Extra multiplication by alpha of vcolor0
                if ('texblend_vcolor0_alpha_ex'+si) in self.info['traits'] and node_vcolor0:
                    node_blend_fac1 = node_blend_fac
                    node_blend_fac = self.mknode(
                        'ShaderNodeMath',
                        node_blend_fac1.location.x-self.colsize, node_blend_fac1.location.y+self.rowsize,
                        operation='MULTIPLY'
                    )
                    node_vcolor0.location.x = min(node_vcolor0.location.x, node_blend_fac.location.x - self.colsize)
                    self.link(node_vcolor0, 1, node_blend_fac, 1)
                    self.link(node_blend_fac, 0, node_blend_fac1, 0)

                # Main blend factor source
                if use_footprint:
                    if 0 in self.vertex_colors:
                        # It actually uses some kind of height map texture in the vertex shader - not reproduced here
                        node_footprint = self.make_group_node('FootprintMask', cur_x-self.colsize, cur_y+self.rowsize)
                        self.link(node_footprint, 0, node_blend_fac, 0)
                    else:
                        self.link_in(node_blend_fac, 0, 1)
                elif ('texblend_vcolor0_alpha'+si) in self.info['traits']:
                    # This is for using alpha of vcolor0 as the only blend factor
                    if node_vcolor0:
                        node_vcolor0.location.x = min(node_vcolor0.location.x, node_blend_fac.location.x - self.colsize)
                        self.link(node_vcolor0, 1, node_blend_fac, 0)
                    else:
                        self.link_in(node_blend_fac, 0, 1)
                elif use_no_blend_alpha:
                    self.link_in(node_blend_fac, 0, 1)
                elif use_vc_blend:
                    self.link_in(node_blend_fac, 0, vc_blend_factors[i])
                else:
                    self.link(self.texture_nodes[i], 1, node_blend_fac, 0)

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

        self.has_alpha = False

        if 'alpha' in self.info['traits']:
            use_vcolor0_alpha = not any(n.startswith('texblend_vcolor0_alpha') for n in self.info['traits'])

            if 0 in self.texture_nodes and self.texture_nodes[0].image.alpha_mode != 'NONE':
                if node_vcolor0 and use_vcolor0_alpha:
                    node_mix = self.mknode(
                        'ShaderNodeMath',
                        self.node_main.location.x - self.colsize,
                        self.node_main.location.y + self.rowsize,
                        operation='MULTIPLY'
                    )
                    self.link(node_vcolor0, 1, node_mix, 0)
                    self.link(self.texture_nodes[0], 1, node_mix, 1)
                    self.link(node_mix, 0, self.node_main, 'Alpha')
                    self.has_alpha = True
                else:
                    self.link(self.texture_nodes[0], 1, self.node_main, 'Alpha')
                    self.has_alpha = True
            else:
                # TODO: figure out the correct rules for enabling alpha & alphablend in VC1, and
                # remove the VC4 check. As is, enabling alpha here in VC1 makes maps look awful
                # due to everything being Alpha Blend and simplistic sorting Eevee uses.
                if node_vcolor0 and use_vcolor0_alpha and 'v4' in self.info['traits']:
                    self.link(node_vcolor0, 1, self.node_main, 'Alpha')
                    self.has_alpha = True

            if self.has_alpha:
                if 'alphablend' in self.info['traits']:
                    self.material.shadow_method = 'CLIP'
                    self.material.blend_method = 'BLEND'
                else:
                    self.material.shadow_method = 'CLIP'
                    self.material.blend_method = 'CLIP'

    def build_displace(self, texnode):
        texnode.image.colorspace_settings.name = 'Non-Color'
        disp_node = self.make_group_node(
            'DisplaceWorldYasZ' if self.owner.rotate_scene else 'DisplaceWorldY',
            texnode.location.x + self.colsize_tex, texnode.location.y
        )
        disp_node.name = "Displacement"
        param = self.info.parameters.get('cDisplacementParam', (1,1,1,0))
        self.link(texnode, 0, disp_node, 'Image')
        self.link_in(disp_node, 'Scale', param[0])

    def build_normal(self):
        self.node_normal = None

        use_normal2 = {'v4','water'}.issubset(self.info['traits'])

        for i, texnode in self.texture_nodes.items():
            if ('displace'+str(i)) in self.info['traits']:
                self.build_displace(texnode)

            elif self.texture_images[i].is_normal_map:
                if i == 2 and use_normal2:
                    pass
                elif i != 1:
                    print('Material {}: texture {} is normal'.format(self.material.name, i))
                elif 'normal1' not in self.info['traits'] and 'warp_uv0_tex1' not in self.info['traits']:
                    print('Material {}: normal not expected'.format(self.material.name))

        if 'normal1' in self.info['traits'] and 1 in self.texture_nodes:
            texnode = self.texture_nodes[1]
            texnode.image.colorspace_settings.name = 'Non-Color'
            cur_x = texnode.location.x + self.colsize_tex

            if use_normal2 and 2 in self.texture_nodes:
                texnode2 = self.texture_nodes[2]
                texnode2.image.colorspace_settings.name = 'Non-Color'
                node_mix = self.mknode('ShaderNodeMixRGB', cur_x, texnode.location.y, blend_type='MIX')
                self.link_in(node_mix, 0, self.info.parameters.get('cWaterSurfPics',(0,0.5,0,0))[1])
                self.link(texnode, 0, node_mix, 1)
                self.link(texnode2, 0, node_mix, 2)
                cur_x += self.colsize
                texnode = node_mix

            self.node_normal = node_normal = self.mknode(
                'ShaderNodeNormalMap',
                cur_x, texnode.location.y,
                uv_map='UVMap-{}'.format(self.texture_uvmaps[1])
            )
            self.link_in(node_normal, 'Strength', self.texture_info[1].get('blend_factor', 1))
            self.link(texnode, 0, node_normal, 'Color')
            self.link(node_normal, 0, self.node_main, 'Normal')

    def build_add_shader(self):
        self.material.shadow_method = 'NONE'
        self.material.blend_method = 'BLEND'
        node_add_blend = self.make_group_node('AddBlend', self.out_x, self.out_y)
        self.node_out.location.x = self.out_x = self.out_x + self.colsize
        self.copy_link_in(node_add_blend, 0, self.node_out, 0)
        self.copy_link_in(node_add_blend, 'Alpha', self.node_main, 'Alpha')
        self.link(node_add_blend, 0, self.node_out, 0)
        self.link_in(self.node_main, 'Alpha', 1)

    def build_backface_cull(self):
        if self.info['use_backface_culling']:
            self.material.use_backface_culling = True

            old_out = self.node_out
            self.node_out = self.make_group_node('BackfaceCullOutput', self.out_x, self.out_y)
            self.node_out.inputs['Displacement'].hide_value = True
            self.copy_link_in(self.node_out, 0, old_out, 0)
            self.tree.nodes.remove(old_out)

    def switch_to_emission(self, base_color):
        self.copy_link_in(self.node_main, 'Emission', self.node_main, 'Base Color')
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

        if 'water' in self.info['traits'] and 'v4' not in self.info['traits']:
            self.build_water()
        else:
            if 'roughness00' in self.info['traits']:
                self.link_in(self.node_main, 'Roughness', 0.0)
            if 'roughness20' in self.info['traits']:
                self.link_in(self.node_main, 'Roughness', 0.2)
            if 'roughness30' in self.info['traits']:
                self.link_in(self.node_main, 'Roughness', 0.3)
            if 'unlit' in self.info['traits']:
                self.switch_to_emission((0,0,0))

        if 'add_shader' in self.info['traits']:
            self.build_add_shader()

        self.build_backface_cull()

        self.deselect_all()


class MaterialBuilder:
    def __init__(self, vscene, rotate_scene=True):
        self.vscene = vscene
        self.known_groups = {}
        self.rotate_scene = rotate_scene

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

            for name, param in matdata.parameters.items():
                mat[name] = param

            mat['traits'] = ','.join(sorted(matdata.get('traits',[])))

        return mat

    ##################################################################
    # Manually Designed Utility Node Groups (auto-converted to code) #
    ##################################################################

    def build_group_BackfaceCullOutput(self):
        node_tree = bpy.data.node_groups.new('BackfaceCullOutput', 'ShaderNodeTree')
        socket = node_tree.inputs.new('NodeSocketShader','Surface')
        socket = node_tree.inputs.new('NodeSocketVector','Displacement')
        if hasattr(socket, 'hide_value'):
            socket.hide_value = True

        nodes = {}
        nodes['Material Output.001'] = node = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node.name = 'Material Output.001'
        node.location = Vector((186.5911865234375, -71.14187622070312))
        node.target = 'EEVEE'
        nodes['Material Output'] = node = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node.name = 'Material Output'
        node.location = Vector((189.20147705078125, 72.20443725585938))
        node.is_active_output = False
        node.target = 'CYCLES'
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
        nodes['Reroute.001'] = node = node_tree.nodes.new('NodeReroute')
        node.name = 'Reroute.001'
        node.location = Vector((159.6106719970703, -150.4766387939453))
        nodes['Transparent BSDF'] = node = node_tree.nodes.new('ShaderNodeBsdfTransparent')
        node.name = 'Transparent BSDF'
        node.location = Vector((-187.24911499023438, -173.21990966796875))
        node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

        node_tree.links.new(nodes['Reroute'].outputs['Output'],nodes['Mix Shader'].inputs[1])
        node_tree.links.new(nodes['Mix Shader'].outputs['Shader'],nodes['Material Output'].inputs['Surface'])
        node_tree.links.new(nodes['Transparent BSDF'].outputs['BSDF'],nodes['Mix Shader'].inputs[2])
        node_tree.links.new(nodes['Geometry'].outputs['Backfacing'],nodes['Mix Shader'].inputs['Fac'])
        node_tree.links.new(nodes['Group Input'].outputs['Surface'],nodes['Reroute'].inputs['Input'])
        node_tree.links.new(nodes['Reroute'].outputs['Output'],nodes['Material Output.001'].inputs['Surface'])
        node_tree.links.new(nodes['Reroute.001'].outputs['Output'],nodes['Material Output'].inputs['Displacement'])
        node_tree.links.new(nodes['Group Input'].outputs['Displacement'],nodes['Reroute.001'].inputs['Input'])
        node_tree.links.new(nodes['Reroute.001'].outputs['Output'],nodes['Material Output.001'].inputs['Displacement'])
        return node_tree


    def build_group_ParallaxUVMap0(self):
        node_tree = bpy.data.node_groups.new('ParallaxUVMap0', 'ShaderNodeTree')

        socket = node_tree.inputs.new('NodeSocketFloatFactor','UV Factor')
        socket.default_value = 1.0
        socket.min_value = 0.0
        socket.max_value = 1.0
        socket = node_tree.inputs.new('NodeSocketFloat','Parallax Scale')
        socket.default_value = 0.1
        socket.min_value = 0.0
        socket.max_value = 10000.0
        socket = node_tree.outputs.new('NodeSocketVector','UV')

        nodes = {}
        nodes['Combine XYZ'] = node = node_tree.nodes.new('ShaderNodeCombineXYZ')
        node.name = 'Combine XYZ'
        node.location = Vector((208.4141845703125, 9.34918212890625))
        node.inputs['Z'].default_value = 0.0
        nodes['Vector Math.003'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math.003'
        node.location = Vector((-168.272705078125, -61.94183349609375))
        node.hide = True
        node.operation = 'CROSS_PRODUCT'
        nodes['Vector Math.004'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math.004'
        node.location = Vector((29.387451171875, -65.856689453125))
        node.hide = True
        node.operation = 'DOT_PRODUCT'
        nodes['Vector Math.005'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math.005'
        node.location = Vector((27.900146484375, -23.601318359375))
        node.hide = True
        node.operation = 'DOT_PRODUCT'
        nodes['Vector Math.001'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math.001'
        node.location = Vector((-167.75634765625, -116.094482421875))
        node.hide = True
        node.operation = 'SUBTRACT'
        nodes['Vector Math'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math'
        node.location = Vector((-363.2919921875, -150.73992919921875))
        node.hide = True
        node.operation = 'PROJECT'
        nodes['Vector Math.007'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math.007'
        node.location = Vector((404.1990966796875, -24.2728271484375))
        node.hide = True
        node.operation = 'SCALE'
        nodes['Group Output'] = node = node_tree.nodes.new('NodeGroupOutput')
        node.name = 'Group Output'
        node.location = Vector((773.3886108398438, 52.410404205322266))
        nodes['Geometry'] = node = node_tree.nodes.new('ShaderNodeNewGeometry')
        node.name = 'Geometry'
        node.location = Vector((-574.841552734375, -2.01385498046875))
        nodes['Tangent'] = node = node_tree.nodes.new('ShaderNodeTangent')
        node.name = 'Tangent'
        node.location = Vector((-371.5527648925781, 45.700469970703125))
        node.direction_type = 'UV_MAP'
        node.uv_map = 'UVMap-0'
        nodes['UV Map'] = node = node_tree.nodes.new('ShaderNodeUVMap')
        node.name = 'UV Map'
        node.location = Vector((203.39146423339844, 139.75259399414062))
        node.uv_map = 'UVMap-0'
        nodes['Mix'] = node = node_tree.nodes.new('ShaderNodeMixRGB')
        node.name = 'Mix'
        node.location = Vector((406.718017578125, 124.55484771728516))
        node.hide = True
        node.blend_type = 'MIX'
        nodes['Group Input'] = node = node_tree.nodes.new('NodeGroupInput')
        node.name = 'Group Input'
        node.location = Vector((209.5933837890625, -129.18670654296875))
        nodes['Vector Math.006'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math.006'
        node.location = Vector((574.8416748046875, 32.70790481567383))
        node.hide = True
        node.operation = 'SUBTRACT'
        nodes['Mapping'] = node = node_tree.nodes.new('ShaderNodeMapping')
        node.name = 'Mapping'
        node.location = Vector((28.314830780029297, 359.1101379394531))
        node.vector_type = 'POINT'
        node.inputs['Location'].default_value = Vector((-1.0, -1.0, -1.0))
        node.inputs['Rotation'].default_value = Euler((0.0, 0.0, 0.0), 'XYZ')
        node.inputs['Scale'].default_value = Vector((2.0, 2.0, 2.0))
        nodes['Texture Coordinate'] = node = node_tree.nodes.new('ShaderNodeTexCoord')
        node.name = 'Texture Coordinate'
        node.location = Vector((-165.31988525390625, 354.2591857910156))

        node_tree.links.new(nodes['Geometry'].outputs['Incoming'],nodes['Vector Math'].inputs[0])
        node_tree.links.new(nodes['Geometry'].outputs['Incoming'],nodes['Vector Math.001'].inputs[0])
        node_tree.links.new(nodes['Vector Math'].outputs['Vector'],nodes['Vector Math.001'].inputs[1])
        node_tree.links.new(nodes['Vector Math.003'].outputs['Vector'],nodes['Vector Math.004'].inputs[0])
        node_tree.links.new(nodes['Vector Math.001'].outputs['Vector'],nodes['Vector Math.004'].inputs[1])
        node_tree.links.new(nodes['Tangent'].outputs['Tangent'],nodes['Vector Math.005'].inputs[0])
        node_tree.links.new(nodes['Vector Math.001'].outputs['Vector'],nodes['Vector Math.005'].inputs[1])
        node_tree.links.new(nodes['Vector Math.006'].outputs['Vector'],nodes['Group Output'].inputs['UV'])
        node_tree.links.new(nodes['Combine XYZ'].outputs['Vector'],nodes['Vector Math.007'].inputs[0])
        node_tree.links.new(nodes['Vector Math.005'].outputs['Value'],nodes['Combine XYZ'].inputs['X'])
        node_tree.links.new(nodes['Vector Math.004'].outputs['Value'],nodes['Combine XYZ'].inputs['Y'])
        node_tree.links.new(nodes['Geometry'].outputs['Normal'],nodes['Vector Math.003'].inputs[0])
        node_tree.links.new(nodes['Geometry'].outputs['Normal'],nodes['Vector Math'].inputs[1])
        node_tree.links.new(nodes['Tangent'].outputs['Tangent'],nodes['Vector Math.003'].inputs[1])
        node_tree.links.new(nodes['Vector Math.007'].outputs['Vector'],nodes['Vector Math.006'].inputs[1])
        node_tree.links.new(nodes['UV Map'].outputs['UV'],nodes['Mix'].inputs['Color2'])
        node_tree.links.new(nodes['Mix'].outputs['Color'],nodes['Vector Math.006'].inputs[0])
        node_tree.links.new(nodes['Texture Coordinate'].outputs['Window'],nodes['Mapping'].inputs['Vector'])
        node_tree.links.new(nodes['Group Input'].outputs['UV Factor'],nodes['Mix'].inputs['Fac'])
        node_tree.links.new(nodes['Mapping'].outputs['Vector'],nodes['Mix'].inputs['Color1'])
        node_tree.links.new(nodes['Group Input'].outputs['Parallax Scale'],nodes['Vector Math.007'].inputs['Scale'])
        return node_tree


    def build_group_AddBlend(self):
        node_tree = bpy.data.node_groups.new('AddBlend', 'ShaderNodeTree')
        socket = node_tree.inputs.new('NodeSocketShader','Shader')
        socket = node_tree.inputs.new('NodeSocketFloatFactor','Alpha')
        socket.default_value = 1.0
        socket.min_value = 0.0
        socket.max_value = 1.0
        socket = node_tree.outputs.new('NodeSocketShader','Shader')

        nodes = {}
        nodes['Transparent BSDF'] = node = node_tree.nodes.new('ShaderNodeBsdfTransparent')
        node.name = 'Transparent BSDF'
        node.location = (-53, -84)
        node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        nodes['Add Shader'] = node = node_tree.nodes.new('ShaderNodeAddShader')
        node.name = 'Add Shader'
        node.location = (166, 16)
        nodes['Group Output'] = node = node_tree.nodes.new('NodeGroupOutput')
        node.name = 'Group Output'
        node.location = (377, 10)
        nodes['Mix Shader'] = node = node_tree.nodes.new('ShaderNodeMixShader')
        node.name = 'Mix Shader'
        node.location = (-52, 60)
        nodes['Group Input'] = node = node_tree.nodes.new('NodeGroupInput')
        node.name = 'Group Input'
        node.location = (-276, 42)

        node_tree.links.new(nodes['Add Shader'].outputs['Shader'],nodes['Group Output'].inputs['Shader'])
        node_tree.links.new(nodes['Transparent BSDF'].outputs['BSDF'],nodes['Add Shader'].inputs[1])
        node_tree.links.new(nodes['Mix Shader'].outputs['Shader'],nodes['Add Shader'].inputs[0])
        node_tree.links.new(nodes['Group Input'].outputs['Shader'],nodes['Mix Shader'].inputs[2])
        node_tree.links.new(nodes['Group Input'].outputs['Alpha'],nodes['Mix Shader'].inputs['Fac'])
        return node_tree


    def build_group_FootprintMask(self):
        node_tree = bpy.data.node_groups.new('FootprintMask', 'ShaderNodeTree')
        socket = node_tree.outputs.new('NodeSocketFloat','Value')

        nodes = {}
        nodes['Group Output'] = node = node_tree.nodes.new('NodeGroupOutput')
        node.name = 'Group Output'
        node.location = Vector((463.9424133300781, -0.0))
        nodes['Vertex Color'] = node = node_tree.nodes.new('ShaderNodeVertexColor')
        node.name = 'Vertex Color'
        node.location = Vector((-263.9423828125, 9.416603088378906))
        node.layer_name = 'Color-0'
        nodes['Mapping'] = node = node_tree.nodes.new('ShaderNodeMapping')
        node.name = 'Mapping'
        node.location = Vector((-74.17340087890625, 112.81229400634766))
        node.vector_type = 'POINT'
        node.inputs['Location'].default_value = Vector((0.0, 1.0, 0.0))
        node.inputs['Rotation'].default_value = Euler((0.0, 0.0, 0.0), 'XYZ')
        node.inputs['Scale'].default_value = Vector((1.0, -1.0, 0.0))
        nodes['Separate XYZ'] = node = node_tree.nodes.new('ShaderNodeSeparateXYZ')
        node.name = 'Separate XYZ'
        node.location = Vector((96.73281860351562, 25.607032775878906))
        nodes['Math'] = node = node_tree.nodes.new('ShaderNodeMath')
        node.name = 'Math'
        node.location = Vector((263.9424133300781, 48.329872131347656))
        node.operation = 'MINIMUM'
        node.use_clamp = False
        node.inputs[1].default_value = 1.0

        node_tree.links.new(nodes['Vertex Color'].outputs['Color'],nodes['Mapping'].inputs['Vector'])
        node_tree.links.new(nodes['Mapping'].outputs['Vector'],nodes['Separate XYZ'].inputs['Vector'])
        node_tree.links.new(nodes['Separate XYZ'].outputs['X'],nodes['Math'].inputs[0])
        node_tree.links.new(nodes['Math'].outputs['Value'],nodes['Group Output'].inputs['Value'])
        return node_tree


    def build_group_DisplaceWorldYasZ(self):
        node_tree = bpy.data.node_groups.new('DisplaceWorldYasZ', 'ShaderNodeTree')
        self.build_group_displace(node_tree, 'G', 'B')
        return node_tree

    def build_group_DisplaceWorldY(self):
        node_tree = bpy.data.node_groups.new('DisplaceWorldY', 'ShaderNodeTree')
        self.build_group_displace(node_tree, 'G', 'G')
        return node_tree

    def build_group_displace(self, node_tree, color_from, color_to):
        socket = node_tree.inputs.new('NodeSocketColor','Image')
        socket = node_tree.inputs.new('NodeSocketFloat','Scale')
        socket.default_value = 1.0
        socket.min_value = 0.0
        socket.max_value = 1000.0
        socket = node_tree.outputs.new('NodeSocketVector','Displacement')

        nodes = {}
        nodes['Separate RGB'] = node = node_tree.nodes.new('ShaderNodeSeparateRGB')
        node.name = 'Separate RGB'
        node.location = Vector((-172.277099609375, 0.9827880859375))
        nodes['Group Output'] = node = node_tree.nodes.new('NodeGroupOutput')
        node.name = 'Group Output'
        node.location = Vector((372.27716064453125, -0.0))
        nodes['Vector Displacement.001'] = node = node_tree.nodes.new('ShaderNodeVectorDisplacement')
        node.name = 'Vector Displacement.001'
        node.location = Vector((172.27716064453125, -0.98284912109375))
        node.space = 'WORLD'
        node.inputs['Midlevel'].default_value = 0.5
        nodes['Combine RGB'] = node = node_tree.nodes.new('ShaderNodeCombineRGB')
        node.name = 'Combine RGB'
        node.location = Vector((-0.4635009765625, 0.005615234375))
        node.inputs['R'].default_value = 0.5
        node.inputs['G'].default_value = 0.5
        node.inputs['B'].default_value = 0.5
        nodes['Group Input'] = node = node_tree.nodes.new('NodeGroupInput')
        node.name = 'Group Input'
        node.location = Vector((-364.29052734375, -76.51406860351562))

        node_tree.links.new(nodes['Group Input'].outputs['Image'],nodes['Separate RGB'].inputs['Image'])
        node_tree.links.new(nodes['Combine RGB'].outputs['Image'],nodes['Vector Displacement.001'].inputs['Vector'])
        node_tree.links.new(nodes['Vector Displacement.001'].outputs['Displacement'],nodes['Group Output'].inputs['Displacement'])
        node_tree.links.new(nodes['Group Input'].outputs['Scale'],nodes['Vector Displacement.001'].inputs['Scale'])
        node_tree.links.new(nodes['Separate RGB'].outputs[color_from],nodes['Combine RGB'].inputs[color_to])


    def build_group_WarpUV(self):
        node_tree = bpy.data.node_groups.new('WarpUV', 'ShaderNodeTree')
        socket = node_tree.inputs.new('NodeSocketVector','UV')
        socket = node_tree.inputs.new('NodeSocketColor','Warp Map')
        socket.default_value = (0.5, 0.5, 0.5, 1.0)
        socket = node_tree.inputs.new('NodeSocketFloat','Scale')
        socket.default_value = 1.0
        socket.min_value = 0
        socket.max_value = 10000.0
        socket = node_tree.outputs.new('NodeSocketVector','UV')

        nodes = {}
        nodes['Vector Math'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math'
        node.location = Vector((180.054931640625, 69.86552429199219))
        node.operation = 'ADD'
        nodes['Vector Math.001'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math.001'
        node.location = Vector((-2.255859375, -29.963592529296875))
        node.operation = 'SCALE'
        nodes['Vector Math.002'] = node = node_tree.nodes.new('ShaderNodeVectorMath')
        node.name = 'Vector Math.002'
        node.location = Vector((-178.4896240234375, -5.614990234375))
        node.operation = 'SUBTRACT'
        node.inputs[1].default_value = (0.5, 0.5, 0.5)
        nodes['Math.001'] = node = node_tree.nodes.new('ShaderNodeMath')
        node.name = 'Math.001'
        node.location = Vector((-180.054931640625, -211.08755493164062))
        node.operation = 'MULTIPLY'
        node.inputs[1].default_value = 2.0
        nodes['Group Output'] = node = node_tree.nodes.new('NodeGroupOutput')
        node.name = 'Group Output'
        node.location = Vector((380.054931640625, -0.0))
        nodes['Group Input'] = node = node_tree.nodes.new('NodeGroupInput')
        node.name = 'Group Input'
        node.location = Vector((-381.2703552246094, 66.95870971679688))

        node_tree.links.new(nodes['Group Input'].outputs['UV'],nodes['Vector Math'].inputs[0])
        node_tree.links.new(nodes['Vector Math'].outputs['Vector'],nodes['Group Output'].inputs['UV'])
        node_tree.links.new(nodes['Vector Math.001'].outputs['Vector'],nodes['Vector Math'].inputs[1])
        node_tree.links.new(nodes['Group Input'].outputs['Scale'],nodes['Math.001'].inputs[0])
        node_tree.links.new(nodes['Math.001'].outputs['Value'],nodes['Vector Math.001'].inputs['Scale'])
        node_tree.links.new(nodes['Group Input'].outputs['Warp Map'],nodes['Vector Math.002'].inputs[0])
        node_tree.links.new(nodes['Vector Math.002'].outputs['Vector'],nodes['Vector Math.001'].inputs[0])
        return node_tree
