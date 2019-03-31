#!/usr/bin/python3

import struct

DEBUG = False


class ValkFile:
    filename = None
    def __init__(self, F, offset=None):
        self.F = F
        if offset is None:
            offset = F.tell()
        self.offset = offset
        self.inner_files = []
        self.read_meta()

    def seek(self, pos, relative=False):
        if DEBUG == 2:
            print("Seeking to 0x{:x}".format(pos), relative)
        if relative:
            return self.F.seek(pos, relative) - self.offset
        return self.F.seek(self.offset + pos) - self.offset

    def tell(self):
        return self.F.tell() - self.offset

    def read(self, size):
        if DEBUG == 2:
            print("Reading 0x{:x} bytes".format(size))
        return self.F.read(size)

    def read_and_unpack(self, size, unpack):
        oldpos = self.F.tell()
        value = struct.unpack(unpack, self.read(size))[0]
        return value

    def read_byte(self):
        return self.read_and_unpack(1, 'B')

    def read_byte_signed(self):
        return self.read_and_unpack(1, 'b')

    def read_word_le(self):
        return self.read_and_unpack(2, '<H')

    def read_word_be(self):
        return self.read_and_unpack(2, '>H')

    def read_word_le_signed(self):
        return self.read_and_unpack(2, '<h')

    def read_word_be_signed(self):
        return self.read_and_unpack(2, '>h')

    def read_long_le(self):
        return self.read_and_unpack(4, '<I')

    def read_long_be(self):
        return self.read_and_unpack(4, '>I')

    def read_float_le(self):
        return self.read_and_unpack(4, '<f')

    def read_float_be(self):
        return self.read_and_unpack(4, '>f')

    def read_half_float_be(self):
        # http://davidejones.com/blog/1413-python-precision-floating-point/
        word = self.read_and_unpack(2, '>h')
        sign = (word >> 15) & 0x0001
        exponent = (word >> 10) & 0x001f
        fraction = word & 0x03ff
        int32 = None
        if exponent == 0:
            if fraction == 0:
                int32 = sign << 31
            else:
                while not (fraction & 0x0400):
                    fraction = fraction << 1
                    exponent -= 1
                exponent += 1
                fraction &= 0x0400
        elif exponent == 31:
            if fraction == 0:
                int32 = (sign << 31) | 0x7f800000
            else:
                int32 = (sign << 31) | 0x7f800000 | (fraction << 13)
        if int32 is None:
            exponent = exponent + (127 -15)
            fraction = fraction << 13
            int32 = (sign << 31) | (exponent << 23) | fraction
        packed = struct.pack('I', int32)
        return struct.unpack('f', packed)[0]

    def read_string(self, encoding="ascii"):
        array = []
        byte = self.read(1)
        while byte != b'\x00':
            array.append(byte)
            byte = self.read(1)
        return b''.join(array).decode(encoding)

    def _print_header_hex(self):
        # For debugging and pattern-finding purposes
        unk1 = struct.unpack('>H', self.read(2))[0]
        unk2 = struct.unpack('>H', self.read(2))[0]
        print("{} {:08x} {:08x} {:04x} {:04x}".format(self.ftype, self.main_length, self.header_length, unk1, unk2), end="")
        if (self.header_length > 0x10):
            depth = self.read_long_le()
            next_file = self.read_long_le()
            print(" {:08x} {:08x}".format(depth, next_file), end="")
        header_pos = self.tell()
        if self.header_length > header_pos:
            rest = self.read(self.header_length - header_pos)
            print(" ".join(["{:02x}".format(b) for b in rest]), end="")
        print()

    def read_meta(self):
        self.seek(0)
        self.ftype = self.read(4).decode('ascii')
        if DEBUG:
            print("Creating", self.ftype)
        self.main_length = self.read_long_le()
        self.header_length = self.read_long_le()
        if DEBUG:
            self._print_header_hex()
        if self.ftype != 'EOFC' and self.header_length >= 0x20:
            self.seek(0xe)
            unk2 = struct.unpack('>H', self.read(2))[0]
            self.seek(0x14)
            next_offset = self.read_long_le()
        self.total_length = self.header_length + self.main_length

    def file_chain_done(self, running_length, max_length, inner_file):
        if max_length is None:
            done = inner_file.ftype == 'EOFC'
        else:
            done = running_length >= max_length
        return done

    def read_file_chain(self, start=0, max_length=None):
        running_length = 0
        inner_files = []
        if max_length is None:
            done = False
        else:
            done = running_length >= max_length
        while not done:
            chunk_begin = start + running_length
            inner_file = valk_factory(self, chunk_begin)
            inner_files.append(inner_file)
            running_length += inner_file.total_length
            done = self.file_chain_done(running_length, max_length, inner_file)
        return inner_files

    def container_func(self):
        if self.header_length < 0x20:
            return
        self.seek(0x14)
        chunk_length = self.read_long_le()
        chain_begin = self.header_length + chunk_length
        chain_length = self.main_length - chunk_length
        inner_files = self.read_file_chain(chain_begin, chain_length)
        for inner_file in inner_files:
            self.add_inner_file(inner_file)

    def find_inner_files(self):
        self.container_func()
        for inner_file in self.inner_files:
            inner_file.find_inner_files()

    def container_path(self, so_far=[]):
        if hasattr(self.F, 'container_path'):
            return self.F.container_path() + [self.ftype]
        else:
            return [self.filename, self.ftype]

    def add_inner_file(self, inner_file):
        self.inner_files.append(inner_file)
        if not hasattr(self, inner_file.ftype):
            setattr(self, inner_file.ftype, [])
        files_of_this_type = getattr(self, inner_file.ftype)
        files_of_this_type.append(inner_file)


class ValkUnknown(ValkFile):
    pass


class ValkIZCA(ValkFile):
    def container_func(self):
        self.seek(self.header_length)
        section_count = self.read_long_le()
        self.seek(4, True)
        section_list = []
        file_pointer_list = []
        for i in range(section_count):
            toc_pointer = self.read_long_le()
            toc_entry_count = self.read_long_le()
            section_list.append((i, toc_pointer, toc_entry_count))
        for (section, toc_pointer, toc_entry_count) in section_list:
            self.seek(toc_pointer)
            for i in range(toc_entry_count):
                file_pointer = self.read_long_le()
                file_pointer_list.append((section, file_pointer))
        for section, file_pointer in file_pointer_list:
            inner_file = valk_factory(self, file_pointer)
            # TODO: Put file in section?
            self.add_inner_file(inner_file)


class ValkMLX0(ValkFile):
    # Doesn't contain other files.
    # Found in IZCA files.
    # Gives information about type of model file?
    pass


class ValkCCOL(ValkFile):
    # Doesn't contain other files.
    pass


class ValkPJNT(ValkFile):
    # Doesn't contain other files.
    pass


class ValkPACT(ValkFile):
    # Doesn't contain other files.
    pass


class ValkEOFC(ValkFile):
    # Doesn't contain other files.
    pass


class ValkHSHP(ValkFile):
    # Standard container
    # Shape keys
    def read_data(self):
        assert len(self.KFSH) == 1
        kfsh = self.KFSH[0]
        kfsh.read_data()
        self.shape_keys = kfsh.shape_keys


class ValkKFSH(ValkFile):
    # Standard container
    # Shape keys
    def read_data(self):
        assert len(self.KFSS) == 1 and len(self.KFSG) == 1
        kfss = self.KFSS[0]
        kfsg = self.KFSG[0]
        kfss.read_data()
        self.shape_keys = kfss.shape_keys
        kfsg.vertex_type = kfss.vertex_type
        for shape_key in self.shape_keys:
            shape_key["vertices"] = kfsg.read_vertices(
                shape_key["kfmg_ptr"],
                shape_key["vertex_count"])


class ValkKFSS(ValkFile):
    # Doesn't contain other files.
    # Describes shape keys
    def read_toc(self):
        self.seek(self.header_length + 0x10)
        self.key_count = self.read_long_be()
        self.key_list_ptr = self.read_long_be()
        self.seek(self.header_length + 0x24)
        self.moreinfo_ptr = self.read_long_be()

    def read_moreinfo(self):
        self.seek(self.moreinfo_ptr)
        self.vertex_type = self.read_long_le()
        assert self.vertex_type in (0, 2, 3)

    def read_key_list(self):
        self.seek(self.key_list_ptr)
        self.shape_keys = []
        for i in range(self.key_count):
            self.read(6)
            shape_key = {
                #'hmdl_number?': self.read_word_be(), # Always 1
                'vertex_count': self.read_word_be(),
                'moreinfo_ptr': self.read_long_be(),
                'kfmg_ptr': self.read_long_be(),
                }
            self.shape_keys.append(shape_key)

    def read_data(self):
        self.read_toc()
        self.read_moreinfo()
        self.read_key_list()


class ValkKFSG(ValkFile):
    # Doesn't contain other files.
    # Holds shape key data
    def read_vertices(self, kfmg_ptr, count):
        vertices = []
        self.seek(self.header_length + kfmg_ptr)
        for i in range(count):
            vertex = {
                "translate_x": self.read_float_be(),
                "translate_y": self.read_float_be(),
                "translate_z": self.read_float_be(),
                }
            if self.vertex_type == 3:
                self.read(8) # Probably UV?
            vertices.append(vertex)
        return vertices


class ValkKFMD(ValkFile):
    # Standard container
    def read_data(self):
        assert len(self.KFMS) == 1 and len(self.KFMG) == 1
        kfms = self.KFMS[0]
        kfmg = self.KFMG[0]
        kfms.read_data()
        self.bones = kfms.bones
        self.materials = kfms.materials
        self.textures = kfms.textures
        if kfms.vc_game == 1:
            kfmg.face_ptr = 0
            kfmg.vertex_ptr = 0
        elif kfms.vc_game == 4:
            kfmg.seek(0x30)
            kfmg.face_ptr = kfmg.read_long_le()
            kfmg.read(4)
            kfmg.vertex_ptr = kfmg.read_long_le()
        kfmg.vc_game = kfms.vc_game
        self.meshes = kfms.meshes
        for mesh in self.meshes:
            if kfms.vc_game == 1:
                fmt = 0
            elif kfms.vc_game == 4:
                obj = mesh['object']
                fmt = obj['vertex_format']
            vertex_format = kfms.vertex_formats[fmt]
            mesh['faces'] = kfmg.read_faces(
                mesh['faces_first_word'],
                mesh['faces_word_count'],
                vertex_format)
            mesh['vertices'] = kfmg.read_vertices(
                mesh['first_vertex'],
                mesh['vertex_count'],
                vertex_format)


class ValkKFMS(ValkFile):
    # Doesn't contain other files.
    # Describes model armature, materials, meshes, and textures.
    def read_toc(self):
        self.seek(self.header_length)
        unk1 = self.read(4)
        if unk1[0] == 3:
            self.vc_game = 4
            self.endianness = '<'
        elif unk1[0] == 1:
            self.vc_game = 2
            self.endianness = '<'
        elif unk1[3] == 1:
            self.vc_game = 1
            self.endianness = '>'
        else:
            raise NotImplementedError('Unrecognized model version')
        if self.vc_game == 4:
            self.bone_count = self.read_long_le()
            self.deform_count = self.read_long_le() # Need to verify this
            self.read(4)
            self.model_height = self.read_float_le()
            self.material_count = self.read_long_le()
            self.object_count = self.read_long_le()
            self.mesh_count = self.read_long_le()
            self.read(4) # count?
            self.texture_count = self.read_long_le()
            self.vertex_format_count = self.read_long_le()
            self.vertex_formats = []
            self.read(4)
            self.read(16)
            self.bone_list_ptr = self.read_long_le() + 0x20
            self.read(4) # is this a 64-bit pointer?
            self.read_long_le() # pointer to extra per-bone data
            self.read(4) # is this a 64-bit pointer?
            self.bone_xform_list_ptr = self.read_long_le() + 0x20
            self.read(4) # is this a 64-bit pointer?
            self.material_list_ptr = self.read_long_le() + 0x20
            self.read(4) # is this a 64-bit pointer?
            self.object_list_ptr = self.read_long_le() + 0x20
            self.read(4) # is this a 64-bit pointer?
            self.mesh_list_ptr = self.read_long_le() + 0x20
            self.read(4) # is this a 64-bit pointer?
            self.read(4) # unknown pointer?
            self.read(4) # is this a 64-bit pointer?
            self.texture_list_ptr = self.read_long_le() + 0x20
            self.read(4) # is this a 64-bit pointer?
            self.mesh_info_ptr = self.read_long_le() + 0x20
            self.read(4) # is this a 64-bit pointer?
        else:
            self.bone_count = self.read_long_be()
            self.deform_count = self.read_long_be()
            self.read(4)
            self.model_height = self.read_float_be()
            self.bone_list_ptr = self.read_long_be()
            self.read_long_be() # pointer to extra per-bone data
            self.bone_xform_list_ptr = self.read_long_be()
            self.material_count = self.read_long_be()
            self.material_list_ptr = self.read_long_be()
            self.object_count = self.read_long_be()
            self.object_list_ptr = self.read_long_be()
            self.mesh_count = self.read_long_be()
            self.mesh_list_ptr = self.read_long_be()
            self.read(4)
            self.read(4)
            self.texture_count = self.read_long_be()
            self.texture_list_ptr = self.read_long_be()
            self.read_word_be() # These 3 words are counts that correspond
            self.read_word_be() # to the next group of 3 longs, which are
            self.read_word_be() # pointers. Purpose is unknown.
            self.read(2)
            self.read_long_be()
            self.read_long_be()
            self.read_long_be()
            self.read(4)
            self.vertex_format_count = 1
            self.vertex_formats = []
            self.mesh_info_ptr = self.read_long_be()

    def read_kfmg_info(self):
        self.seek(self.mesh_info_ptr)
        if self.vc_game == 4:
            for i in range(self.vertex_format_count):
                self.seek(self.mesh_info_ptr + 0x80 * i)
                self.read(4)
                self.vertex_formats.append({
                    'bytes_per_vertex': self.read_long_le(),
                    'face_ptr': self.read_long_le(),
                    'face_count': self.read_long_le(),
                    'vertex_ptr': self.read_long_le(),
                    'vertex_count': self.read_long_le(),
                    })
        else:
            self.read(4)
            self.vertex_formats.append({
                'bytes_per_vertex': self.read_long_be(),
                'face_ptr': self.read_long_be(),
                'face_count': self.read_long_be(),
                'vertex_ptr': self.read_long_be(),
                'vertex_count': self.read_long_be(),
                })
            self.read(4)
            self.read(4)

    def read_bone_list(self):
        self.seek(self.bone_list_ptr)
        self.bones = []
        for i in range(self.bone_count):
            bone = {}
            bone['ptr'] = self.tell()
            if self.vc_game == 1:
                self.read(4)
                bone['id'] = self.read_word_be()
                bone['parent_id'] = self.read_word_be()
                bone['dim1'] = self.read_float_be()
                bone['dim2'] = self.read_float_be()
                bone['parent_ptr'] = self.read_long_be()
                bone['fav_child_ptr'] = self.read_long_be()
                bone['unk_bone_ptr2'] = self.read_long_be()
                bone['bound_box_ptr'] = self.read_long_be()
                self.read(2)
                bone['object_count'] = self.read_word_be()
                self.read(4)
                bone['deform_count'] = self.read_word_be() # First bone only
                bone['is_deform'] = self.read_word_be()
                bone['object_ptr1'] = self.read_long_be()
                bone['object_ptr2'] = self.read_long_be()
                bone['object_ptr3'] = self.read_long_be()
                bone['deform_ids_ptr'] = self.read_long_be() # First bone only
                bone['deform_ptr'] = self.read_long_be()
                self.read(32)
            elif self.vc_game == 4:
                self.read(4)
                bone['id'] = self.read_word_le()
                bone['parent_id'] = self.read_word_le()
                bone['dim1'] = self.read_float_le()
                bone['dim2'] = self.read_float_le()
                bone['parent_ptr'] = self.read_long_le() + 0x20
                self.read(4) # 64-bit?
                bone['fav_child_ptr'] = self.read_long_le() + 0x20
                self.read(4) # 64-bit?
                bone['unk_bone_ptr2'] = self.read_long_le()
                self.read(4) # 64-bit?
                bone['bound_box_ptr'] = self.read_long_le()
                self.read(4) # 64-bit?
                self.read(4) # 0x20202020
                self.read(2)
                bone['object_count'] = self.read_word_le()
                self.read(4)
                bone['deform_count'] = self.read_word_le() # First bone only
                bone['is_deform'] = self.read_word_le()
                bone['object_ptr1'] = self.read_long_le()
                self.read(4) # 64-bit?
                bone['object_ptr2'] = self.read_long_le()
                self.read(4) # 64-bit?
                bone['object_ptr3'] = self.read_long_le()
                self.read(4) # 64-bit?
                self.read(8)
                bone['deform_ids_ptr'] = self.read_long_le() # First bone only
                self.read(4) # 64-bit?
                bone['deform_ptr'] = self.read_long_le()
                self.read(4) # 64-bit?
                self.read(48)
            self.bones.append(bone)

    def link_bones(self):
        for bone in self.bones:
            bone['parent'] = None
            bone['children'] = []
            bone['fav_child'] = None
            if bone['parent_id'] == bone['id']:
                continue
            parent_bone = self.bones[bone['parent_id']]
            bone['parent'] = parent_bone
            parent_bone['children'].append(bone)
            if bone['ptr'] == parent_bone['fav_child_ptr']:
                parent_bone['fav_child'] = bone

    def read_bone_xforms(self):
        self.seek(self.bone_xform_list_ptr)
        if self.vc_game == 1:
            read_float = self.read_float_be
        elif self.vc_game == 4:
            read_float = self.read_float_le
        for bone in self.bones:
            bone['location'] = [read_float() for x in range(3)]
            self.read(4)
            bone['rotation'] = [read_float() for x in range(4)]
            # Adjust order of quaternion elements
            bone['rotation'] = bone['rotation'][3:] + bone['rotation'][:3]
            bone['scale'] = [read_float() for x in range(3)]
            self.read(4)

    def read_bone_deforms(self):
        self.deform_bones = {}
        for bone in self.bones:
            if bone['deform_ptr'] == 0:
                # Should set deform_id equal to bone_id?
                continue
            if self.vc_game == 1:
                self.seek(bone['deform_ptr'])
                bone['matrix_ptr'] = self.read_long_be()
                bone['deform_id'] = self.read_long_be()
            elif self.vc_game == 4:
                self.seek(bone['deform_ptr'] + 0x20)
                bone['matrix_ptr'] = self.read_long_le() + 0x20
                self.read(4) # 64-bit?
                self.read(2) # unknown
                bone['deform_id'] = self.read_long_le()
                self.read(2) # unknown
            self.deform_bones[bone['deform_id']] = bone

    def read_bone_matrices(self):
        if self.vc_game == 1:
            read_float = self.read_float_be
        elif self.vc_game == 4:
            read_float = self.read_float_le
        for bone in self.bones:
            if 'matrix_ptr' not in bone:
                continue
            self.seek(bone['matrix_ptr'])
            bone['matrix_raw'] = (
                (read_float(), read_float(), read_float(), read_float()),
                (read_float(), read_float(), read_float(), read_float()),
                (read_float(), read_float(), read_float(), read_float()),
                (read_float(), read_float(), read_float(), read_float())
                )

    def read_material_list(self):
        self.materials = {}
        if self.vc_game == 1:
            item_length = 0xa0
        elif self.vc_game == 4:
            item_length = 0xf0
        for i in range(self.material_count):
            self.seek(self.material_list_ptr + i * item_length)
            material = {}
            material['id'] = i
            if self.vc_game == 1:
                material['ptr'] = self.tell()
                material['unk1'] = self.read(4)
                material['flags'] = self.read_long_be()
                material['use_normal'] = bool(material['flags'] & 0x12) # 0x10 and 0x2 both seem to indicate normal maps
                material['use_alpha'] = bool(material['flags'] & 0x40)
                material['use_backface_culling'] = bool(material['flags'] & 0x400)
                material['unk2'] = self.read(8)
                material['texture0_ptr'] = self.read_long_be()
                material['texture1_ptr'] = self.read_long_be()
            elif self.vc_game == 4:
                material['ptr'] = self.tell()
                material['flags1'] = self.read_long_le()
                transparency1 = material['flags1'] in [0x05, 0x21]
                material['texture_count'] = self.read_byte()
                material['flags2'] = self.read_word_be()
                transparency2 = material['flags2'] == 0x0201
                material['use_transparency'] = transparency1 or transparency2
                material['flags3'] = self.read_byte()
                material['use_backface_culling'] = material['flags3'] == 1
                self.read(0x78)
                material['texture0_ptr'] = self.read_long_le() + 0x20 if material['texture_count'] > 0 else False
                self.read(4) # 64-bit?
                material['texture1_ptr'] = self.read_long_le() + 0x20 if material['texture_count'] > 1 else False
                self.read(4) # 64-bit?
                material['texture2_ptr'] = self.read_long_le() + 0x20 if material['texture_count'] > 2 else False
                self.read(4) # 64-bit?
                material['texture3_ptr'] = self.read_long_le() + 0x20 if material['texture_count'] > 3 else False
                self.read(4) # 64-bit?
            self.materials[material['ptr']] = material

    def read_object_list(self):
        self.seek(self.object_list_ptr)
        self.objects = []
        for i in range(self.object_count):
            if self.vc_game == 1:
                object_row = {
                    'id': self.read_long_be(),
                    'u01': self.read_word_be(), # Has vertex groups?
                    'parent_bone_id': self.read_word_be(),
                    'material_ptr': self.read_long_be(),
                    'mesh_count': self.read_long_be(),
                    'mesh_list_ptr': self.read_long_be(),
                    'kfmg_vertex_offset': self.read_long_be(),
                    'vertex_count': self.read_word_be(),
                    'u02': self.read(6),
                    }
            elif self.vc_game == 4:
                object_row = {
                    'id': self.read_long_le(),
                    'u01': self.read_word_le(), # Has vertex groups?
                    'parent_bone_id': self.read_word_le(),
                    'material_ptr': self.read_long_le() + 0x20,
                    'u02': self.read_long_le(),
                    'kfmg_vertex_offset': self.read_long_le() + 0x20,
                    'vertex_count': self.read_word_le(),
                    'vertex_format': self.read_word_le(),
                    'mesh_count': self.read_long_le(),
                    'u03': self.read_long_le(),
                    'mesh_list_ptr': self.read_long_le() + 0x20,
                    'u04': self.read(4 * 7),
                    }
            self.objects.append(object_row)

    def read_mesh_list(self):
        self.meshes = []
        for obj in self.objects:
            self.seek(obj['mesh_list_ptr'])
            for i in range(obj['mesh_count']):
                if self.vc_game == 1:
                    mesh_row = {
                        'vertex_group_count': self.read_word_be(),
                        'u01': self.read_word_be(),
                        'u02': self.read_word_be(),
                        'vertex_count': self.read_word_be(),
                        'faces_word_count': self.read_word_be(),
                        'n01': self.read_long_be(),
                        'vertex_group_map_ptr': self.read_word_be(),
                        'first_vertex': self.read_long_be(),
                        'faces_first_word': self.read_long_be(),
                        'first_vertex_id': self.read_long_be(),
                        'n02': self.read_long_be(),
                        'object': obj,
                        }
                elif self.vc_game == 4:
                    mesh_row = {
                        'vertex_group_count': self.read_word_le(),
                        'u01': self.read_word_le(),
                        'u02': self.read_word_le(),
                        'vertex_count': self.read_word_le(),
                        'faces_word_count': self.read_word_le(),
                        'n01': self.read_word_le(),
                        'first_vertex': self.read_long_le(),
                        'faces_first_word': self.read_long_le(),
                        'first_vertex_id': self.read_long_le(),
                        'vertex_group_map_ptr': self.read_long_le() + 0x20,
                        'n02': self.read_long_le(),
                        'object': obj,
                        }
                self.meshes.append(mesh_row)

    def read_vertex_group_maps(self):
        vertex_group_map = {}
        for mesh in self.meshes:
            self.seek(mesh['vertex_group_map_ptr'])
            for i in range(mesh['vertex_group_count']):
                if self.vc_game == 1:
                    global_id = self.read_word_be()
                    local_id = self.read_word_be()
                elif self.vc_game == 4:
                    global_id = self.read_word_le()
                    local_id = self.read_word_le()
                vertex_group_map[local_id] = global_id
            mesh['vertex_group_map'] = vertex_group_map.copy()

    def read_texture_list(self):
        self.textures = {}
        if self.vc_game == 1:
            item_length = 0x40
            read_image = self.read_word_be
        elif self.vc_game == 4:
            item_length = 0x60
            read_image = self.read_word_le
        for i in range(self.texture_count):
            self.seek(self.texture_list_ptr + i * item_length)
            texture = {}
            texture['id'] = i
            texture['ptr'] = self.tell()
            self.read(4)
            texture['image'] = read_image()
            self.textures[texture['ptr']] = texture

    def link_materials(self):
        for material_ptr, material in self.materials.items():
            if material['texture0_ptr']:
                material['texture0'] = self.textures[material['texture0_ptr']]
            else:
                material['texture0'] = None
            if material['texture1_ptr']:
                material['texture1'] = self.textures[material['texture1_ptr']]
            else:
                material['texture1'] = None
            if self.vc_game == 4:
                if material['texture2_ptr']:
                    material['texture2'] = self.textures[material['texture2_ptr']]
                else:
                    material['texture2'] = None
                if material['texture3_ptr']:
                    material['texture3'] = self.textures[material['texture3_ptr']]
                else:
                    material['texture3'] = None

    def read_data(self):
        self.read_toc()
        self.read_kfmg_info()
        self.read_bone_list()
        self.link_bones()
        self.read_bone_xforms()
        self.read_bone_deforms()
        self.read_bone_matrices()
        self.read_material_list()
        self.read_object_list()
        self.read_mesh_list()
        self.read_vertex_group_maps()
        self.read_texture_list()
        self.link_materials()


class ValkKFMG(ValkFile):
    # Doesn't contain other files.
    # Holds mesh vertex and face data.
    def read_faces(self, first_word, word_count, vertex_format):
        fmt_face_offset = vertex_format['face_ptr']
        self.seek(self.header_length + self.face_ptr + fmt_face_offset + first_word * 2)
        end_ptr = self.tell() + word_count * 2
        start_direction = 1
        if self.vc_game == 1:
            read_vertex_id = self.read_word_be
        elif self.vc_game == 4:
            read_vertex_id = self.read_word_le
        v1 = read_vertex_id()
        v2 = read_vertex_id()
        face_direction = start_direction
        faces = []
        while self.tell() < end_ptr:
            v3 = read_vertex_id()
            if v3 == 0xffff:
                v1 = read_vertex_id()
                v2 = read_vertex_id()
                face_direction = start_direction
            else:
                face_direction *= -1
                if v1 != v2 and v2 != v3 and v3 != v1:
                    if face_direction > 0:
                        face = [v3, v2, v1, 0]
                    else:
                        face = [v3, v1, v2, 0]
                    faces.append(face)
                v1 = v2
                v2 = v3
        return faces

    def read_vertex(self, bytes_per_vertex):
        if bytes_per_vertex == 0x2c:
            # VC1
            vertex = {
                'location_x': self.read_float_be(),
                'location_y': self.read_float_be(),
                'location_z': self.read_float_be(),
                'unknown_1': self.read(4),
                'normal_x': self.read_half_float_be(),
                'normal_y': self.read_half_float_be(),
                'normal_z': self.read_half_float_be(),
                'unknown_2': self.read(2),
                'unknown_3': self.read(8),
                'u': self.read_half_float_be(),
                'v': self.read_half_float_be() * -1,
                'u2': self.read_half_float_be(),
                'v2': self.read_half_float_be() * -1,
                'unknown_4': self.read(4),
                }
        elif bytes_per_vertex == 0x30:
            # VC1
            vertex = {
                'location_x': self.read_float_be(),
                'location_y': self.read_float_be(),
                'location_z': self.read_float_be(),
                'vertex_group_1': self.read_byte(),
                'vertex_group_2': self.read_byte(),
                'vertex_group_3': self.read_byte(), # Junk?
                'vertex_group_4': self.read_byte(), # Junk?
                'vertex_group_weight_1': self.read_half_float_be(),
                'vertex_group_weight_2': self.read_half_float_be(),
                'unknown_1': self.read(4),
                'u': self.read_half_float_be(),
                'v': self.read_half_float_be() * -1,
                'u2': self.read_half_float_be(),
                'v2': self.read_half_float_be() * -1,
                'unknown_2': self.read(4),
                'normal_x': self.read_half_float_be(),
                'normal_y': self.read_half_float_be(),
                'normal_z': self.read_half_float_be(),
                'unknown_2': self.read(6),
                }
        elif bytes_per_vertex == 0x40:
            # VC4
            vertex = {
                'location_x': self.read_float_le(),
                'location_y': self.read_float_le(),
                'location_z': self.read_float_le(),
                'normal_x': self.read_float_le(),
                'normal_y': self.read_float_le(),
                'normal_z': self.read_float_le(),
                'unknown_1': self.read(4 * 4),
                'u': self.read_float_le(),
                'v': self.read_float_le() * -1,
                'vertex_group_weight_1': self.read_float_le(),
                'vertex_group_weight_2': self.read_float_le(),
                'vertex_group_weight_3': self.read_float_le(),
                'vertex_group_1': self.read_byte(),
                'vertex_group_2': self.read_byte(),
                'vertex_group_3': self.read_byte(),
                'vertex_group_4': self.read_byte(), # Junk?
                }
        elif bytes_per_vertex == 0x48:
            # VC4
            vertex = {
                'location_x': self.read_float_le(),
                'location_y': self.read_float_le(),
                'location_z': self.read_float_le(),
                'normal_x': self.read_float_le(),
                'normal_y': self.read_float_le(),
                'normal_z': self.read_float_le(),
                'unknown_1': self.read(4 * 4),
                'u': self.read_float_le(),
                'v': self.read_float_le() * -1,
                'u2': self.read_float_le(),
                'v2': self.read_float_le() * -1,
                'unknown_2': self.read(4 * 4),
                }
        elif bytes_per_vertex == 0x50:
            # VC1
            vertex = {
                'location_x': self.read_float_be(),
                'location_y': self.read_float_be(),
                'location_z': self.read_float_be(),
                'unknown_1': self.read(4 * 3),
                'unknown_2': self.read(4 * 2),
                'normal_x': self.read_float_be(),
                'normal_y': self.read_float_be(),
                'normal_z': self.read_float_be(),
                'unknown_3': self.read(4),
                'u': self.read_float_be(),
                'v': self.read_float_be() * -1,
                'u2': self.read_float_be(),
                'v2': self.read_float_be() * -1,
                'unknown_4': self.read(4 * 4),
                }
        return vertex

    def read_vertices(self, first_vertex, vertex_count, vertex_format):
        fmt_bytes_per_vertex = vertex_format['bytes_per_vertex']
        fmt_vertex_offset = vertex_format['vertex_ptr']
        self.seek(self.header_length + self.vertex_ptr + fmt_vertex_offset + first_vertex * fmt_bytes_per_vertex)
        vertices = []
        for i in range(vertex_count):
            vertex = self.read_vertex(fmt_bytes_per_vertex)
            vertices.append(vertex)
        return vertices


class ValkABDA(ValkFile):
    # Special container
    def determine_endianness(self):
        self.seek(0x20)
        endian_indicator = self.read(2)
        if endian_indicator == b'\x77\xa1':
            read_long = self.read_long_be
        elif endian_indicator == b'\x29\x55':
            read_long = self.read_long_le
        else:
            raise NotImplementedError("Unrecognized endianness indicator", endian_indicator)
        return read_long

    def container_func(self):
        read_long = self.determine_endianness()
        self.seek(0x24)
        file_count = read_long()
        pof0_ptr = read_long()
        self.seek(4, True)
        toc = []
        for i in range(file_count):
            # Always little-endian
            file_ptr = self.read_long_le()
            toc.append(file_ptr)
            self.seek(4, True)
        for file_ptr in toc:
            inner_file = valk_factory(self, file_ptr)
            self.add_inner_file(inner_file)
        if pof0_ptr:
            inner_file = valk_factory(self, pof0_ptr)
            self.add_inner_file(inner_file)


class ValkABRS(ValkFile):
    # Special container for HMDL, HTEX, and other files that seem
    # model-related
    def determine_endianness(self):
        self.seek(0x20)
        endian_indicator = self.read(2)
        if endian_indicator == b'\x77\xa1':
            read_long = self.read_long_be
        elif endian_indicator == b'\x29\x55':
            read_long = self.read_long_le
        else:
            raise NotImplementedError("Unrecognized endianness indicator", endian_indicator)
        return read_long

    def container_func(self):
        read_long = self.determine_endianness()
        self.seek(0x24)
        file_count = read_long()
        pof0_ptr = read_long()
        toc = []
        for i in range(file_count):
            # Always little-endian
            file_ptr = self.read_long_le()
            if file_ptr:
                toc.append(file_ptr)
            self.seek(4, True)
        for file_ptr in toc:
            inner_files = self.read_file_chain(file_ptr)
            for inner_file in inner_files:
                self.add_inner_file(inner_file)
        if pof0_ptr:
            inner_file = valk_factory(self, pof0_ptr)
            self.add_inner_file(inner_file)


class ValkCSBD(ValkFile):
    # Collection of @UTF chunks.
    pass


class ValkHTER(ValkFile):
    # Small files that contain lists of textures used by MXE files
    def read_toc(self):
        self.seek(self.header_length + 4)
        self.texture_pack_count = self.read_long_be()
        self.texture_pack_list_ptr = self.read_long_be()

    def read_texture_pack_list(self):
        self.seek(self.texture_pack_list_ptr)
        for i in range(self.texture_pack_count):
            pack = {
                "id_count": self.read_long_be(),
                "id_list_ptr": self.read_long_be(),
                }
            self.read(8)
            self.texture_packs.append(pack)
        for pack in self.texture_packs:
            self.seek(pack["id_list_ptr"])
            pack["htsf_ids"] = []
            for i in range(pack["id_count"]):
                htsf_id = self.read_long_be()
                pack["htsf_ids"].append(htsf_id)

    def read_data(self):
        self.texture_packs = []
        self.read_toc()
        self.read_texture_pack_list()



class ValkNAIS(ValkFile):
    # Relatively small files that contain lists of things. Related to AI?
    pass


class ValkEVSR(ValkFile):
    # Missing MSLP files if we turn off tails
    pass


class ValkMSCR(ValkFile):
    # One per EVSR file. Doesn't seem to contain other files.
    # Contains lists of things.
    pass


class ValkSFNT(ValkFile):
    def container_func(self):
        self.seek(0x20)
        file_count = self.read_long_le()
        toc_start = self.read_long_le()
        self.seek(toc_start)
        toc = []
        for i in range(file_count):
            file_ptr = self.read_long_le()
            toc.append(file_ptr)
        for file_ptr in toc:
            if file_ptr:
                inner_file = valk_factory(self, file_ptr)
                self.add_inner_file(inner_file)


class ValkMFNT(ValkFile):
    # Doesnt' contain other files.
    # Bitmap font data.
    def container_func(self):
        # Says chunk size is 0, but doesn't contain other files.
        pass


class ValkMFGT(ValkFile):
    # Doesnt' contain other files.
    pass


class ValkHFPR(ValkFile):
    # Doesnt' contain other files.
    pass


class ValkGHSL(ValkFile):
    # Doesnt' contain other files.
    pass


class ValkMTPA(ValkFile):
    # Doesn't contain other files.
    # Text.
    def container_func(self):
        if self.header_length < 0x20:
            return
        self.seek(0x14)
        chunk_length = self.read_long_le()
        chain_begin = self.header_length + chunk_length
        chain_length = self.main_length - chunk_length - self.header_length
        inner_files = self.read_file_chain(chain_begin, chain_length)
        for inner_file in inner_files:
            self.add_inner_file(inner_file)


class ValkVBHV(ValkFile):
    # Contains VBAC, VBCT, VBTI, and VBTB files in tail
    pass


class ValkNAGD(ValkFile):
    # Doesn't contain other files.
    pass


class ValkCLDC(ValkFile):
    # Contains LIPD files in tail.
    pass


class ValkHSPT(ValkFile):
    # Doesn't contain other files.
    # Appears to contain vector art.
    pass


class ValkHTSF(ValkFile):
    # Each HTSF contains a single DDS file.
    def container_func(self):
        self.seek(self.header_length)
        self.seek(0x20, True) # Always zeros?
        inner_file = valk_factory(self, self.tell())
        inner_file.total_length = self.main_length - 0x20
        self.add_inner_file(inner_file)


class ValkDDS(ValkFile):
    # Texture image. Not a Valkyria-specific file format.
    def read_meta(self):
        self.ftype = 'DDS'

    def container_func(self):
        pass

    def read_data(self):
        self.seek(0)
        self.data = self.read(self.total_length)


class ValkKFCA(ValkFile):
    # Doesn't contain other files.
    # Very small. Camera position?
    pass


class ValkKFCM(ValkFile):
    # Doesn't contain other files.
    pass


class ValkKFMA(ValkFile):
    # Doesn't contain other files.
    pass


class ValkKFMH(ValkFile):
    # Doesn't contain other files.
    # Small.
    pass


class ValkKFMI(ValkFile):
    # Doesn't contain other files.
    pass


class ValkKFMO(ValkFile):
    # Doesn't contain other files.
    # Specifies an armature pose or animation
    def read_toc(self):
        self.seek(self.header_length + 4)
        self.bone_count = self.read_long_be()
        self.read(2 * 4)
        self.frame_count = self.read_float_be()
        assert (self.frame_count - int(self.frame_count)) == 0
        assert self.frame_count > 0
        self.frame_count = int(self.frame_count)
        self.frames_per_second = self.read_float_be()
        unknown0 = self.read_float_be()
        assert self.frames_per_second == 59.939998626708984
        assert unknown0 in [1.997999906539917, 1.9999979734420776]
        self.bone_list_ptr = self.read_long_be()

    def read_bone_list(self):
        self.bones = []
        self.seek(self.bone_list_ptr)
        for i in range(self.bone_count):
            bone = {
                'flags': self.read_word_be(),
                'zero_0': self.read_word_be(),
                'zero_1': self.read_long_be(),
                'anim_ptr': self.read_long_be(),
                'xform_ptr': self.read_long_be(),
                }
            assert bone["flags"] in [0xef00, 0xefe0, 0xe0e0, 0xe000, 0x0f00, 0x00e0, 0x0000]
            assert bone["zero_0"] == 0 and bone["zero_1"] == 0
            self.bones.append(bone)

    def read_coord_animation(self, bone, ptr_key):
        if not (ptr_key in bone and bone[ptr_key]):
            return [0] * self.frame_count
        self.seek(bone[ptr_key])
        one = self.read_byte()
        data_type = self.read_byte()
        bits_after_decimal = self.read_byte()
        zero_0 = self.read_byte()
        zero_1 = self.read_long_be()
        zero_2 = self.read_long_be()
        frames_ptr = self.read_long_be()
        #difference = 0
        #if self.prev_frames_ptr:
        #    difference = frames_ptr - self.prev_frames_ptr
        assert one == 1
        assert zero_0 == 0 and zero_1 == 0 and zero_2 == 0
        assert data_type in [1, 2, 3]
        assert bits_after_decimal in [0x00, 0x01, 0x02, 0x06, 0x07, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]
        self.seek(frames_ptr)
        frames = []
        for i in range(self.frame_count + 1):
            if data_type == 1:
                frames.append(self.read_float_be())
            elif data_type == 2:
                frames.append(self.read_word_be_signed() / 2**bits_after_decimal)
            elif data_type == 3:
                frames.append(self.read_byte_signed() / 2**bits_after_decimal)
        #print(ptr_key, end=" ")
        #print("\t{:02x} {:02x} {:08x} {:04x} {:04x}".format(data_type, bits_after_decimal, frames_ptr, difference, int(self.frame_count * 2) + 2), end=" ")
        #print(frames)
        #self.prev_frames_ptr = frames_ptr
        return frames

    def read_bones(self):
        #self.prev_frames_ptr = None
        i = -1
        for bone in self.bones:
            i += 1
            if not bone["xform_ptr"]:
                continue
            self.seek(bone["xform_ptr"])
            if bone["flags"] & 0xe000:
                bone["location"] = (self.read_float_be(), self.read_float_be(), self.read_float_be())
            else:
                bone["location"] = None
            if bone["flags"] & 0x0f00:
                bone["rotation"] = (self.read_float_be(), self.read_float_be(), self.read_float_be(), self.read_float_be())
                # Convert XYZW to WXYZ
                bone["rotation"] = bone["rotation"][3:] + bone["rotation"][:3]
                magnitude = sum([x**2 for x in bone["rotation"]])**(1/2)
                is_normal = abs(1.0 - magnitude) < 0.00001
            else:
                bone["rotation"] = None
                magnitude = None
                is_normal = None
            if bone["flags"] & 0x00e0:
                bone["scale"] = (self.read_float_be(), self.read_float_be(), self.read_float_be())
            else:
                bone["scale"] = None
            #print("{:02x} Pose:".format(i), bone["location"], bone["rotation"], bone["scale"], bone["anim_ptr"], is_normal, magnitude)
            if bone["anim_ptr"]:
                self.seek(bone["anim_ptr"])
                if bone["flags"] & 0xe000:
                    bone["location_ptr_x"] = self.read_long_be()
                    bone["location_ptr_y"] = self.read_long_be()
                    bone["location_ptr_z"] = self.read_long_be()
                if bone["flags"] & 0x0f00:
                    bone["rotation_ptr_x"] = self.read_long_be()
                    bone["rotation_ptr_y"] = self.read_long_be()
                    bone["rotation_ptr_z"] = self.read_long_be()
                    bone["rotation_ptr_w"] = self.read_long_be()
                if bone["flags"] & 0x00e0:
                    bone["scale_ptr_x"] = self.read_long_be()
                    bone["scale_ptr_y"] = self.read_long_be()
                    bone["scale_ptr_z"] = self.read_long_be()
            location_frames_x = self.read_coord_animation(bone, "location_ptr_x")
            location_frames_y = self.read_coord_animation(bone, "location_ptr_y")
            location_frames_z = self.read_coord_animation(bone, "location_ptr_z")
            rotation_frames_w = self.read_coord_animation(bone, "rotation_ptr_w")
            rotation_frames_x = self.read_coord_animation(bone, "rotation_ptr_x")
            rotation_frames_y = self.read_coord_animation(bone, "rotation_ptr_y")
            rotation_frames_z = self.read_coord_animation(bone, "rotation_ptr_z")
            scale_frames_x = self.read_coord_animation(bone, "scale_ptr_x")
            scale_frames_y = self.read_coord_animation(bone, "scale_ptr_y")
            scale_frames_z = self.read_coord_animation(bone, "scale_ptr_z")
            bone["location_frames"] = list(zip(location_frames_x, location_frames_y, location_frames_z))
            bone["rotation_frames"] = list(zip(rotation_frames_w, rotation_frames_x, rotation_frames_y, rotation_frames_z))
            bone["scale_frames"] = list(zip(scale_frames_x, scale_frames_y, scale_frames_z))

    def read_data(self):
        self.read_toc()
        self.read_bone_list()
        self.read_bones()


class ValkKFSC(ValkFile):
    # Doesn't contain other files.
    # Small.
    pass


class ValkKFSM(ValkFile):
    # Doesn't contain other files.
    # Small.
    pass


class ValkKSPR(ValkFile):
    # Doesn't contain other files.
    pass


class ValkMXEC(ValkFile):
    # Doesn't contain other files.
    # Points to HTX and HMD files.
    model_types = [
            "EnEventDecor",
            "EnHeightField",
            "EnSky",
            "EnTalkEventMap",
            "EnTalkEventObj",
            "EnUvScroll",
            "SlgEnBorder",
            "SlgEnBreakableStructure",
            "SlgEnBufferConvex",
            "SlgEnBufferHeightMap",
            "SlgEnBunkerCannon",
            "SlgEnCatwalk",
            "SlgEnChainBreakdown",
            "SlgEnCombatConvex",
            "SlgEnCombatHeightMap",
            "SlgEnGrass",
            "SlgEnGregoal",
            "SlgEnLift",
            "SlgEnLorry",
            "SlgEnMarmot1st",
            "SlgEnProduceBorder",
            "SlgEnPropeller",
            "SlgEnReplaceModel",
            "SlgEnSearchLight",
            "SlgEnSteepleBarrier",
            "SlgEnStronghold",
            "SlgEnTerrain",
            "SlgEnTriggerBase",
            "SlgEnWireFence",
            "VlTree",
            "VlWindmill",
            ]
    def __init__(self, F, offset=None):
        self.PRINT_FILES = False
        self.PRINT_PARAMS = False
        self.PRINT_MODEL_PARAMS = False
        self.PRINT_MODEL_FILES = False
        super().__init__(F, offset)

    def read_toc(self):
        self.seek(self.header_length + 0x4)
        self.param_block_ptr = self.read_long_be()
        self.model_block_ptr = self.read_long_be()
        self.file_block_ptr = self.read_long_be()
        if self.param_block_ptr:
            self.seek(self.param_block_ptr + 0x4)
            self.param_count = self.read_long_be()
            self.param_list_ptr = self.read_long_be()
        if self.model_block_ptr:
            self.seek(self.model_block_ptr + 0x4)
            self.model_count = self.read_long_be()
            self.model_list_ptr = self.read_long_be()
        if self.file_block_ptr:
            self.seek(self.file_block_ptr + 0x4)
            self.file_count = self.read_long_be()
            self.file_list_ptr = self.read_long_be()

    def read_parameter_list(self):
        from subprocess import check_output
        if not hasattr(self, "param_list_ptr"):
            return
        self.seek(self.param_list_ptr)
        rows = []
        for i in range(self.param_count):
            row = {
                "id": self.read_long_be(),
                "name_ptr": self.read_long_be(),
                "data_length": self.read_long_be(),
                "data_ptr": self.read_long_be(),
                }
            rows.append(row)
        for row in rows:
            self.seek(row["name_ptr"])
            row["name"] = self.read_string(encoding = "shift_jis_2004")
            if self.PRINT_PARAMS:
                print(row)
                print(check_output(["xxd", "-s", str(row["data_ptr"] + self.offset), "-l", str(row["data_length"]), self.F.filename]).decode("ascii"))
        self.parameters = rows

    def read_model_list(self):
        if not hasattr(self, "model_list_ptr"):
            return
        self.seek(self.model_list_ptr)
        rows = []
        for i in range(self.model_count):
            row = {
                "unk1": self.read_long_be(),
                "name_ptr": self.read_long_be(),
                "param_count": self.read_long_be(),
                "param_list_ptr": self.read_long_be(),
                }
            self.seek(0x30, True) # Always zero?
            rows.append(row)
        for row in rows:
            self.seek(row["name_ptr"])
            row["name"] = self.read_string(encoding = "shift_jis_2004")
        self.models = rows

    def read_file_list(self):
        if not hasattr(self, "file_list_ptr"):
            return
        self.seek(self.file_list_ptr)
        file_rows = []
        for i in range(self.file_count):
            row = {
                "is_inside": self.read_long_be(),
                "id": self.read_long_be(),
                "path_ptr": self.read_long_be(),
                "filename_ptr": self.read_long_be(),
                "type": self.read_long_be(),
                "htr_index": self.read_long_be(),
                "unk1": self.read(0xc),
                "mmr_index": self.read_long_be(),
                "unk2": self.read(0x18),
                }
            # 0 = Not inside another file
            # 0x100 = Is inside merge.htx, indexed by merge.htr
            # 0x200 = Is inside mmf, indexed by mmr
            assert row["is_inside"] in [0, 0x100, 0x200]
            # 0x1 = hmd
            # 0x2 = htx
            # 0x3 = hmt
            # 0x6 = mcl
            # 0x8 = mlx
            # 0x9 = abr
            # 0xa = abd
            # 0xc = SpeedTree
            # 0x14 = pvs
            # 0x15 = htx (merge)
            # 0x16 = htr
            # 0x18 = mmf
            # 0x19 = mmr
            assert row["type"] in [0x1, 0x2, 0x3, 0x6, 0x8, 0x9, 0xa, 0xc, 0x14, 0x15, 0x16, 0x18, 0x19]
            assert row["unk2"] == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
            del row["unk2"]
            file_rows.append(row)
            if row["type"] == 0x15:
                self.merge_htx_file = row
            elif row["type"] == 0x16:
                self.htr_file = row
            elif row["type"] == 0x18:
                self.mmf_file = row
        for row in file_rows:
            self.seek(row["filename_ptr"])
            row["filename"] = self.read_string()
            if self.PRINT_FILES:
                print(row)
        self.files = file_rows

    def read_model_param_ids(self):
        from subprocess import check_output
        for model in self.models:
            self.seek(model["param_list_ptr"])
            param_refs = []
            for i in range(model["param_count"]):
                param_text_ptr = self.read_long_be()
                param_id_count = self.read_long_be()
                param_id_ptr = self.read_long_be()
                param_refs.append([param_text_ptr, param_id_count, param_id_ptr])
                self.read(4)
            param_groups = []
            for param_text_ptr, param_id_count, param_id_ptr in param_refs:
                param_group = {}
                self.seek(param_text_ptr)
                param_group["text"] = self.read_string("shift_jis_2004")
                param_group["param_ids"] = []
                self.seek(param_id_ptr)
                if self.PRINT_MODEL_PARAMS:
                    print(param_group["text"])
                for i in range(param_id_count):
                    param_id = self.read_long_be()
                    if self.PRINT_MODEL_PARAMS:
                        print(self.parameters[param_id])
                        print(check_output(["xxd", "-s", str(self.parameters[param_id]["data_ptr"] + self.offset), "-l", str(self.parameters[param_id]["data_length"]), self.F.filename]).decode("ascii"))
                    param_group["param_ids"].append(param_id)
                param_groups.append(param_group)
            model["param_groups"] = param_groups

    def read_model_files(self):
        # (common)
        #   0x40 X Y Z floats
        #   0x60 scaleX? scaleY? scaleZ? floats
        #   0x74 model file id
        #   0x84 texture file id
        # EnTalkEventMap
        # EnTalkEventObj
        # EnEventDecor
        # EnHeightField
        # EnSky
        # EnUvScroll (water)
        # SlgEnCombatHeightMap ?
        # SlgEnBufferHeightMap ?
        # SlgEnBreakableStructure
        # SlgEnCombatConvex
        # SlgEnUnitPlacementPoint[0]
        # VlTree
        #   0x114 VlTree hst file id
        #   0x11c VlTree htx file id
        #   0x124 VlTree _cmp.htx file id
        # EnCEffect
        #   0x114 EnCEffect abr
        #   0x11c EnCEffect abd
        # SlgEnStronghold
        #   0x114-0x144 hmt, cvd, htx, hmd
        def file_exists(filename):
            import os
            path = os.path.dirname(self.F.filename)
            filepath = os.path.join(path, filename)
            exists = False
            for trypath in [filepath, filepath.upper()]:
                try:
                    F = open(trypath, "rb")
                    F.close()
                    exists = True
                except IOError:
                    pass
            return exists
        for model in self.models:
            for group in model["param_groups"]:
                if group["text"] in self.model_types:
                    assert len(group["param_ids"]) == 1
                    param_id = group["param_ids"][0]
                    param = self.parameters[param_id]
                    self.seek(param["data_ptr"] + 0x40)
                    model["location_x"] = self.read_float_be()
                    model["location_y"] = self.read_float_be()
                    model["location_z"] = self.read_float_be()
                    self.seek(param["data_ptr"] + 0x50)
                    model["rotation_x"] = self.read_float_be()
                    model["rotation_y"] = self.read_float_be()
                    model["rotation_z"] = self.read_float_be()
                    self.seek(param["data_ptr"] + 0x60)
                    model["scale_x"] = self.read_float_be()
                    model["scale_y"] = self.read_float_be()
                    model["scale_z"] = self.read_float_be()
                    self.seek(param["data_ptr"] + 0x74)
                    model["model_file_id"] = self.read_long_be()
                    self.seek(param["data_ptr"] + 0x84)
                    model["texture_file_id"] = self.read_long_be()
                    if model["model_file_id"] != 0xffffffff and model["texture_file_id"] != 0xffffffff:
                        model["model_file"] = self.files[model["model_file_id"]]
                        model["texture_file"] = self.files[model["texture_file_id"]]
                        if self.PRINT_MODEL_FILES:
                            model_exists = file_exists(model["model_file"]["filename"])
                            texture_exists = file_exists(model["texture_file"]["filename"])
                            print("Model:", model["model_file"]["filename"], model_exists)
                            print("Texture:", model["texture_file"]["filename"], model_exists)

    def read_data(self):
        self.parameters = []
        self.models = []
        self.files = []
        self.read_toc()
        self.read_parameter_list()
        self.read_model_list()
        self.read_file_list()
        self.read_model_param_ids()
        self.read_model_files()


class ValkMXMC(ValkFile):
    # Doesn't contain other files.
    pass


class ValkMXMI(ValkFile):
    # Contains files immediately after header
    # Contains various model-related files in tail.
    def file_chain_done(self, running_length, max_length, inner_file):
        done1 = inner_file.ftype == 'EOFC'
        done2 = running_length >= max_length
        return done1 or done2

    def read_file_chain(self, start=0, max_length=None):
        running_length = 0
        inner_files = []
        inner_file = None
        if max_length is None:
            done = False
        else:
            done = running_length >= max_length
        while not done:
            if inner_file is not None and inner_file.ftype == 'HSPT':
                # MXMI files that contain HSPT files are weird. There's
                # data after the HSPT's EOFC chunk, but that data doesn't
                # have a header. So we'll read the HSPT's EOFC, then
                # skip past the weird data to the next EOFC chunk.
                chunk_begin = start + running_length
                inner_file = valk_factory(self, chunk_begin)
                inner_files.append(inner_file)
                old_running_length = running_length
                running_length = max_length - 0x20
            chunk_begin = start + running_length
            inner_file = valk_factory(self, chunk_begin)
            inner_files.append(inner_file)
            running_length += inner_file.total_length
            done = self.file_chain_done(running_length, max_length, inner_file)
        return inner_files

    def container_func(self):
        chain_begin = self.header_length
        self.seek(0x14)
        chain_length = self.read_long_le()
        inner_files = self.read_file_chain(chain_begin, chain_length)
        for inner_file in inner_files:
            self.add_inner_file(inner_file)
        super().container_func()
    pass


class ValkMXMH(ValkFile):
    # Doesn't contain other files.
    # Small. Contains a filename.
    def read_data(self):
        self.seek(self.header_length)
        path_ptr = self.read_long_be()
        filename_ptr = self.read_long_be()
        assert 0x10 <= path_ptr < filename_ptr < self.main_length
        self.seek(self.header_length + filename_ptr)
        self.filename = self.read_string()


class ValkMXPC(ValkFile):
    # Doesn't contain other files.
    pass


class ValkMXPT(ValkFile):
    # Doesn't contain other files.
    pass


class ValkMXTF(ValkFile):
    # Doesn't contain other files.
    # Small. Index of textures?
    pass


class ValkMXTL(ValkFile):
    # Doesn't contain other files.
    # Texture list
    def read_data(self):
        self.seek(self.header_length)
        hmdl_count = self.read_long_le()
        self.texture_lists = []
        for i in range(hmdl_count):
            section_start = self.tell()
            unk1 = self.read_long_le()
            assert unk1 == 8
            htsf_count = self.read_long_le()
            textures = []
            for j in range(htsf_count):
                filename_offset = self.read_long_le()
                unk2 = self.read_long_le()
                assert unk2 == 0
                htsf_number = self.read_long_le()
                bookmark = self.tell()
                self.seek(section_start + filename_offset)
                filename = self.read_string()
                self.seek(bookmark)
                textures.append((htsf_number, filename))
            self.texture_lists.append(textures)


class ValkCMDC(ValkFile):
    # Contains CMND files in tail.
    pass


class ValkENRS(ValkFile):
    # Found in tails
    pass


class ValkPOF0(ValkFile):
    # Found in tails
    pass


class ValkCCRS(ValkFile):
    # Found in tails
    pass


class ValkMTXS(ValkFile):
    # Found in tails
    pass


class ValkLIPD(ValkFile):
    # Doesn't contain other files.
    pass


class ValkMSLP(ValkFile):
    # Contains files immediately after header
    def container_func(self):
        chain_begin = self.header_length
        chain_length = self.main_length
        inner_files = self.read_file_chain(chain_begin, chain_length)
        for inner_file in inner_files:
            self.add_inner_file(inner_file)


class ValkCMND(ValkFile):
    # Contains CVMD files in tail.
    pass


class ValkCVMD(ValkFile):
    # Doesn't contain other files.
    pass


class ValkVBAC(ValkFile):
    # Doesn't contain other files.
    # Small
    pass


class ValkVBCT(ValkFile):
    # Contains VBTI files in tail.
    pass


class ValkVBTI(ValkFile):
    # Contains VBTB files in tail.
    # Small
    pass


class ValkVBTB(ValkFile):
    # Doesn't contain other files.
    # Small
    pass


class ValkABDT(ValkFile):
    # Doesn't contain other files.
    pass


class ValkVBBT(ValkFile):
    # Contains VBTI files in tail.
    pass


class ValkVBCE(ValkFile):
    # Doesn't contain other files.
    pass


class ValkVSSP(ValkFile):
    # Contains MSCR file immediately after header.
    def container_func(self):
        chain_begin = self.header_length
        chain_length = self.main_length
        inner_files = self.read_file_chain(chain_begin, chain_length)
        for inner_file in inner_files:
            self.add_inner_file(inner_file)
    pass


class ValkVSPA(ValkFile):
    # Contains MSCR file immediately after header.
    def container_func(self):
        chain_begin = self.header_length
        chain_length = self.main_length
        inner_files = self.read_file_chain(chain_begin, chain_length)
        for inner_file in inner_files:
            self.add_inner_file(inner_file)
    pass


class ValkVSAS(ValkFile):
    # Contains MSCR file immediately after header.
    def container_func(self):
        chain_begin = self.header_length
        chain_length = self.main_length
        inner_files = self.read_file_chain(chain_begin, chain_length)
        for inner_file in inner_files:
            self.add_inner_file(inner_file)
    pass


class ValkVSCO(ValkFile):
    # Contains MSCR file immediately after header.
    def container_func(self):
        chain_begin = self.header_length
        chain_length = self.main_length
        inner_files = self.read_file_chain(chain_begin, chain_length)
        for inner_file in inner_files:
            self.add_inner_file(inner_file)
    pass


class ValkMXEN(ValkFile):
    # Standard container
    pass


class ValkMXMF(ValkFile):
    # Standard container
    # Contains MXMB file, which contains MXMI files, which contain a single
    # HMDL file with an MXMH filename specifier.
    def read_data(self):
        self.named_models = {}
        assert len(self.MXMB) == 1
        for mxmi in self.MXMB[0].MXMI:
            if not hasattr(mxmi, "HMDL"):
                continue
            assert len(mxmi.HMDL) == 1
            assert len(mxmi.MXMH) == 1
            hmdl = mxmi.HMDL[0]
            mxmh = mxmi.MXMH[0]
            mxmh.read_data()
            hmdl.filename = mxmh.filename
            self.named_models[mxmh.filename] = hmdl


class ValkMXMB(ValkFile):
    # Standard container
    pass


class ValkMXMR(ValkFile):
    # Standard container
    pass


class ValkHMDL(ValkFile):
    # Standard container
    # There are 3235.
    pass


class ValkHTEX(ValkFile):
    # Standard container
    pass


class ValkHSPR(ValkFile):
    # Standard container
    pass


class ValkHCAM(ValkFile):
    # Standard container
    pass


class ValkHCMT(ValkFile):
    # Standard container
    pass


class ValkHSCR(ValkFile):
    # Standard container
    pass


class ValkHSCM(ValkFile):
    # Standard container
    pass


class ValkHMOT(ValkFile):
    # Standard container
    # Contains a KFMO file
    def read_data(self):
        assert len(self.KFMO) == 1
        kfmo = self.KFMO[0]
        kfmo.read_data()
        self.bones = kfmo.bones


class ValkMXPV(ValkFile):
    # Standard container
    pass


class ValkHMMT(ValkFile):
    # Standard container
    pass


class ValkHMRP(ValkFile):
    # Standard container
    pass


class ValkKFML(ValkFile):
    # Standard container
    pass


class ValkKFMM(ValkFile):
    # Standard container
    pass


class Valk2MIG(ValkFile):
    # Unknown block found in Valkyria Chornicles 2 models

    def read_meta(self):
        self.seek(0)
        self.ftype = self.read(4).decode('ascii')
        if DEBUG:
            print("Creating", self.ftype)
        self.seek(0x14)
        self.main_length = self.read_long_le()
        self.header_length = self.read_long_le()
        self.total_length = self.header_length + self.main_length


class Valk4POF1(ValkFile):
    # Unknown block found in Valkyria Chornicles 4 models
    pass


class Valk4WIRS(ValkFile):
    # Unknown block found in Valkyria Chornicles 4 models
    pass


class Valk4MBHV(ValkFile):
    # Unknown block found in Valkyria Chornicles 4
    pass


class Valk4MBMP(ValkFile):
    # Unknown block found in Valkyria Chornicles 4
    pass


class Valk4MBHD(ValkFile):
    # Unknown block found in Valkyria Chornicles 4
    pass


class Valk4MBMD(ValkFile):
    # Unknown block found in Valkyria Chornicles 4
    pass


class Valk4SDPK(ValkFile):
    # Unknown block found in Valkyria Chornicles 4
    pass


file_types = {
    'IZCA': ValkIZCA,
    'MLX0': ValkMLX0,
    'HSHP': ValkHSHP,
    'KFSH': ValkKFSH,
    'KFSS': ValkKFSS,
    'KFSG': ValkKFSG,
    'EOFC': ValkEOFC,
    'MXEN': ValkMXEN,
    'NAIS': ValkNAIS,
    'HSPR': ValkHSPR,
    'HTEX': ValkHTEX,
    'ABDA': ValkABDA,
    'ABRS': ValkABRS,
    'HCAM': ValkHCAM,
    'HCMT': ValkHCMT,
    'HSCR': ValkHSCR,
    'HSCM': ValkHSCM,
    'HMDL': ValkHMDL,
    'KFMD': ValkKFMD,
    'KFMS': ValkKFMS,
    'KFMG': ValkKFMG,
    'KFML': ValkKFML,
    'HMOT': ValkHMOT,
    'EVSR': ValkEVSR,
    'MSCR': ValkMSCR,
    'SFNT': ValkSFNT,
    'GHSL': ValkGHSL,
    'MTPA': ValkMTPA,
    'MXMF': ValkMXMF,
    'MXMB': ValkMXMF,
    'MXMR': ValkMXMR,
    'HTER': ValkHTER,
    'VBHV': ValkVBHV,
    'CCOL': ValkCCOL,
    'PJNT': ValkPJNT,
    'PACT': ValkPACT,
    'NAGD': ValkNAGD,
    'MXPV': ValkMXPV,
    'HMMT': ValkHMMT,
    'HMRP': ValkHMRP,
    'CSBD': ValkCSBD,
    'CLDC': ValkCLDC,
    'HTSF': ValkHTSF,
    'KFCA': ValkKFCA,
    'KFCM': ValkKFCM,
    'KFMA': ValkKFMA,
    'KFMH': ValkKFMH,
    'KFMI': ValkKFMI,
    'KFMM': ValkKFMM,
    'KFMO': ValkKFMO,
    'KFSC': ValkKFSC,
    'KFSM': ValkKFSM,
    'KSPR': ValkKSPR,
    'MXEC': ValkMXEC,
    'MXMC': ValkMXMC,
    'MXMI': ValkMXMI,
    'MXMH': ValkMXMH,
    'CMDC': ValkCMDC,
    'MXPC': ValkMXPC,
    'MXPT': ValkMXPT,
    'MXTF': ValkMXTF,
    'MXTL': ValkMXTL,
    'DDS ': ValkDDS,
    'ENRS': ValkENRS,
    'POF0': ValkPOF0,
    'CCRS': ValkCCRS,
    'MTXS': ValkMTXS,
    'HSPT': ValkHSPT,
    'LIPD': ValkLIPD,
    'MSLP': ValkMSLP,
    'CMND': ValkCMND,
    'CVMD': ValkCVMD,
    'VBAC': ValkVBAC,
    'VBCT': ValkVBCT,
    'VBTI': ValkVBTI,
    'VBTB': ValkVBTB,
    'ABDT': ValkABDT,
    'MFNT': ValkMFNT,
    'MFGT': ValkMFGT,
    'HFPR': ValkHFPR,
    'VBBT': ValkVBBT,
    'VBCE': ValkVBCE,
    'VSSP': ValkVSSP,
    'VSPA': ValkVSPA,
    'VSAS': ValkVSAS,
    'VSCO': ValkVSCO,
    'MIG.': Valk2MIG,
    'POF1': Valk4POF1,
    'WIRS': Valk4WIRS,
    'MBHV': Valk4MBHV,
    'MBMP': Valk4MBMP,
    'MBHD': Valk4MBHD,
    'MBMD': Valk4MBMD,
    'SDPK': Valk4SDPK,
    }

def valk_factory(F, offset=0, parent=None):
    F.seek(offset)
    ftype = F.read(4).decode('ascii')
    if ftype == '':
        return None
    F.seek(-4, True)
    if DEBUG:
        if hasattr(F, 'ftype'):
            print("Attempting to create", ftype, "file found in {} at 0x{:x}".format(F.ftype, F.tell()))
    fclass = file_types.get(ftype, ValkUnknown)
    if fclass == ValkUnknown:
        raise NotImplementedError("File type {} not recognized.".format(repr(ftype)))
    return fclass(F, offset)

def valk_open(filename):
    files = []
    F = open(filename, 'rb')
    FV = valk_factory(F)
    FV.filename = filename
    files.append(FV)
    while FV.ftype != 'EOFC':
        if FV.ftype != 'MTPA':
            F.seek(FV.offset + FV.total_length)
        else:
            F.seek(FV.offset + FV.main_length)
        FV = valk_factory(F, F.tell())
        if FV:
            FV.filename = filename
            files.append(FV)
        else:
            break
    return files
