#!/usr/bin/python3

import struct
import collections

DEBUG = False

def read_tuple(fn, count=3):
    return tuple(fn() for i in range(count))

def is_zero_bytes(data):
    return data == b'\x00' * len(data)

def hash_fnv1a64(string):
    hashval = 0xcbf29ce484222325
    for c in string:
        hashval = ((hashval ^ ord(c)) * 0x100000001B3) & 0xffffffffffffffff
    return hashval


class InfoDict(dict):
    "Dictionary that allows access to contents as fields."
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def update_fields(self, **kwargs):
        "data.update_fields(foo=1, bar=2 ...)"
        self.update(kwargs)


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
        if relative:
            if DEBUG >= 2:
                print("Seeking relative {:#x}".format(pos))
            return self.F.seek(pos, relative) - self.offset
        else:
            if DEBUG >= 2:
                print("Seeking to {:#x} + {:#x}".format(pos, self.offset))
            return self.F.seek(self.offset + pos) - self.offset

    def follow_ptr(self, pointer):
        if getattr(self, 'vc_game', 1) == 4:
            pointer += self.header_length
        return self.seek(pointer)

    def tell_ptr(self):
        pointer = self.tell()
        if getattr(self, 'vc_game', 1) == 4:
            pointer -= self.header_length
        return pointer

    def tell(self):
        return self.F.tell() - self.offset

    def read(self, size):
        if DEBUG >= 3:
            print("Reading {:#x} bytes".format(size))
        return self.F.read(size)

    def read_and_unpack(self, size, unpack):
        oldpos = self.F.tell()
        value = struct.unpack(unpack, self.read(size))[0]
        return value

    def read_byte(self):
        return self.read_and_unpack(1, 'B')

    def read_byte_signed(self):
        return self.read_and_unpack(1, 'b')

    def read_byte_factor(self):
        return self.read_byte() / 255

    def read_word_le(self):
        return self.read_and_unpack(2, '<H')

    def read_word_be(self):
        return self.read_and_unpack(2, '>H')

    def read_word_auto(self):
        return self.read_and_unpack(2, self.endianness + 'H')

    def read_word_le_signed(self):
        return self.read_and_unpack(2, '<h')

    def read_word_be_signed(self):
        return self.read_and_unpack(2, '>h')

    def read_long_le(self):
        return self.read_and_unpack(4, '<I')

    def read_long_be(self):
        return self.read_and_unpack(4, '>I')

    def read_long_auto(self):
        return self.read_and_unpack(4, self.endianness + 'I')

    def read_float_le(self):
        return self.read_and_unpack(4, '<f')

    def read_float_be(self):
        return self.read_and_unpack(4, '>f')

    def read_float_auto(self):
        return self.read_and_unpack(4, self.endianness + 'f')

    def read_half_float_le(self):
        return self.__decode_half_float(self.read_and_unpack(2, '<h'))

    def read_half_float_be(self):
        return self.__decode_half_float(self.read_and_unpack(2, '>h'))

    def read_half_float_auto(self):
        return self.__decode_half_float(self.read_and_unpack(2, self.endianness + 'h'))

    def __decode_half_float(self, word):
        # http://davidejones.com/blog/1413-python-precision-floating-point/
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

    def read_long_long_le(self):
        # https://youtu.be/sZsJyCyGBSI
        return self.read_and_unpack(8, '<Q')

    def read_string(self, encoding="ascii"):
        array = []
        byte = self.read(1)
        while byte != b'\x00':
            array.append(byte)
            byte = self.read(1)
        return b''.join(array).decode(encoding)

    def read_string_buffer(self, size, encoding="ascii"):
        strbuf = self.read(size)
        index = strbuf.find(b'\x00')
        if index >= 0:
            strbuf = strbuf[0:index]
        return strbuf.decode(encoding)

    def _print_header_hex(self):
        # For debugging and pattern-finding purposes
        print("{} {:08x} {:08x} {:04x} {:04x}".format(self.ftype, self.main_length, self.header_length, *self.header_unk), end="")
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
        self.header_unk = read_tuple(self.read_word_le, 2)
        if DEBUG:
            self._print_header_hex()
        if self.ftype != 'EOFC' and self.header_length >= 0x20:
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

    def container_offset(self):
        if hasattr(self.F, 'container_offset'):
            return self.F.container_offset() + self.offset
        else:
            return self.offset

    def print_location(self, message, *args):
        path = ':'.join(self.container_path())
        pos = self.tell() + self.container_offset()
        print('%s:%x: %s' % (path, pos, message.format(*args)))

    def skip_zero(self, size):
        data = self.read(size)
        if not is_zero_bytes(data):
            self.print_location('expected zero bytes, found: {}', data)

    def check_unknown_fields(self, name, data, pattern):
        "Checks that fields in data have expected values. Use to detect when unknown fields change."
        # Fields named 'pad...' should be zero bytes
        if isinstance(data, dict):
            for k, curval in list(data.items()):
                if k.startswith('pad'):
                    assert k not in pattern
                    if is_zero_bytes(curval):
                        del data[k]
                    else:
                        self.print_location('unexpected nonzero {} [{}]: value is {}\n{}', name, k, curval, data)
        # Check values using pattern
        for k, v in pattern.items():
            if isinstance(v, set):
                if data[k] in v:
                    continue
            elif callable(v):
                if v(data[k]):
                    continue
            else:
                if data[k] == v:
                    continue
            self.print_location('unexpected {} [{}]: value is {}\n{}', name, k, data[k], data)

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
        kfsg.vc_game = kfss.vc_game
        self.shape_keys = kfss.shape_keys
        kfsg.vertex_formats = kfss.vertex_formats
        kfsg.read_data()
        for shape_key in self.shape_keys:
            vertfmt = kfsg.vertex_formats[shape_key['vertex_format']]
            slice_start = shape_key['vertex_offset']
            slice_end = slice_start + shape_key['vertex_count']
            if kfss.vc_game == 1:
                shape_key['vertices'] = vertfmt['vertices'][slice_start:slice_end]
            else:
                shape_key['vertices'] = vertfmt['vertices']
            shape_key['vc_game'] = kfsg.vc_game


VFormatField = collections.namedtuple('VFormatField', ['offset', 'info_type', 'data_type', 'value_count'])

class VC4VertexFormatMixin:
    VERT_LOCATION = (0x1, 0xa, 0x3)
    VERT_WEIGHTS = (0x2, 0xa, 0x3)
    VERT_GROUPS = (0x3, 0x1, 0x4)
    VERT_NORMAL= (0x4, 0xa, 0x3)
    VERT_TANGENT = (0x5, 0xa, 0x3)
    VERT_UV1 = (0x7, 0xa, 0x2)
    VERT_UV2 = (0x8, 0xa, 0x2)
    VERT_UV3 = (0x9, 0xa, 0x2)
    VERT_UV4 = (0xa, 0xa, 0x2)
    VERT_UV5 = (0xb, 0xa, 0x2)
    VERT_COLOR = (0xf, 0xa, 0x4)

    def read_vformat_spec(self):
        entry = InfoDict(
            id_hash     = self.read_long_long_le(),
            field_count = self.read_long_le(),
            total_bytes = self.read_long_le(),
            name        = self.read_string_buffer(32),
            field_ptr   = self.read_long_long_le(),
            padding     = self.read(8 + 16),
        )
        return entry

    def read_vformat_field(self):
        entry = VFormatField(
            self.read_long_le(), # bytes before this item in the struct
            self.read_long_le(), # info type: Position, Normal, Color, UV, Weights
            self.read_long_le(), # data type: 0x1 = Byte, 0xa = Float
            self.read_long_le(), # value count: 2 (u,v) or 3 (x,y,z) or 4 (r,g,b,a)
        )
        self.check_unknown_fields('vformat field', entry, {
            2: {1, 10},
            3: {1, 2, 3, 4},
        })
        return entry

    def read_vformat_fields(self, vformat):
        self.follow_ptr(vformat.field_ptr)
        vformat.fields = [ self.read_vformat_field() for i in range(vformat.field_count) ]


class ValkKFSS(ValkFile, VC4VertexFormatMixin):
    # Doesn't contain other files.
    # Describes shape keys
    def read_toc(self):
        self.seek(self.header_length)
        version = self.read_long_le()
        if version == 0:
            self.vc_game = 1
        elif version == 0x3:
            self.vc_game = 4
        if self.vc_game == 1:
            self.seek(self.header_length + 0x10)
            self.key_count = self.read_long_be()
            self.key_list_ptr = self.read_long_be()
            self.seek(self.header_length + 0x20)
            self.vertex_format_count = self.read_long_be()
            self.vertex_format_ptr = self.read_long_be()
        elif self.vc_game == 4:
            self.seek(self.header_length + 0x10)
            self.group_count = self.read_long_le() # Don't know what these are for yet
            self.key_count = self.read_long_le()
            self.read(4)
            self.vertex_format_count = self.read_long_le()
            self.seek(self.header_length + 0x30)
            self.group_list_ptr = self.read_long_long_le()
            self.key_list_ptr = self.read_long_le()
            self.seek(self.header_length + 0x48)
            self.vertex_format_ptr = self.read_long_le()

    def read_vertex_formats(self):
        if self.vc_game == 1:
            self.follow_ptr(self.vertex_format_ptr + 0x8)
            vertfmt = {
                'kfsg_ptr': 0,
                'bytes_per_vertex': self.read_long_be(),
                'unk': self.read(0x8),
                'vertex_count': self.read_long_be(),
            }
            del(vertfmt['unk'])
            assert vertfmt['bytes_per_vertex'] in [0x0, 0xc, 0x14]
            self.vertex_formats = [vertfmt]
        elif self.vc_game == 4:
            item_length = 0x80
            self.vertex_formats = []
            for i in range(self.vertex_format_count):
                item_start = self.vertex_format_ptr + item_length * i
                self.follow_ptr(item_start)
                vertfmt = {
                    'kfmg_ptr': self.read_long_le(),
                    'kfsg_ptr': self.read_long_le(),
                    'vertex_count': self.read_long_le(),
                    'skip_count': self.read_long_le(),
                    'skip_ptr': self.read_long_le(),
                    'unk': self.read(0x10),
                    'bytes_per_vertex': self.read_long_le(),
                }
                del(vertfmt['unk'])
                struct_def_row_count = self.read_long_le()
                if struct_def_row_count:
                    vertfmt['struct_def'] = []
                self.read(4)
                struct_def_ptr = self.read_long_long_le()
                self.follow_ptr(struct_def_ptr)
                for j in range(struct_def_row_count):
                    info_type = self.read_long_le() # Position, UV
                    data_type = self.read_long_le() # 0x1 = Byte, 0xa = Float
                    unknown = self.read_long_le() # related to UV somehow?
                    offset = self.read_long_le() # bytes before this item in the struct
                    value_count = self.read_long_le() # 2 (u,v) or 3 (x,y,z)
                    struct_row = (offset, (info_type, data_type, value_count))
                    vertfmt['struct_def'].append(struct_row)
                if vertfmt['bytes_per_vertex'] not in [0x0, 0x8, 0xc, 0x14, 0x18, 0x1c, 0x28]:
                    self.print_location('unexpected bytes_per_vertex: {}', vertfmt)
                self.vertex_formats.append(vertfmt)

    def read_key_list(self):
        if self.vc_game == 1:
            item_length = 0x10
        elif self.vc_game == 4:
            item_length = 0x20
        self.shape_keys = []
        for i in range(self.key_count):
            self.follow_ptr(self.key_list_ptr + i * item_length)
            if self.vc_game == 1:
                self.read(6)
                shape_key = {
                    #'hmdl_number?': self.read_word_be(), # Always 1
                    'vertex_format': 0,
                    'vertex_count': self.read_word_be(),
                    }
                t3ptr = self.read_long_be()
                self.follow_ptr(t3ptr)
                shape_key['vertex_offset'] = self.read_long_be()
            elif self.vc_game == 4:
                self.read(2)
                vertex_format = self.read_word_le()
                self.read(0xc)
                t3ptr = self.read_word_le()
                self.follow_ptr(t3ptr)
                shape_key = {
                    'vertex_format': vertex_format,
                    'vertex_offset': self.read_long_le(),
                    'vertex_count': self.read_long_le(),
                    }
            self.shape_keys.append(shape_key)

    def read_skip_lists(self):
        if self.vc_game == 1:
            for vertfmt in self.vertex_formats:
                vertfmt['skip_keep_list'] = [(0, vertfmt['vertex_count'])]
        elif self.vc_game == 4:
            for vertfmt in self.vertex_formats:
                self.follow_ptr(vertfmt['skip_ptr'])
                vertfmt['skip_keep_list'] = []
                for i in range(vertfmt['skip_count']):
                    skip = self.read_long_le()
                    keep = self.read_long_le()
                    vertfmt['skip_keep_list'].append((skip, keep))

    def read_data(self):
        self.read_toc()
        self.read_vertex_formats()
        self.read_key_list()
        self.read_skip_lists()


class ValkKFSG(ValkFile, VC4VertexFormatMixin):
    # Doesn't contain other files.
    # Holds shape key data
    def read_data(self):
        if self.vc_game == 1:
            read_float = self.read_float_be
        elif self.vc_game == 4:
            read_float = self.read_float_le
        for vertfmt in self.vertex_formats:
            vertices = []
            self.seek(self.header_length + vertfmt['kfsg_ptr'])
            for skip, keep in vertfmt['skip_keep_list']:
                for i in range(skip):
                    vertex = {
                        "translate": (0.0, 0.0, 0.0),
                        "translate_u": 0.0,
                        "translate_v": 0.0,
                        }
                    vertices.append(vertex)
                for i in range(keep):
                    if self.vc_game == 1:
                        vertex = {
                            "translate": read_tuple(read_float),
                            }
                        if vertfmt['bytes_per_vertex'] > 0xc:
                            self.read(vertfmt['bytes_per_vertex'] - 0xc)
                    elif self.vc_game == 4:
                        struct = vertfmt['struct_def']
                        read_float = self.read_float_le
                        vertex = {}
                        for offset, element in struct:
                            if element == self.VERT_LOCATION:
                                vertex['translate'] = read_tuple(read_float)
                            elif element == self.VERT_UV1:
                                vertex['translate_u'] = read_float()
                                vertex['translate_v'] = -1 * read_float()
                            elif element == self.VERT_NORMAL:
                                vertex['translate_normal_x'] = read_float()
                                vertex['translate_normal_y'] = read_float()
                                vertex['translate_normal_z'] = read_float()
                            else:
                                raise NotImplementedError('Unsupported shape key vertex info: {}'.format(element))
                    vertices.append(vertex)
            vertfmt['vertices'] = vertices


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
        kfmg.endianness = kfms.endianness
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


def _make_shader_traits():
    tex2 = 'texture1' # blend with tex alpha and LayerVSParam[1].w
    tex3 = 'texture2' # blend with tex alpha and LayerVSParam[2].w
    bump2 = 'normal1'
    vc2 = 'texblend_vcolor1' # blend textures with vcolor1.xy instead of alpha
    multex = 'texblend_mul' # multiply textures together
    notexalpha = 'texblend_noalpha' # don't use alpha for blending textures
    a = 'alpha'
    ag = 'ag' # some alpha stuff
    ls = 'alphablend' # hlsl shaders identical; maybe 'layer sorted' for alpha blend?
    ns = 'no_shadow' # no crosshatching
    lshadow = 'light_shadow' # light crosshatching
    unlit = 'unlit' # use emission to ignore lighting
    spec = 'specular'
    # Valkyria Chronicles 1 shader table
    vc1_traits = {
        0: {}, # sometimes used, silence warning
        0x100: {ls,}, # VL_UFPaintLS
        0x101: {ls,tex2}, # VL_UFPaintLSTex2
        0x103: {ls,bump2}, # VL_UFPaintLSTex2Bump2
        0x104: {ls,tex2,tex3}, # VL_UFPaintLSTex3
        0x106: {ls,tex2,vc2}, # VL_UFPaintLSTex2Vertex
        0x107: {ls,tex2,tex3,vc2}, # VL_UFPaintLSTex3Vertex
        0x124: {}, # VL_UFPaint
        0x125: {tex2}, # VL_UFPaintTex2
        0x127: {bump2}, # VL_UFPaintTex2Bump2
        0x128: {tex2,tex3}, # VL_UFPaintTex3
        0x130: {tex2,vc2}, # VL_UFPaintTex2Vertex
        0x131: {tex2,tex3,vc2}, # VL_UFPaintTex3Vertex
        0x148: {ls,a}, # VL_FPaintLS
        0x149: {ls,a,tex2}, # VL_FPaintLSTex2
        0x151: {ls,a,bump2}, # VL_FPaintLSTex2Bump2
        0x152: {ls,a,tex2,tex3}, # VL_FPaintLSTex3
        0x154: {ls,a,tex2,vc2}, # VL_FPaintLSTex2Vertex
        0x155: {ls,a,tex2,tex3,vc2}, # VL_FPaintLSTex3Vertex
        0x172: {a}, # VL_FPaint
        0x173: {a,tex2}, # VL_FPaintTex2
        0x175: {a,bump2}, # VL_FPaintTex2Bump2
        0x176: {a,tex2,tex3}, # VL_FPaintTex3
        0x178: {a,tex2,vc2}, # VL_FPaintTex2Vertex
        0x179: {a,tex2,tex3,vc2}, # VL_FPaintTex3Vertex
        0x300: {a,unlit}, # VL_Sky
        0x301: {a,unlit,tex2}, # VL_SkyTex2
        0x303: {a,bump2}, # VL_SkyTex2Bump2
        0x304: {a,unlit,tex2,tex3}, # VL_SkyTex3
        0x305: {'water',bump2}, # VL_WaterSurfaceNoReflect
        0x306: {'water',spec,bump2}, # VL_WaterSurface
        0x312: {a,bump2}, # vl_bullet_mark
        0x324: {ns,a}, # VL_FPaintNS
        0x325: {ns,a,tex2}, # VL_FPaintNSTex2
        0x327: {ns,a,bump2}, # VL_FPaintNSTex2Bump2
        0x328: {ns,a,tex2,tex3}, # VL_FPaintNSTex3
        0x348: {'hair',a}, # VL_TransHair
        0x349: {'hair',a,tex2}, # VL_TransHairTex2
        0x351: {'hair',a,bump2,spec}, # VL_TransHairTex2Bump2
        0x352: {'hair',a,tex3}, # VL_TransHairTex3
        0x353: {'hair',a,bump2,spec,lshadow}, # VL_TransHairLightShadowTex2Bump2
        0x400: {ns}, # VL_Default
        0x401: {ns,tex2}, # VL_DefaultTex2
        0x403: {ns,bump2}, # VL_DefaultTex2Bump2
        0x404: {ns,tex3}, # VL_DefaultTex3
        0x500: {ls,lshadow}, # VL_UFPaintLSNSLightShadow
        0x501: {ls,lshadow,tex2}, # VL_UFPaintLSNSLightShadowTex2
        0x503: {ls,lshadow,bump2}, # VL_UFPaintLSNSLightShadowTex2Bump2
        0x504: {ls,lshadow,tex2,tex3}, # VL_UFPaintLSNSLightShadowTex3
        0x505: {ls,lshadow,bump2,spec,'ring'}, # VL_UFPaintLSNSLightShadowTex2Bump2ring
        0x508: {ls,lshadow,bump2,'objspace'}, # VL_UFPaintLSNSLightShadowTex2Bump2objspace
        0x548: {a,ls,lshadow}, # VL_FPaintLSNSLightShadow
        0x549: {a,ls,lshadow,tex2}, # VL_FPaintLSNSLightShadowTex2
        0x551: {a,ls,lshadow,bump2}, # VL_FPaintLSNSLightShadowTex2Bump2
        0x552: {a,ls,lshadow,tex2,tex3}, # VL_FPaintLSNSLightShadowTex3
        0x553: {a,ls,lshadow,bump2,spec,'ring'}, # VL_FPaintLSNSLightShadowTex2Bump2ring
        0x556: {a,ls,lshadow,bump2,'objspace'}, # VL_FPaintLSNSLightShadowTex2Bump2objspace
        0x601: {a,multex,notexalpha,ls,tex2}, # vl_alphacompo_mul_lstex2
        0x604: {a,multex,notexalpha,ls,tex2,tex3}, # vl_alphacompo_mul_lstex3
        0x625: {a,multex,notexalpha,tex2}, # vl_alphacompo_mul_tex2
        0x628: {a,multex,notexalpha,tex2,tex3}, # vl_alphacompo_mul_tex3
        0x648: {a,ag,ls}, # VL_FPaintLSAG
        0x649: {a,ag,ls,tex2}, # VL_FPaintLSAGTex2
        0x651: {a,ag,ls,bump2}, # VL_FPaintLSAGTex2Bump2
        0x652: {a,ag,ls,tex2,tex3}, # VL_FPaintLSAGTex3
        0x672: {a,ag}, # VL_FPaintAG
        0x673: {a,ag,tex2}, # VL_FPaintAGTex2
        0x675: {a,ag,bump2}, # VL_FPaintAGTex2Bump2
        0x676: {a,ag,tex2,tex3}, # VL_FPaintAGTex3
        #0x748: {}, # VL_FPaintLSLightMap
        #0x749: {}, # VL_FPaintLSLightMapTex2
        #0x751: {}, # VL_FPaintLSLightMapTex2Bump2
        #0x752: {}, # VL_FPaintLSLightMapTex3
        #0x754: {}, # VL_FPaintLSLightMapTex2Vertex
        #0x755: {}, # VL_FPaintLSLightMapTex3Vertex
        #0x772: {}, # VL_FPaintLightMap
        #0x773: {}, # VL_FPaintLightMapTex2
        #0x775: {}, # VL_FPaintLightMapTex2Bump2
        #0x776: {}, # VL_FPaintLightMapTex3
        #0x778: {}, # VL_FPaintLightMapTex2Vertex
        #0x779: {}, # VL_FPaintLightMapTex3Vertex
        0x1148: {a,ls}, # VL_FPaintLSPrim
        0x1172: {a}, # VL_FPaintPrim
        0x1248: {a,ls,ns}, # VL_FPaintLSNSPrim
        0x1272: {a,ns}, # VL_FPaintNSPrim
        0x1400: {a,ns}, # VL_DefaultPrim
        0x1648: {a,ls,ag}, # VL_FPaintLSAGPrim
        0x1672: {a,ag}, # VL_FPaintAGPrim
    }
    # Valkyria Chronicles 4 shader table (None entries provide just the name)
    tex4 = 'texture3' # blend with tex alpha and LayerVSParam[3].w
    tex5 = 'texture4' # blend with tex alpha and LayerVSParam[4].w
    char = 'character'
    map = 'map'
    pm = 'pm' # 'parallax mapping'? might be reflection related?
    uv0 = 'all_uv0'
    ice = [map,'ice',bump2,'warp_uv0_tex1']
    ice_t3 = [*ice,'tex1_uv2',tex3,tex4,'texblend_vcolor0_alpha_ex3']
    vc4_traits = {
        'cri_mana_h264_miyako': None,
        'cri_mana_sofdec_prime_alpha_miyako': None,
        'cri_mana_sofdec_prime_miyako': None,
        # ---------------------------------------------
        'fen_char_t1': {char},
        'fen_char_t1_eye': {char,uv0,tex2,tex3,tex4,'texblend_mul3',lshadow},
        'fen_char_t1_eye_pm': {char,uv0,tex2,tex3,tex4,'texblend_mul3',lshadow,pm},
        'fen_char_t1_hair': {char,uv0,'specular_alpha1',lshadow},
        'fen_char_t1_hair_pm': {char,uv0,'specular_alpha1',lshadow,pm},
        'fen_char_t1_pm': {char,pm},
        'fen_char_t1_skin': {char,lshadow},
        'fen_char_t1_skin_pm': {char,lshadow,pm},
        'fen_char_t2': {char,tex2,'texblend_dec1y'},
        'fen_char_t2_decare': {char,tex2}, # variations in the use of cDecareParam
        'fen_char_t2_decare_add': {char,tex2,'texblend_dec1w'}, # applies after lighting
        'fen_char_t2_pm': {char,tex2,'texblend_dec1y',pm},
        'fen_char_t2_skin': {char,tex2,lshadow},
        'fen_char_t3': {char,tex2,tex3},
        'fen_char_t3_pm': {char,tex2,tex3,pm},
        'fen_char_t4': {char,tex2,tex3,tex4},
        'fen_char_t4_pm': {char,tex2,tex3,tex4,pm},
        # ---------------------------------------------
        'fen_height_base_t1': None,
        'fen_height_base_t1_c4': None,
        'fen_height_base_t1_pt': None,
        'fen_height_base_t1_pt_c4': None,
        'fen_height_base_t1t': None,
        'fen_height_base_t1t_pt': None,
        'fen_height_base_t2': None,
        'fen_height_base_t2_c4': None,
        'fen_height_base_t2_pt': None,
        'fen_height_base_t2_pt_c4': None,
        'fen_height_base_t2t': None,
        'fen_height_base_t2t_pt': None,
        'fen_height_base_t3': None,
        'fen_height_base_t3_c4': None,
        'fen_height_base_t3_pt': None,
        'fen_height_base_t3_pt_c4': None,
        'fen_height_base_t4': None,
        'fen_height_base_t4_pt': None,
        'fen_height_base_t5': None,
        'fen_height_base_t5_pt': None,
        'fen_height_map_t1': None,
        'fen_height_map_t1_c4': None,
        'fen_height_map_t1_pt': None,
        'fen_height_map_t1_pt_c4': None,
        'fen_height_map_t1t': None,
        'fen_height_map_t1t_pt': None,
        'fen_height_map_t2': None,
        'fen_height_map_t2_c4': None,
        'fen_height_map_t2_pt': None,
        'fen_height_map_t2_pt_c4': None,
        'fen_height_map_t2t': None,
        'fen_height_map_t2t_pt': None,
        'fen_height_map_t3': None,
        'fen_height_map_t3_c4': None,
        'fen_height_map_t3_pt': None,
        'fen_height_map_t3_pt_c4': None,
        'fen_height_map_t4': None,
        'fen_height_map_t4_pt': None,
        'fen_height_map_t5': None,
        'fen_height_map_t5_pt': None,
        # ---------------------------------------------
        'fen_map_ice_t2': {*ice},
        'fen_map_ice_t2_dm': {*ice,'displace2'},
        'fen_map_ice_t2_dm_ws': {*ice,'displace2',spec,'roughness00'}, # "water surface"
        'fen_map_ice_t2_lm': {*ice,tex3,'texblend_lm2'},
        'fen_map_ice_t2_lm_ta': {*ice,tex3,tex4,'texblend_lm3'},
        'fen_map_ice_t2_lm_va': {*ice,tex3,'texblend_vcolor0_alpha2',tex4,'texblend_lm3'},
        'fen_map_ice_t2_parallax': {*ice,'parallax'},
        'fen_map_ice_t2_parallax_lm': {*ice,tex3,'texblend_lm2','parallax'},
        'fen_map_ice_t2_parallax_lm_ta': {*ice,tex3,tex4,'texblend_lm3','parallax'},
        'fen_map_ice_t2_parallax_lm_va': {*ice,tex3,'texblend_vcolor0_alpha2',tex4,'texblend_lm3','parallax'},
        'fen_map_ice_t2_parallax_pm': {*ice,'parallax',pm},
        'fen_map_ice_t2_parallax_ta': {*ice,tex3,'parallax'},
        'fen_map_ice_t2_parallax_ta_pm': {*ice,tex3,'parallax',pm},
        'fen_map_ice_t2_parallax_va': {*ice,tex3,'texblend_vcolor0_alpha2','parallax'},
        'fen_map_ice_t2_parallax_va_pm': {*ice,tex3,'texblend_vcolor0_alpha2','parallax',pm},
        'fen_map_ice_t2_pm': {*ice,pm},
        'fen_map_ice_t2_ta': {*ice,tex3},
        'fen_map_ice_t2_ta_pm': {*ice,tex3,pm},
        'fen_map_ice_t2_va': {*ice,tex3,'texblend_vcolor0_alpha2'},
        'fen_map_ice_t2_va_pm': {*ice,tex3,'texblend_vcolor0_alpha2',pm},
        'fen_map_ice_t2_ws': {*ice,spec,'roughness00'}, # "water surface"
        'fen_map_ice_t3': {*ice_t3},
        'fen_map_ice_t3_dm': {*ice_t3,'displace4','tex4_uv1'},
        'fen_map_ice_t3_lm': {*ice_t3,tex5,'texblend_lm4'},
        'fen_map_ice_t3_parallax': {*ice_t3,'parallax'},
        'fen_map_ice_t3_parallax_dm': {*ice_t3,'displace4','tex4_uv1','parallax'},
        'fen_map_ice_t3_parallax_dm_ws': {*ice_t3,'displace4','tex4_uv1',spec,'roughness00','parallax'},
        'fen_map_ice_t3_parallax_lm': {*ice_t3,tex5,'texblend_lm4','parallax'},
        # ---------------------------------------------
        'fen_map_sky_t1': {map,unlit},
        'fen_map_sky_t2': {map,unlit,tex2},
        'fen_map_t1': {map},
        'fen_map_t1_lm': {map,tex2,'texblend_lm1'},
        'fen_map_t1_lm_pm': {map,tex2,'texblend_lm1',pm},
        'fen_map_t1_pm': {map,pm},
        'fen_map_t2': {map,tex2,'texblend_vcolor0_alpha1'},
        'fen_map_t2_decare': {map,tex2,'texblend_vcolor0_alpha_ex1'},
        'fen_map_t2_decare_lm': {map,tex2,'texblend_vcolor0_alpha_ex1',tex3,'texblend_lm2'},
        'fen_map_t2_decare_lm_pm': {map,tex2,'texblend_vcolor0_alpha_ex1',tex3,'texblend_lm2',pm},
        'fen_map_t2_decare_pm': {map,tex2,'texblend_vcolor0_alpha_ex1',pm},
        'fen_map_t2_footprint': {map,tex2,'texblend_footprint'},
        'fen_map_t2_footprint_lm': {map,tex2,'texblend_footprint',tex3,'texblend_lm2'},
        'fen_map_t2_footprint_lm_pm': {map,tex2,'texblend_footprint',tex3,'texblend_lm2',pm},
        'fen_map_t2_footprint_pm': {map,tex2,'texblend_footprint',pm},
        'fen_map_t2_lm': {map,tex2,'texblend_vcolor0_alpha1',tex3,'texblend_lm2'},
        'fen_map_t2_lm_pm': {map,tex2,'texblend_vcolor0_alpha1',tex3,'texblend_lm2',pm},
        'fen_map_t2_pm': {map,tex2,'texblend_vcolor0_alpha1',pm},
        'fen_map_t3_decare': {map,tex2,tex3},
        'fen_map_t3_decare_pm': {map,tex2,tex3,pm},
        'fen_map_t4_decare': {map,tex2,tex3,tex4},
        'fen_map_t4_decare_pm': {map,tex2,tex3,tex4,pm},
        'fen_map_water': {'water',bump2,'tex1_uv0','tex2_uv1','tex3_uv1',spec,'roughness00'},
        'fen_map_water_pm': {'water',bump2,'tex1_uv0','tex2_uv1','tex3_uv1',spec,'roughness00',pm},
        # ---------------------------------------------
        'fen_prim_bright_pixel': None,
        'fen_prim_compo_height_map': None,
        'fen_prim_copy': None,
        'fen_prim_copy_a2': None,
        'fen_prim_decode_shadow': None,
        'fen_prim_decode_z': None,
        'fen_prim_dof_add': None,
        'fen_prim_edge': None,
        'fen_prim_edge2': None,
        'fen_prim_edge_noise': None,
        'fen_prim_frame_add': None,
        'fen_prim_fxaa': None,
        'fen_prim_gauss_x': None,
        'fen_prim_gauss_y': None,
        'fen_prim_glare_add': None,
        'fen_prim_height_map_base_pc': None,
        'fen_prim_height_map_pc': None,
        'fen_prim_make_height_map_depth': None,
        'fen_prim_pct_ad': None,
        'fen_prim_pre_frame_add': None,
        'fen_prim_refraction': None,
        'fen_prim_shadeoff': None,
        'fen_prim_shadeoff_2p': None,
        'fen_prim_shadow_add': None,
        'fen_prim_shadow_gauss_x': None,
        'fen_prim_shadow_gauss_y': None,
        'fen_prim_shadow_pixel': None,
        'fen_prim_ssao': None,
        'fen_prim_t1_sp': None,
        'fen_prim_t2': None,
        'fen_prim_t2_add': None,
        'fen_prim_t2_add_sp': None,
        'fen_prim_t2_sp': None,
        'fen_prim_wind_map_t1': None,
        'fen_prim_wind_map_t2': None,
        'fen_refraction': None,
        # ---------------------------------------------
        'fen_shadow_t1': None,
        'fen_shadow_t1_c4': None,
        'fen_shadow_t1_pt': None,
        'fen_shadow_t1_pt_c4': None,
        'fen_shadow_t1t': None,
        'fen_shadow_t1t_pt': None,
        'fen_shadow_t2': None,
        'fen_shadow_t2_c4': None,
        'fen_shadow_t2_pt': None,
        'fen_shadow_t2_pt_c4': None,
        'fen_shadow_t2t': None,
        'fen_shadow_t2t_pt': None,
        'fen_shadow_t3': None,
        'fen_shadow_t3_c4': None,
        'fen_shadow_t3_pt': None,
        'fen_shadow_t3_pt_c4': None,
        'fen_shadow_t4': None,
        'fen_shadow_t4_pt': None,
        'fen_shadow_t5': None,
        'fen_shadow_t5_pt': None,
        # ---------------------------------------------
        'fen_t1': set(),
        'fen_t1_ad': None,
        'fen_t1_invalid_light': {unlit},
        'fen_t1_invalid_light_ad': None,
        'fen_t1_invalid_light_sp': {unlit,'soft_particle'},
        'fen_t1_sp': None,
        'fen_t2': None,
        'fen_t2_ad': None,
        'fen_t2_alpha_compo': None,
        'fen_t2_alpha_compo_invalid_light': None,
        'fen_t2_alpha_compo_invalid_light_sp': None,
        'fen_t2_alpha_compo_sp': None,
        'fen_t2_invalid_light': None,
        'fen_t2_invalid_light_ad': None,
        'fen_t2_invalid_light_sp': None,
        'fen_t2_sp': None,
        'fen_warp_t1': {'warp_uv0_tex1'},
        'fen_warp_t1_invalid_light': {unlit,'warp_uv0_tex1'},
        'fen_warp_t1_invalid_light_sp': {unlit,'warp_uv0_tex1','soft_particle'},
        'fen_warp_t1_sp': None,
        # ---------------------------------------------
        'fen_wind_map_t1': None,
        'fen_wind_map_t1_conv': None,
        'fen_wind_map_t2': None,
        'fen_wind_map_t2_conv': None,
        'fen_wind_map_t3': None,
        'fen_wind_map_t3_conv': None,
        'fen_wind_map_t4': None,
        'fen_wind_map_t4_conv': None,
        'kf_prim_pc': None,
        'kf_prim_pct': None,
        'kf_prim_pct2': None,
        'kf_prim_pctwh': None,
        'kf_prim_pnc': None,
        'kf_prim_pnct': None,
        'kf_prim_pntct2': None,
        'ui_blend_miyako': None,
        'ui_blur_miyako': None,
        'ui_blur_tmp_miyako': None,
        'ui_color_miyako': None,
        'ui_flash_back_miyako': None,
        'ui_mask_miyako': None,
    }
    vc4_names = { hash_fnv1a64(name): name for name in vc4_traits.keys() }
    return vc1_traits, vc4_traits, vc4_names

VC1_SHADER_TRAITS, VC4_SHADER_TRAITS, VC4_SHADER_NAMES = _make_shader_traits()

class ValkKFMS(ValkFile, VC4VertexFormatMixin):
    # Doesn't contain other files.
    # Describes model armature, materials, meshes, and textures.
    def read_toc(self):
        self.seek(self.header_length)
        unk1 = self.read(4)
        if unk1[0] == 3:
            self.vc_game = 4
            self.endianness = '<'
        elif unk1[0] == 1:
            print('Little-Endian VC1 model found.')
            self.vc_game = 1
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
            self.bone_list_ptr = self.read_long_long_le()
            self.read_long_long_le() # pointer to extra per-bone data
            self.bone_xform_list_ptr = self.read_long_long_le()
            self.material_list_ptr = self.read_long_long_le()
            self.object_list_ptr = self.read_long_long_le()
            self.mesh_list_ptr = self.read_long_long_le()
            self.read(8) # unknown pointer?
            self.texture_list_ptr = self.read_long_long_le()
            self.mesh_info_ptr = self.read_long_long_le()
        else:
            self.bone_count = self.read_long_auto()
            self.deform_count = self.read_long_auto()
            self.read(4)
            self.model_height = self.read_float_auto()
            self.bone_list_ptr = self.read_long_auto()
            self.read_long_be() # pointer to extra per-bone data
            self.bone_xform_list_ptr = self.read_long_auto()
            self.material_count = self.read_long_auto()
            self.material_list_ptr = self.read_long_auto()
            self.object_count = self.read_long_auto()
            self.object_list_ptr = self.read_long_auto()
            self.mesh_count = self.read_long_auto()
            self.mesh_list_ptr = self.read_long_auto()
            self.read(4)
            self.read(4)
            self.texture_count = self.read_long_auto()
            self.texture_list_ptr = self.read_long_auto()
            self.read_word_auto() # These 3 words are counts that correspond
            self.read_word_auto() # to the next group of 3 longs, which are
            self.read_word_auto() # pointers. Purpose is unknown.
            self.read(2)
            self.read_long_auto()
            self.read_long_auto()
            self.read_long_auto()
            self.read(4)
            self.vertex_format_count = 1
            self.vertex_formats = []
            self.mesh_info_ptr = self.read_long_auto()

    def read_kfmg_info_struct(self):
        entry = InfoDict(
            unk_hash         = self.read_long_auto(),
            bytes_per_vertex = self.read_long_auto(),
            face_ptr         = self.read_long_auto(),
            face_count       = self.read_long_auto(),
            vertex_ptr       = self.read_long_auto(),
            vertex_count     = self.read_long_auto(),
            padding          = self.read(8),
        )
        if self.vc_game == 4:
            entry.update_fields(
                vformat          = self.read_vformat_spec(),
                padding2         = self.read(16),
            )
        self.check_unknown_fields('kfmg_info', entry, {})
        return entry

    def read_kfmg_info(self):
        self.follow_ptr(self.mesh_info_ptr)
        self.vertex_formats = [ self.read_kfmg_info_struct() for i in range(getattr(self, 'vertex_format_count')) ]
        for fmt in self.vertex_formats:
            if 'vformat' in fmt:
                self.read_vformat_fields(fmt.vformat)

    def read_bone_list(self):
        self.follow_ptr(self.bone_list_ptr)
        self.bones = []
        for i in range(self.bone_count):
            bone = {}
            if self.vc_game == 1:
                bone['ptr'] = self.tell()
                bone['unknown_1'] = self.read(4)
                bone['id'] = self.read_word_auto()
                bone['parent_id'] = self.read_word_auto()
                bone['dim1'] = self.read_float_auto()
                bone['dim2'] = self.read_float_auto()
                bone['parent_ptr'] = self.read_long_auto()
                bone['first_child_ptr'] = self.read_long_auto()
                bone['next_sibling_ptr'] = self.read_long_auto()
                bone['bound_box_ptr'] = self.read_long_auto()
                bone['unknown_2'] = self.read(2)
                bone['object_count'] = self.read_word_auto()
                bone['unknown_3'] = self.read(4)
                bone['deform_count'] = self.read_word_auto() # First bone only
                bone['is_deform'] = self.read_word_auto()
                bone['object_ptr1'] = self.read_long_auto()
                bone['object_ptr2'] = self.read_long_auto()
                bone['object_ptr3'] = self.read_long_auto()
                bone['deform_ids_ptr'] = self.read_long_auto() # First bone only
                bone['deform_ptr'] = self.read_long_auto()
                bone['unknown_4'] = self.read(32)
            elif self.vc_game == 4:
                bone['ptr'] = self.tell() - 0x20
                self.read(4)
                bone['id'] = self.read_word_le()
                bone['parent_id'] = self.read_word_le()
                bone['dim1'] = self.read_float_le()
                bone['dim2'] = self.read_float_le()
                bone['parent_ptr'] = self.read_long_long_le()
                bone['first_child_ptr'] = self.read_long_long_le()
                bone['next_sibling_ptr'] = self.read_long_long_le()
                bone['bound_box_ptr'] = self.read_long_long_le()
                self.read(4) # 0x20202020
                self.read(2)
                bone['object_count'] = self.read_word_le()
                self.read(4)
                bone['deform_count'] = self.read_word_le() # First bone only
                bone['is_deform'] = self.read_word_le()
                bone['object_ptr1'] = self.read_long_long_le()
                bone['object_ptr2'] = self.read_long_long_le()
                bone['object_ptr3'] = self.read_long_long_le()
                self.read(8)
                bone['deform_ids_ptr'] = self.read_long_long_le() # First bone only
                bone['deform_ptr'] = self.read_long_long_le()
                self.read(48)
            self.bones.append(bone)

    def link_bones(self):
        for bone in self.bones:
            bone['parent'] = None
            bone['children'] = []
            if bone['parent_id'] == bone['id']:
                continue
            parent_bone = self.bones[bone['parent_id']]
            bone['parent'] = parent_bone
            parent_bone['children'].append(bone)

    def read_bone_xforms(self):
        self.follow_ptr(self.bone_xform_list_ptr)
        if self.vc_game == 1:
            read_float = self.read_float_auto
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
            self.follow_ptr(bone['deform_ptr'])
            if self.vc_game == 1:
                bone['matrix_ptr'] = self.read_long_auto()
                bone['deform_id'] = self.read_long_auto()
            elif self.vc_game == 4:
                bone['matrix_ptr'] = self.read_long_long_le()
                self.read(2) # unknown
                bone['deform_id'] = self.read_long_le()
                self.read(2) # unknown
            self.deform_bones[bone['deform_id']] = bone

    def read_bone_matrices(self):
        if self.vc_game == 1:
            read_float = self.read_float_auto
        elif self.vc_game == 4:
            read_float = self.read_float_le
        for bone in self.bones:
            if 'matrix_ptr' not in bone:
                continue
            self.follow_ptr(bone['matrix_ptr'])
            bone['matrix_raw'] = (
                (read_float(), read_float(), read_float(), read_float()),
                (read_float(), read_float(), read_float(), read_float()),
                (read_float(), read_float(), read_float(), read_float()),
                (read_float(), read_float(), read_float(), read_float())
                )

    def read_material_struct(self):
        if self.vc_game == 1:
            material = InfoDict(
                ptr                  = self.tell_ptr(),
                unk1                 = self.read_long_auto(),
                shader_id            = self.read_long_auto(),
                pad2a                = self.read(4),
                num_textures         = self.read_byte(),
                src_blend            = self.read_byte(), # D3DBLEND
                dst_blend            = self.read_byte(), # D3DBLEND
                use_backface_culling = self.read_byte(),
                texture0_ptr         = self.read_long_auto(),
                texture1_ptr         = self.read_long_auto(),
                texture2_ptr         = self.read_long_auto(),
                pad3b                = self.read(4),
                pad4                 = self.read(16),
                unk5a                = read_tuple(self.read_float_auto, 4),
                unk5b                = read_tuple(self.read_float_auto, 4),
                unk5c                = read_tuple(self.read_float_auto, 4),
                unk5d                = read_tuple(self.read_float_auto, 4),
                pad6                 = self.read(16),
                pad7a                = self.read(8),
                unk7b                = self.read_long_auto(),
                pad7c                = self.read(4),
                unk8a                = self.read(12),
                unk8d                = self.read_long_auto(),
            )
            self.check_unknown_fields('material', material, {
                'src_blend': {5, 2},
                'dst_blend': {6, 2, 1},
                #'unk5a': (1,1,1,1), # evmap08_01
                'unk5b': (0,0,0,1),
                #'unk5c': (1,1,1,0), # evmap08_01
                #'unk5d': (0,0,0,1), # C04aB evmap08_01
                'unk8a': {b'\x20\x20\x20\x20\x00\x00\x00\x00\x00\x00\x00\x00',
                          b'\x20\x20\x20\x20BH\x00\x00\x00\x00\x00\x00'}, # C04aB
            })
            sid = material.shader_id
            material.traits = VC1_SHADER_TRAITS.get(sid, set())
            if sid not in VC1_SHADER_TRAITS:
                self.print_location('unexpected shader id {:x}: {}', sid, material)
        else:
            material = InfoDict(
                ptr                  = self.tell_ptr(),
                flags1               = self.read_long_auto(),
                num_textures         = self.read_byte(),
                src_blend            = self.read_byte(), # D3DBLEND
                dst_blend            = self.read_byte(), # D3DBLEND
                use_backface_culling = self.read_byte(),
                shader_hash          = self.read_long_long_le(),
                unk3a                = self.read_long_auto(),
                unk3b                = self.read_long_auto(),
                unk3c                = self.read_long_auto(),
                unk3d                = self.read_long_auto(), # hash or float?
                pad4a                = self.read(4),
                unk4b                = self.read(4),
                unk4c                = self.read_float_auto(),
                unk4da               = self.read_byte(),
                num_parameters       = self.read_byte(),
                unk4dc               = self.read_byte(),
                pad4dd               = self.read(1),
                unk5a                = read_tuple(self.read_float_auto, 4),
                unk5b                = read_tuple(self.read_float_auto, 4),
                unk5c                = read_tuple(self.read_float_auto, 4),
                unk5d                = read_tuple(self.read_float_auto, 4),
                parameter_ptr        = self.read_long_long_le(), # named uniforms
                pad6c                = self.read(8),
                texture0_ptr         = self.read_long_long_le(),
                texture1_ptr         = self.read_long_long_le(),
                texture2_ptr         = self.read_long_long_le(),
                texture3_ptr         = self.read_long_long_le(),
                texture4_ptr         = self.read_long_long_le(),
                padding              = self.read(0x48),
            )
            self.check_unknown_fields('material', material, {
                'src_blend': {5, 2},
                'dst_blend': {6, 2, 1},
                'unk3a': set(range(50)),
                #'unk3c': {
                #    0,1,2,
                #    330,680, # vla005a
                #    444,556, # vlb106a
                #    521,     # vlb107a
                #    3,       # vl_vc301 (dlc1)
                #    681,682, # vl_bs003
                #    600,     # vl_puppet_a
                #    445,635,608, # evmap_ground_01_02_a
                #    },
                'unk4b': b'    ',
                'unk4c': {
                    24,
                    160.9803924560547, # vl_pf016 (dlc1)
                    1, # evmap_ground_01_01_a
                    179.49696350097656, # evmap_ground_06_07_a
                    200.0, # omap_ground_03_01a
                    },
                'unk4da': {0,2}, # evmap_ground_06_02_a
                'unk4dc': set(range(6)), # evmap_ground_02_02_a
                'unk5a': {
                    (1,1,1,1),(0,0,0,1),(0,0,0,0),
                    (0.5799953937530518, 0.27499809861183167, 0.0, 1.0), # omap_ground_00_01a
                    (1.0, 1.0, 1.0, 2.0), (0.0, 0.0, 0.0, 0.800000011920929), # omap_ground_06_01a
                    },
                'unk5b': {
                    (0,0,0,0),
                    (1,1,1,0), # evmap_ground_01_01_a
                    (0.9174334406852722, 0.9174334406852722, 0.9174334406852722, 0.0), # evmap_ground_02_09_a
                    },
                'unk5c': {
                    (1,1,1,0),
                    (0.20000000298023224,0.20000000298023224,0.20000000298023224,0), # vlb108a, evmap_ground_02_02_a
                    (0.7787594199180603,0.7787594199180603,0.7787594199180603,0), # vlb108a, evmap_ground_03_02_a
                    (1.0, 0.7841535210609436, 0.7559999823570251, 0.0), # evmap_ground_03_08_a
                    (0.5799953937530518, 0.27499809861183167, 0.0, 1.0), # omap_ground_00_01a
                    (0.5799953937530518, 0.27499809861183167, 0.0, 0.0), # omap_ground_00_01a
                    (5.0, 5.0, 5.0, 0.0), # omap_ground_06_01a
                    },
                'unk5d': {
                    (1,1,1,0),
                    (0,0,0,0), # evmap_ground_01_01_a
                    }
            })
            shash = material.shader_hash
            material.shader_name = sid = VC4_SHADER_NAMES.get(shash) or hex(shash)[2:]
            material.traits = (VC4_SHADER_TRAITS.get(sid) or set()) | {'v4'}
            if VC4_SHADER_TRAITS.get(sid) is None:
                print('no traits for shader id ', sid)
            if (material.flags1 & 0x24): # vlb112a etc - face uses (5,6) but shouldn't have alpha
                # All VC4 shaders include an Alpha Clip check using the first texture alpha
                # and a threshold uniform, but the source of the threshold value is unknown.
                material.traits.add('alpha')
                # Check the flags as a heuristic to avoid setting big parts of characters to Blend.
                # This trait is only relevant to Blender Eevee, in Cycles it's effectively always on.
                if (material.flags1 & 0x4) and material.src_blend == 5: # D3DBLEND_SRCALPHA
                    material.traits.add('alphablend')
            if material.dst_blend == 2: # D3DBLEND_ONE
                material.traits.add('add_shader')

        return material

    def read_material_param_struct(self):
        entry = InfoDict(
            id_hash     = self.read_long_long_le(),
            name_ptr    = self.read_long_long_le(),
            type_spec   = read_tuple(self.read_long_auto, 2),
            pad1        = self.read(8),
        )
        if entry.type_spec == (1,0):
            entry.update_fields(
                data        = self.read_long_auto(),
                padding     = self.read(16+12),
            )
        else:
            entry.update_fields(
                data        = read_tuple(self.read_float_auto, 4),
                padding     = self.read(16),
            )
        self.check_unknown_fields('material param', entry, {
            'type_spec': {
                (5,2),
                (1,0), # lightmap_compo_type, use_normal_lerp
                (5,1), # cDisplacementParam
            },
        })
        return entry

    def read_material_list(self):
        self.follow_ptr(self.material_list_ptr);
        material_list = [ self.read_material_struct() for i in range(getattr(self, 'material_count', 1)) ]

        for mat in material_list:
            if mat.get('parameter_ptr'):
                self.follow_ptr(mat.parameter_ptr)
                mat.parameter_list = [ self.read_material_param_struct() for i in range(mat.num_parameters) ]

                for param in mat.parameter_list:
                    self.follow_ptr(param.name_ptr)
                    param.name = self.read_string()

                    assert hash_fnv1a64(param.name) == param.id_hash

                mat.parameters = { param.name: param.data for param in mat.parameter_list }
            else:
                mat.parameters = {}

        self.materials = { mat.ptr: mat for mat in material_list }

    def read_object_list(self):
        self.follow_ptr(self.object_list_ptr)
        self.objects = []
        for i in range(self.object_count):
            if self.vc_game == 1:
                object_row = {
                    'id': self.read_long_auto(),
                    'parent_is_armature': self.read_word_auto(),
                    'parent_bone_id': self.read_word_auto(),
                    'material_ptr': self.read_long_auto(),
                    'u01': self.read_word_auto(),
                    'mesh_count': self.read_word_auto(),
                    'mesh_list_ptr': self.read_long_auto(),
                    'kfmg_vertex_offset': self.read_long_auto(),
                    'vertex_count': self.read_word_auto(),
                    }
                if object_row['u01'] != 0:
                    print('u01 nonzero', object_row)
                self.read(6)
            elif self.vc_game == 4:
                object_row = {
                    'id': self.read_long_le(),
                    'parent_is_armature': self.read_word_le(),
                    'parent_bone_id': self.read_word_le(),
                    'material_ptr': self.read_long_le(), # 64-bit?
                    'u02': self.read_long_le(),
                    'kfmg_vertex_offset': self.read_long_le(),
                    'vertex_count': self.read_word_le(),
                    'vertex_format': self.read_word_le(),
                    'mesh_count': self.read_long_le(),
                    'u03': self.read_long_le(),
                    'mesh_list_ptr': self.read_long_le(), # 64-bit?
                    }
                self.read(4 * 7)
            self.objects.append(object_row)

    def read_mesh_list(self):
        self.meshes = []
        for obj in self.objects:
            self.follow_ptr(obj['mesh_list_ptr'])
            for i in range(obj['mesh_count']):
                if self.vc_game == 1:
                    mesh_row = {
                        'vertex_group_count': self.read_word_auto(),
                        'u01': self.read_word_auto(),
                        'u02': self.read_word_auto(),
                        'vertex_count': self.read_word_auto(),
                        'faces_word_count': self.read_word_auto(),
                        'n01': self.read_long_auto(),
                        'vertex_group_map_ptr': self.read_word_auto(),
                        'first_vertex': self.read_long_auto(),
                        'faces_first_word': self.read_long_auto(),
                        'first_vertex_id': self.read_long_auto(),
                        'n02': self.read_long_auto(),
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
                        'vertex_group_map_ptr': self.read_long_le(), # 64-bit?
                        'n02': self.read_long_le(),
                        'object': obj,
                        }
                self.meshes.append(mesh_row)

    def read_vertex_group_maps(self):
        vertex_group_map = {}
        for mesh in self.meshes:
            self.follow_ptr(mesh['vertex_group_map_ptr'])
            for i in range(mesh['vertex_group_count']):
                if self.vc_game == 1:
                    global_id = self.read_word_auto()
                    local_id = self.read_word_auto()
                elif self.vc_game == 4:
                    global_id = self.read_word_le()
                    local_id = self.read_word_le()
                vertex_group_map[local_id] = global_id
            mesh['vertex_group_map'] = vertex_group_map.copy()

    def read_texture_struct(self, i):
        texture = InfoDict(
            id           = i,
            ptr          = self.tell_ptr(),
            unk0         = self.read_long_auto(),
            image        = self.read_word_auto(),
            unk1         = self.read_word_auto(),
            blend_factor = self.read_float_auto(),
            pad3a        = self.read(1),
            unk3b        = self.read_byte(),
            unk3c        = self.read_byte(),
            unk3d        = self.read_byte(),
            pad3c        = self.read(8),
            unk4         = self.read_float_auto(),
            unk5         = self.read_float_auto(),
        )
        if self.vc_game == 1:
            texture.update_fields(
                pad6        = self.read(16),
                unk7a       = self.read(4),
                pad7b       = self.read(12),
            )
            self.check_unknown_fields('texture', texture, {
                'unk0': {0,2,8},
                'unk1': 0,
                'blend_factor': 1,
                'unk3b': {1,3},
                'unk3c': {0,1},
                'unk3d': {0,1},
                'unk4': 1, 'unk5': 1,
                'unk7a': b'    ',
            })
        else:
            texture.update_fields(
                unk6a       = self.read_long_auto(),
                pad6b       = self.read(12),
                pad7a       = self.read(4),
                unk7b       = self.read(4),
                unk7c       = self.read_long_auto(),
                pad7d       = self.read(4),
                pad8        = self.read(16*2),
            )
            self.check_unknown_fields('texture', texture, {
                'unk0': {0,1,2,8,9},
                'unk1': {0,1,2,3},
                #'blend_factor': {
                #    1,
                #    0,  # vl_vc301 (dlc1) - blend factor? set to 0 for damage texture
                #    0.10000000149011612, # evmap_ground_02_02_a
                #    0.07999999821186066, # evmap_ground_02_09_a
                #    0.7961783409118652, # evmap_ground_03_02_a
                #    0.5,0.20000000298023224,0.8248175382614136,0.8500000238418579,0.05999999865889549 #... evmap_ground_03_08_a
                #    },
                'unk3b': {1,3},
                'unk3c': {0,1}, # omap_ground_06_01a
                'unk3d': {0,1}, # omap_ground_06_01a
                'unk4': 1, 'unk5': 1,
                'unk7b': b'    ',
                'unk7c': {
                    1,
                    0,  # vl_bs001
                    101, 105, # evmap_ground_03_06_a
                    97, 108, # evmap_ground_04_06_a
                    },
            })
        return texture

    def read_texture_list(self):
        self.follow_ptr(self.texture_list_ptr)
        texture_list = [ self.read_texture_struct(i) for i in range(self.texture_count) ]
        self.textures = { tex.ptr: tex for tex in texture_list }

    TEXTURE_FIELD_NAMES = [ ('texture%d' % (i), 'texture%d_ptr' % (i)) for i in range(5) ]

    def link_materials(self):
        for material_ptr, material in self.materials.items():
            for texname, ptrname in self.TEXTURE_FIELD_NAMES:
                if ptrname in material:
                    ptr = material[ptrname]
                    material[texname] = self.textures[ptr] if ptr else None

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


class ValkKFMG(ValkFile, VC4VertexFormatMixin):
    # Doesn't contain other files.
    # Holds mesh vertex and face data.

    def read_faces(self, first_word, word_count, vertex_format):
        fmt_face_offset = vertex_format['face_ptr']
        self.seek(self.header_length + self.face_ptr + fmt_face_offset + first_word * 2)
        end_ptr = self.tell() + word_count * 2
        start_direction = 1
        if self.vc_game == 1:
            read_vertex_id = self.read_word_auto
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
                        face = (v3, v2, v1)
                    else:
                        face = (v3, v1, v2)
                    faces.append(face)
                v1 = v2
                v2 = v3
        return faces

    def read_vertex(self, vertex_format):
        bytes_per_vertex = vertex_format['bytes_per_vertex']
        if self.vc_game == 1 and bytes_per_vertex == 0x2c:
            vertex = {
                'location': read_tuple(self.read_float_auto),
                # Some kind of direction data as 4 bytes - tangent?
                'unknown_vec': self.read(4),
                'normal': read_tuple(self.read_half_float_auto),
                'normal_pad': self.read(2),
                'color': read_tuple(self.read_byte_factor, 4),
                'color2': read_tuple(self.read_byte_factor, 4), # evmap05_02
                'uv': read_tuple(self.read_half_float_auto, 2),
                'uv2': read_tuple(self.read_half_float_auto, 2),
                'uv3': read_tuple(self.read_half_float_auto, 2), # evmap18_06
                }
            if vertex['normal_pad'] != b'\x00\x00':
                print('vertex 0x2c normal_pad nonzero:', vertex)
        elif self.vc_game == 1 and bytes_per_vertex == 0x30:
            vertex = {
                'location': read_tuple(self.read_float_auto),
                'vertex_group_1': self.read_byte(),
                'vertex_group_2': self.read_byte(),
                'vertex_group_3': self.read_byte(),
                'vertex_group_pad': self.read_byte(),
                'vertex_group_weight_1': self.read_half_float_auto(),
                'vertex_group_weight_2': self.read_half_float_auto(),
                'color': read_tuple(self.read_byte_factor, 4),
                'uv': read_tuple(self.read_half_float_auto, 2),
                'uv2': read_tuple(self.read_half_float_auto, 2),
                'uv3': read_tuple(self.read_half_float_auto, 2),
                'normal': read_tuple(self.read_half_float_auto),
                'normal_pad': self.read(2),
                # Some kind of direction data as 4 bytes - tangent?
                'unknown_vec': self.read(4),
                }
            if vertex['normal_pad'] != b'\x00\x00':
                print('vertex 0x30 normal_pad nonzero:', vertex)
        elif self.vc_game == 1 and bytes_per_vertex == 0x50:
            vertex = {
                'location': read_tuple(self.read_float_auto),
                # (0.0, 1.0, 0.0) - or maybe ([0,0,0,0], 1.0, 0.0) for vertex groups?..
                'unknown_1a': self.read_long_auto(),
                'unknown_1b': self.read_float_auto(),
                'unknown_1c': self.read_long_auto(),
                # Some kind of direction data as 4 bytes - two tangents?
                'unknown_vec': self.read(4 * 2),
                'normal': read_tuple(self.read_float_auto),
                'color': read_tuple(self.read_byte_factor, 4), # val_mp004
                'uv': read_tuple(self.read_float_auto, 2),
                'uv2': read_tuple(self.read_float_auto, 2),
                'unknown_4': self.read(4 * 4),
                }
            if vertex['unknown_1a'] != 0 or vertex['unknown_1b'] != 1 or vertex['unknown_1c'] != 0:
                print('vertex 0x50 unknown_1 nonzero:', vertex)
            if vertex['unknown_4'] != b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
                print('vertex 0x50 unknown_4 nonzero:', vertex)
        elif self.vc_game == 4:
            struct = vertex_format.vformat.fields
            read_float = self.read_float_le
            vertex = {}
            vertex_begin = self.tell()
            for field in struct:
                element = field[1:]
                if element == self.VERT_LOCATION:
                    vertex['location'] = read_tuple(read_float)
                elif element == self.VERT_WEIGHTS:
                    vertex['vertex_group_weight_1'] = read_float()
                    vertex['vertex_group_weight_2'] = read_float()
                    vertex['vertex_group_weight_3'] = read_float()
                elif element == self.VERT_GROUPS:
                    vertex['vertex_group_1'] = self.read_byte()
                    vertex['vertex_group_2'] = self.read_byte()
                    vertex['vertex_group_3'] = self.read_byte()
                    vertex['vertex_group_4'] = self.read_byte()
                elif element == self.VERT_NORMAL:
                    vertex['normal'] = read_tuple(read_float)
                elif element == self.VERT_TANGENT:
                    vertex['tangent'] = read_tuple(read_float)
                elif element == self.VERT_UV1:
                    vertex['uv'] = read_tuple(read_float, 2)
                elif element == self.VERT_UV2:
                    vertex['uv2'] = read_tuple(read_float, 2)
                elif element == self.VERT_UV3:
                    vertex['uv3'] = read_tuple(read_float, 2)
                elif element == self.VERT_UV4:
                    vertex['uv4'] = read_tuple(read_float, 2)
                elif element == self.VERT_UV5:
                    vertex['uv5'] = read_tuple(read_float, 2)
                elif element == self.VERT_COLOR:
                    vertex['color'] = read_tuple(read_float, 4)
                else:
                    raise NotImplementedError('Unknown vertex data element: {}'.format(element))
            # Sometimes vertex data is padded, and bytes_per_vertex is larger
            # than the actual amount of data in a vertex.
            self.seek(vertex_begin + bytes_per_vertex)
        else:
            raise NotImplementedError('Unsupported vertex type. Bytes per vertex: {}'.format(bytes_per_vertex))
        return vertex

    def read_vertices(self, first_vertex, vertex_count, vertex_format):
        fmt_bytes_per_vertex = vertex_format['bytes_per_vertex']
        fmt_vertex_offset = vertex_format['vertex_ptr']
        self.seek(self.header_length + self.vertex_ptr + fmt_vertex_offset + first_vertex * fmt_bytes_per_vertex)
        vertices = []
        for i in range(vertex_count):
            vertex = self.read_vertex(vertex_format)
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
        if self.texture_pack_count < 2**16:
            self.vc_game = 1
        else:
            self.vc_game = 4
            self.seek(self.header_length + 0x4)
            self.texture_pack_count = self.read_long_le()
            self.seek(self.header_length + 0x20)
            self.texture_pack_list_ptr = self.read_long_le()

    def read_texture_pack_list(self):
        self.follow_ptr(self.texture_pack_list_ptr)
        for i in range(self.texture_pack_count):
            if self.vc_game == 1:
                pack = {
                    "id_count": self.read_long_be(),
                    "id_list_ptr": self.read_long_be(),
                    }
                self.read(8)
            elif self.vc_game == 4:
                pack = {
                    "id_count": self.read_long_le(),
                    "unk": self.read(4),
                    "id_list_ptr": self.read_long_long_le(),
                    }
            self.texture_packs.append(pack)
        for pack in self.texture_packs:
            self.follow_ptr(pack["id_list_ptr"])
            pack["htsf_ids"] = []
            for i in range(pack["id_count"]):
                if self.vc_game == 1:
                    htsf_id = self.read_long_be()
                elif self.vc_game == 4:
                    htsf_id = self.read_long_le()
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
        self.follow_ptr(toc_start)
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
        self.follow_ptr(self.bone_list_ptr)
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
        self.follow_ptr(bone[ptr_key])
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
        self.follow_ptr(frames_ptr)
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
            self.follow_ptr(bone["xform_ptr"])
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
                self.follow_ptr(bone["anim_ptr"])
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
            "SlgEnObject",
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
        self.PRINT_MODELS = False
        self.PRINT_MODEL_PARAMS = False
        self.PRINT_MODEL_FILES = False
        super().__init__(F, offset)

    def read_toc(self):
        self.seek(self.header_length)
        version = self.read(4)
        if version[1] == 0:
            self.vc_game = 1
        elif version[2] == 0:
            self.vc_game = 4
        if self.vc_game == 1:
            self.param_block_ptr = self.read_long_be()
            self.model_block_ptr = self.read_long_be()
            self.file_block_ptr = self.read_long_be()
            if self.param_block_ptr:
                self.follow_ptr(self.param_block_ptr + 0x4)
                self.param_count = self.read_long_be()
                self.param_list_ptr = self.read_long_be()
            if self.model_block_ptr:
                self.follow_ptr(self.model_block_ptr + 0x4)
                self.model_count = self.read_long_be()
                self.model_list_ptr = self.read_long_be()
            if self.file_block_ptr:
                self.follow_ptr(self.file_block_ptr + 0x4)
                self.file_count = self.read_long_be()
                self.file_list_ptr = self.read_long_be()
        elif self.vc_game == 4:
            self.seek(self.header_length + 0x20)
            self.param_block_ptr = self.read_long_long_le()
            self.model_block_ptr = self.read_long_long_le()
            self.file_block_ptr = self.read_long_long_le()
            if self.param_block_ptr:
                self.follow_ptr(self.param_block_ptr + 0x8)
                self.param_count = self.read_long_le()
                self.follow_ptr(self.param_block_ptr + 0x20)
                self.param_list_ptr = self.read_long_long_le()
            if self.model_block_ptr:
                self.follow_ptr(self.model_block_ptr + 0x4)
                self.model_count = self.read_long_le()
                self.follow_ptr(self.model_block_ptr + 0x20)
                self.model_list_ptr = self.read_long_long_le()
            if self.file_block_ptr:
                self.follow_ptr(self.file_block_ptr + 0x4)
                self.file_count = self.read_long_le()
                self.follow_ptr(self.file_block_ptr + 0x20)
                self.file_list_ptr = self.read_long_long_le()

    def read_parameter_list(self):
        from subprocess import check_output
        if not hasattr(self, "param_list_ptr"):
            return
        self.follow_ptr(self.param_list_ptr)
        rows = []
        for i in range(self.param_count):
            if self.vc_game == 1:
                row = {
                    "id": self.read_long_be(),
                    "name_ptr": self.read_long_be(),
                    "data_length": self.read_long_be(),
                    "data_ptr": self.read_long_be(),
                    }
            elif self.vc_game == 4:
                row = {
                    "id": self.read_long_le(),
                    "data_length": self.read_long_le(),
                    "name_ptr": self.read_long_long_le(),
                    "data_ptr": self.read_long_long_le(),
                    }
                self.read(8)
            rows.append(row)
        for row in rows:
            self.follow_ptr(row["name_ptr"])
            row["name"] = self.read_string(encoding = "shift_jis_2004")
            if self.PRINT_PARAMS:
                data_pos = self.follow_ptr(row["data_ptr"])
                print('Param:', row)
                print(check_output(["xxd", "-s", str(data_pos + self.offset), "-l", str(row["data_length"]), self.F.filename]).decode("ascii"))
        if self.PRINT_PARAMS:
            print('Done Reading Parameters')
        self.parameters = rows

    def read_model_list(self):
        if not hasattr(self, "model_list_ptr"):
            return
        self.follow_ptr(self.model_list_ptr)
        rows = []
        for i in range(self.model_count):
            if self.vc_game == 1:
                row = {
                    "unk1": self.read_long_be(),
                    "name_ptr": self.read_long_be(),
                    "param_count": self.read_long_be(),
                    "param_list_ptr": self.read_long_be(),
                    }
                self.seek(0x30, True) # Always zero?
            elif self.vc_game == 4:
                row = {
                    "unk1": self.read_long_le(),
                    "param_count": self.read_long_le(),
                    "unk2": self.read(0x10),
                    "name_ptr": self.read_long_long_le(),
                    "param_list_ptr": self.read_long_long_le(),
                    }
                self.seek(0x28, True) # Always zero?
            rows.append(row)
        for row in rows:
            self.follow_ptr(row["name_ptr"])
            row["name"] = self.read_string(encoding = "shift_jis_2004")
            if self.PRINT_MODELS:
                print('Model:', row)
        if self.PRINT_MODELS:
            print('Done Reading Models')
        self.models = rows

    def read_file_list(self):
        if not hasattr(self, "file_list_ptr"):
            return
        self.follow_ptr(self.file_list_ptr)
        file_rows = []
        for i in range(self.file_count):
            if self.vc_game == 1:
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
            elif self.vc_game == 4:
                row = {
                    "is_inside": self.read_long_le(),
                    "id": self.read_long_le(),
                    "type": self.read_long_le(),
                    "htr_index": self.read_long_le(),
                    "unk1": self.read_long_le(),
                    "mmr_index": self.read_long_le(),
                    "path_ptr": self.read_long_long_le(),
                    "filename_ptr": self.read_long_long_le(),
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
            if self.vc_game == 1:
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
            self.follow_ptr(row["filename_ptr"])
            row["filename"] = self.read_string()
            if self.PRINT_FILES:
                print('File:', row)
        if self.PRINT_FILES:
            print('Done Reading Files')
        self.files = file_rows

    def read_model_param_ids(self):
        from subprocess import check_output
        for model in self.models:
            self.follow_ptr(model["param_list_ptr"])
            param_refs = []
            for i in range(model["param_count"]):
                if self.vc_game == 1:
                    param_text_ptr = self.read_long_be()
                    param_id_count = self.read_long_be()
                    param_id_ptr = self.read_long_be()
                    self.read(4)
                elif self.vc_game == 4:
                    param_id_count = self.read_long_le()
                    self.read(4)
                    param_text_ptr = self.read_long_long_le()
                    param_id_ptr = self.read_long_long_le()
                param_refs.append([param_text_ptr, param_id_count, param_id_ptr])
            param_groups = []
            for param_text_ptr, param_id_count, param_id_ptr in param_refs:
                param_group = {}
                self.follow_ptr(param_text_ptr)
                param_group["text"] = self.read_string("shift_jis_2004")
                param_group["param_ids"] = []
                self.follow_ptr(param_id_ptr)
                if self.PRINT_MODEL_PARAMS:
                    print('Model Param Group:', param_group["text"])
                for i in range(param_id_count):
                    if self.vc_game == 1:
                        param_id = self.read_long_be()
                    elif self.vc_game == 4:
                        param_id = self.read_long_le()
                    if self.PRINT_MODEL_PARAMS:
                        print('Model Param:', self.parameters[param_id])
                        print(check_output(["xxd", "-s", str(self.parameters[param_id]["data_ptr"] + self.offset), "-l", str(self.parameters[param_id]["data_length"]), self.F.filename]).decode("ascii"))
                    param_group["param_ids"].append(param_id)
                param_groups.append(param_group)
            model["param_groups"] = param_groups
        if self.PRINT_MODEL_PARAMS:
            print('Done Reading Model Parameters')

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
                    if self.vc_game == 1:
                        self.follow_ptr(param["data_ptr"] + 0x40)
                        model["location_x"] = self.read_float_be()
                        model["location_y"] = self.read_float_be()
                        model["location_z"] = self.read_float_be()
                        self.follow_ptr(param["data_ptr"] + 0x50)
                        model["rotation_x"] = self.read_float_be()
                        model["rotation_y"] = self.read_float_be()
                        model["rotation_z"] = self.read_float_be()
                        self.follow_ptr(param["data_ptr"] + 0x60)
                        model["scale_x"] = self.read_float_be()
                        model["scale_y"] = self.read_float_be()
                        model["scale_z"] = self.read_float_be()
                        self.follow_ptr(param["data_ptr"] + 0x74)
                        model["model_file_id"] = self.read_long_be()
                        self.follow_ptr(param["data_ptr"] + 0x84)
                        model["texture_file_id"] = self.read_long_be()
                    elif self.vc_game == 4:
                        if group["text"] in ["SlgEnObject", "EnHeightField", "EnSky"] + self.model_types:
                            self.follow_ptr(param["data_ptr"] + 0x10)
                            model["location_x"] = self.read_float_le()
                            model["location_y"] = self.read_float_le()
                            model["location_z"] = self.read_float_le()
                            self.follow_ptr(param["data_ptr"] + 0x20)
                            model["rotation_x"] = self.read_float_le()
                            model["rotation_y"] = self.read_float_le()
                            model["rotation_z"] = self.read_float_le()
                            self.follow_ptr(param["data_ptr"] + 0x30)
                            model["scale_x"] = self.read_float_le()
                            model["scale_y"] = self.read_float_le()
                            model["scale_z"] = self.read_float_le()
                            self.follow_ptr(param["data_ptr"] + 0x40)
                            model["model_file_id"] = self.read_long_le()
                            self.follow_ptr(param["data_ptr"] + 0x48)
                            model["texture_file_id"] = self.read_long_le()
                        else:
                            print(group["text"])
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
        if 0x10 <= path_ptr < filename_ptr < self.main_length:
            self.vc_game = 1
        else:
            self.vc_game = 4
            self.seek(self.header_length)
            path_ptr = self.read_long_le()
            filename_ptr = self.read_long_le()
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
    # Unknown block found in Valkyria Chronicles 4 models
    pass


class Valk4WIRS(ValkFile):
    # Unknown block found in Valkyria Chronicles 4 models
    pass


class Valk4MBHV(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    pass


class Valk4MBMP(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    pass


class Valk4MBHD(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    pass


class Valk4MBMD(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    pass


class Valk4SDPK(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    # Contains ALBD, EFSC, HMDL, HMOT, HMMT, and HTEX blocks
    # Mostly visual effects
    def container_func(self):
        self.seek(self.header_length)
        self.read(0x4) # Unknown
        section_count = self.read_long_le()
        self.read(0x8) # Unknown
        for i in range(section_count):
            file_count = self.read_long_le()
            section_id = self.read(8)
            pad = self.read(4)
            for j in range(file_count):
                file_id = self.read(8)
                file_size = self.read_long_le()
                pad = self.read(4)
                begin = self.tell()
                inner_file = valk_factory(self, begin)
                self.add_inner_file(inner_file)
                self.seek(begin + file_size)


class Valk4ATUD(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    pass


class Valk4NSEN(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    # Seems to refer to models?
    # Has different internal structure from most blocks.

    def container_func(self):
        pass

    def read_meta(self):
        self.seek(0)
        self.ftype = self.read(4).decode('ascii')
        if DEBUG:
            print("Creating", self.ftype)
        import os
        self.seek(0, os.SEEK_END)
        self.total_length = self.tell()
        self.header_length = 0


class Valk4REXP(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    # Contains BSON blocks
    pass


class Valk4HSPK(ValkFile):
    # Shader bytecode pack found in Valkyria Chronicles 4
    def read_data(self):
        assert len(self.KSPK) == 1 and len(self.KSPP) == 1
        kspk = self.KSPK[0]
        kspp = self.KSPP[0]
        kspk.read_data()
        kspp.read_data()

        assert len(kspk.entries) == len(kspp.entries)

        self.entries = kspk.entries
        self.is_dxbc = kspp.is_dxbc

        for kspk_entry, kspp_entry in zip(kspk.entries, kspp.entries):
            assert kspk_entry.shader_hash == kspp_entry.shader_hash
            assert kspk_entry.shader_name == kspp_entry.shader_name
            assert hash_fnv1a64(kspk_entry.shader_name) == kspk_entry.shader_hash
            kspk_entry.data_index = kspp_entry.data_index


class Valk4KSPK(ValkFile, VC4VertexFormatMixin):
    # Shader bytecode pack header in Valkyria Chronicles 4
    def read_toc_header(self):
        header = InfoDict(
            unk1            = self.read_long_le(),
            count           = self.read_long_le(),
            unk2            = self.read_long_le(),
            padding         = self.read(4 + 16*3),
        )
        self.check_unknown_fields('toc_header', header, {
            'unk1': 4,
            'unk2': 64,
        })
        return header

    def read_toc_entry(self):
        entry = InfoDict(
            shader_hash     = self.read_long_long_le(), # FNV1A64 hash of name
            shader_name_ptr = self.read_long_long_le(),
            unk10           = self.read_long_le(),
            option_count    = self.read_long_le(),
            option_ptr      = self.read_long_long_le(),
            vformat         = self.read_vformat_spec(),
        )
        self.check_unknown_fields('toc_entry', entry, {
            'unk10': 3,
        })
        return entry

    def read_option_entry(self):
        entry = InfoDict(
            id_hash    = self.read_long_long_le(),
            num_values = self.read_long_le(),
            padding    = self.read(12),
        )
        self.check_unknown_fields('option', entry, {
            'num_values': {2, 3, 4},
        })
        return entry

    def read_data(self):
        self.vc_game = 4

        self.seek(self.header_length)
        self.toc_header = self.read_toc_header()
        self.entries = [ self.read_toc_entry() for i in range(self.toc_header.count) ]

        for entry in self.entries:
            self.follow_ptr(entry.shader_name_ptr)
            entry.shader_name = self.read_string()
            self.follow_ptr(entry.option_ptr)
            entry.options = [ self.read_option_entry() for i in range(entry.option_count) ]

            self.read_vformat_fields(entry.vformat)


class Valk4KSPP(ValkFile):
    # Shader bytecode pack data in Valkyria Chronicles 4
    def read_toc_header(self):
        header = InfoDict(
            unk1            = self.read_long_le(),
            count           = self.read_long_le(),
            unk2            = self.read_long_le(),
            padding         = self.read(4 + 16*3)
        )
        self.check_unknown_fields('toc_header', header, {
            'unk1': 4,
            'unk2': 64,
        })
        return header

    def read_toc_entry(self):
        entry = InfoDict(
            shader_hash     = self.read_long_long_le(), # FNV1A64 hash of name
            shader_name_ptr = self.read_long_long_le(),
            unk10           = self.read_long_le(),
            index_count     = self.read_long_le(),
            index_ptr       = self.read_long_long_le(),
            padding         = self.read(16 * 2),
        )
        self.check_unknown_fields('toc_entry', entry, {
            'unk10': 3,
            'index_count': 2,
        })
        return entry

    def read_index_entry(self):
        entry = InfoDict(
            shader_type     = self.read_long_le(),
            unk04           = self.read_long_le(),
            table_size      = self.read_long_le(),
            item_count      = self.read_long_le(),
            table_ptr       = self.read_long_long_le(),
            item_ptr        = self.read_long_long_le(),
            padding         = self.read(16 * 2),
        )
        self.check_unknown_fields('index_entry', entry, {
            'shader_type': {1, 2},
            'unk04': 0,
        })
        return entry

    def read_shader_entry(self):
        entry = InfoDict(
            shader_size     = self.read_long_le(),
            binding_count   = self.read_long_le(),
            unk08           = self.read_long_le(),
            unk0c           = self.read_long_le(),
            shader_ptr      = self.read_long_long_le(),
            binding_ptr     = self.read_long_long_le(),
            padding         = self.read(16 * 2),
        )
        self.check_unknown_fields('shader_entry', entry, {
            'unk08': 0,
            'unk0c': 0,
        })
        return entry

    def read_binding_entry(self):
        entry = InfoDict(
            id_code         = self.read_long_le(), # attribute offset in shader data
            unk04           = self.read_long_le(),
            id_hash         = self.read_long_long_le(), # FNV1A64 hash of attribute name
        )
        self.check_unknown_fields('binding_entry', entry, {
            'unk04': 1,
        })
        return entry

    def read_data(self):
        self.vc_game = 4
        self.is_dxbc = (self.header_unk[0] == 0xb)

        self.seek(self.header_length)
        self.toc_header = self.read_toc_header()
        self.entries = [ self.read_toc_entry() for i in range(self.toc_header.count) ]

        for entry in self.entries:
            self.follow_ptr(entry.shader_name_ptr)
            entry.shader_name = self.read_string()
            self.follow_ptr(entry.index_ptr)
            entry.data_index = [ self.read_index_entry() for i in range(entry.index_count) ]

            for index in entry.data_index:
                self.follow_ptr(index.table_ptr)
                index.table = [ self.read_long_le() for i in range(index.table_size) ]
                self.follow_ptr(index.item_ptr)
                index.shaders = [ self.read_shader_entry() for i in range(index.item_count) ]

                for shader in index.shaders:
                    self.follow_ptr(shader.shader_ptr)
                    shader.shader_data = self.read(shader.shader_size)
                    self.follow_ptr(shader.binding_ptr)
                    shader.bindings = [ self.read_binding_entry() for i in range(shader.binding_count) ]


class Valk4ATOM(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    # Contains ACBC blocks
    # Music?
    pass


class Valk4ACBC(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    # Music?
    pass


class Valk4CDRL(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    # Something to do with fonts?
    pass


class Valk4XLSB(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    # Contains CHNK blocks
    # Has different internal structure from most blocks.

    def container_func(self):
        pass

    def read_meta(self):
        self.seek(0)
        self.ftype = self.read(4).decode('ascii')
        if DEBUG:
            print("Creating", self.ftype)
        import os
        self.seek(0, os.SEEK_END)
        self.total_length = self.tell()
        self.header_length = 0


class Valk4CRBP(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    # Contains CRBD blocks
    # Body parts?
    pass


class Valk4SACC(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    # Contains SACC, SAC, and VSTD blocks
    pass


class Valk4ALBD(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
    pass


class Valk4EFSC(ValkFile):
    # Unknown block found in Valkyria Chronicles 4
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
    'ATUD': Valk4ATUD,
    'NSEN': Valk4NSEN,
    'REXP': Valk4REXP,
    'HSPK': Valk4HSPK,
    'KSPK': Valk4KSPK,
    'KSPP': Valk4KSPP,
    'ATOM': Valk4ATOM,
    'ACBC': Valk4ACBC,
    'CDRL': Valk4CDRL,
    'XLSB': Valk4XLSB,
    'CRBP': Valk4CRBP,
    'SACC': Valk4SACC,
    'ALBD': Valk4ALBD,
    'EFSC': Valk4EFSC,
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
