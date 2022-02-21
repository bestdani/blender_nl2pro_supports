import dataclasses
import os
import pathlib
import typing
import xml.etree.ElementTree
from _elementtree import Element
from typing import (
    Callable, Generic, Literal, Optional, Tuple, TypeVar,
)

import bpy
import bmesh
import bpy_types
import mathutils
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

TO_NL2 = mathutils.Matrix(
    ((+1.0, +0.0, +0.0, +0.0),
     (+0.0, +0.0, -1.0, +0.0),
     (+0.0, +1.0, +0.0, +0.0),
     (+0.0, +0.0, +0.0, +1.0))
)

T = TypeVar('T')


class OptionalNode(Generic[T]):
    pass


class OptionalAttribute(Generic[T]):
    @staticmethod
    def execute_search(element: Element, name: str):
        print(name)


class Attribute(Generic[T]):
    @staticmethod
    def execute_search(element: Element, name: str):
        print(name)


class XmlMetadata:
    """
        supported annotations
        type
        Optional[type]
        Optional[Literal[type]]
        Optional[ColorAttribute]
    """

    @classmethod
    def from_xml_element(cls, beam: Element) -> 'cls':
        data = cls()
        for field, annotated_type in data.__annotations__.items():
            node_as_parameter = field in cls._SUB_ATTRIBUTES

            if isinstance(annotated_type, type):
                is_optional = False
            else:
                is_optional = True
                # in this case it can only be assumed to be Optional for now
                optional_type = annotated_type.__args__[0]
                if isinstance(optional_type, type):
                    annotated_type = optional_type
                elif optional_type.__origin__ is typing.Literal:
                    optional_type = optional_type.__args__[0]
                    if optional_type is True:
                        node_as_parameter = True

                        def optional_type(x) -> bool:
                            return x is not None
                    else:
                        optional_type = type(optional_type)
                    annotated_type = optional_type

            is_attribute = field in cls._ATTRIBUTES
            if is_attribute:
                value = parse_attribute(
                    beam, annotated_type, field, is_optional
                )
            else:
                value = parse_node(
                    beam, annotated_type, field, is_optional,
                    node_as_parameter
                )
            data.__setattr__(field, value)

        return data


@dataclasses.dataclass
class VertData:
    type: Literal['rascnode', 'beamnode', 'footer']

    def insert_metadata(self, vert, key):
        vert[key] = str(type).encode('utf-8')


@dataclasses.dataclass
class BeamNodeData:
    start_vertex: int
    end_vertex: int

    def insert_metadata(self, vert, keys: dict) -> bytes:
        data = dataclasses.asdict(self)
        for key, value in data.items():
            vert[keys[key]] = str(value).encode('utf-8')


class FooterData(XmlMetadata):
    contype: int
    basetype: int
    rotation: float
    height_above_terrain: float
    size: Optional[float]

    _ATTRIBUTES = {'contype', 'basetype'}
    _SUB_ATTRIBUTES = set()

    def insert_metadata(self, vert, keys: dict) -> bytes:
        data = self.__dict__
        for key, value in data.items():
            vert[keys[key]] = str(value).encode('utf-8')


class PrefabData(XmlMetadata):
    center_rails_coord: float
    custom_track_index: int
    path: str

    _ATTRIBUTES = {'center_rails_coord', 'custom_track_index', 'path'}
    _SUB_ATTRIBUTES = set()

    def set_properties(self, prefab):
        prefab.path = self.path
        prefab.center_rails_coord = self.center_rails_coord
        prefab.custom_track_index = self.custom_track_index


class RascData(XmlMetadata):
    type: int
    center_rails_coord: float
    custom_track_index: int
    size: Optional[float]

    _ATTRIBUTES = {'type', 'center_rails_coord', 'custom_track_index', 'size'}
    _SUB_ATTRIBUTES = set()

    def set_properties(self, rasc):
        rasc.type = self.type
        rasc.center_rails_coord = self.center_rails_coord
        rasc.custom_track_index = self.custom_track_index
        if self.size:
            rasc.size = self.size


class ColorAttribute:
    r: float
    g: float
    b: float

    def __init__(self, element: Element):
        self.r = float(element.attrib['r'])
        self.g = float(element.attrib['g'])
        self.b = float(element.attrib['b'])


class BeamData(XmlMetadata):
    type: int
    size1: Optional[float]
    size2: Optional[float]
    rotation: Optional[float]
    start_extra_length: Optional[float]
    end_extra_length: Optional[float]
    offset_rel_x: Optional[float]
    offset_abs_y1: Optional[float]
    offset_abs_y2: Optional[float]
    colormode_custom: Optional[ColorAttribute]
    colormode_handrails: Optional[Literal[True]]
    colormode_catwalk: Optional[Literal[True]]
    colormode_mainspine: Optional[Literal[True]]
    colormode_unpaintedmetal: Optional[Literal[True]]
    lod: Optional[Literal['high', 'medium', 'low', 'lowest']]
    open_start_cap: Optional[Literal[True]]
    open_end_cap: Optional[Literal[True]]
    open_caps_for_lods: Optional[Literal[True]]
    dim_tunnel: Optional[Literal[True]]
    display_bolts: Optional[Literal[True]]

    _ATTRIBUTES = {'type', 'size1', 'size2'}
    _SUB_ATTRIBUTES = {'colormode_custom', }

    def __init__(self):
        self.type = 0
        self.size1 = None
        self.size2 = None
        self.rotation = None
        self.start_extra_length = None
        self.end_extra_length = None
        self.offset_rel_x = None
        self.offset_abs_y1 = None
        self.offset_abs_y2 = None
        self.colormode_custom = None
        self.colormode_handrails = None
        self.colormode_catwalk = None
        self.colormode_mainspine = None
        self.colormode_unpaintedmetal = None
        self.lod = None
        self.open_start_cap = None
        self.open_end_cap = None
        self.open_caps_for_lods = None
        self.dim_tunnel = None
        self.display_bolts = None

    def insert_metadata(self, edge, keys: dict) -> bytes:
        data = self.__dict__
        if self.colormode_custom:
            data['colormode_custom'] = self.colormode_custom.__dict__
        for key, value in data.items():
            edge[keys[key]] = str(value).encode('utf-8')


def parse_attribute(
        element: Element, annotated_type: Callable[[Generic[T]], T],
        name: str,
        is_optional: bool
) -> T:
    value = element.attrib.get(name)
    if value:
        return annotated_type(value)
    elif is_optional:
        return None
    else:
        raise ValueError(f"Required argument is missing: {name}")


def parse_node(
        element: Element, annotated_type: Callable[[Generic[T]], T],
        name: str,
        is_optional: bool,
        node_as_parameter: bool
) -> T:
    node: Optional[Element] = element.find(name)
    if node is not None:
        if node_as_parameter:
            return annotated_type(node)
        else:
            return annotated_type(node.text)
    elif is_optional:
        return None
    else:
        raise ValueError(f"Required node is missing: {name}")


def parse_vec(text: str) -> mathutils.Vector:
    return TO_NL2 @ mathutils.Vector(float(pos) for pos in text.split())


def create_edge_with_metadata(
        bm: bmesh, vertices: Tuple, beam: Element,
        metadata_layers: dict
):
    edge = bm.edges.new(vertices)
    BeamData.from_xml_element(beam).insert_metadata(edge, metadata_layers)


def import_supports(supports_file: os.PathLike):
    tree = xml.etree.ElementTree.parse(supports_file)
    root = tree.getroot()
    main_supports = root.find('supports')

    import_name = pathlib.Path(supports_file).with_suffix('').name
    support_object = create_support_object(import_name, main_supports)
    support_object.nl2_support.is_support_object = True

    support_object_matrix_inverted = support_object.matrix_local.inverted()
    for prefab in main_supports.findall('prefab'):
        prefab_data = PrefabData.from_xml_element(prefab)
        prefab_object = create_support_object(
            f'prefab_{prefab_data.custom_track_index}_'
            f'{prefab_data.center_rails_coord}',
            prefab.find('atomization/supports')
        )
        prefab_data.set_properties(prefab_object.nl2_support.prefab)
        matrix_local = support_object_matrix_inverted @ \
                       prefab_object.matrix_local
        prefab_object.parent = support_object
        prefab_object.matrix_local = matrix_local


Matrix4x4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


def create_support_object(
        import_name: str,
        supports_root: Element,
        matrix_world_transposed: Matrix4x4 = (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
        )
) -> bpy_types.Object:
    mesh = bpy.data.meshes.new(import_name)
    supports_object = bpy.data.objects.new(import_name, mesh)
    supports_object.matrix_world = matrix_world_transposed
    transformation_matrix = supports_object.matrix_world.inverted()
    bpy.context.layer_collection.collection.objects.link(supports_object)

    bm = bmesh.new()
    bm.from_mesh(mesh)

    nodes = {}
    prefabs = {}
    rasc_list = []
    freenodes = {}
    beamnodes = {}
    footers = {}
    edge_beam_metadata_keys = {
        k: bm.edges.layers.string.new(f'beam.{k}') for k in
        BeamData.__annotations__.keys()
    }

    vert_footer_metadata_keys = {
        k: bm.verts.layers.string.new(f'footer.{k}') for k in
        FooterData.__annotations__.keys()
    }

    vert_beamnode_metadata_keys = {
        k: bm.verts.layers.string.new(f'beamode.{k}') for k in
        BeamNodeData.__annotations__.keys()
    }

    vert_type_metadata = bm.verts.layers.string.new('vert.type')

    def create_support_mesh(supports: Element) -> Tuple[dict, list]:
        nodes = {}
        verts = []

        for track_node in supports.iterfind('rasc'):
            add_track_connector(nodes, track_node, verts)

        for sub_node in supports.iterfind('freenode'):
            add_free_node(nodes, sub_node, verts)

        for sub_node in supports.iterfind('footernode'):
            add_footer_node(nodes, sub_node, verts)

        for beam in supports.iterfind('beam'):
            add_beam(beam, nodes, verts)

        return nodes, verts

    def add_beam(beam, nodes, verts):
        """
            creates a beam either as simple as:
                start_vert -> end_vert

            or if beam nodes are present as a chain like:
                start_vert -> bn1 -> bn2 -> bn3 -> end_vert
        """
        start = beam.attrib['start']
        end = beam.attrib['end']
        if start in nodes and end in nodes:
            start_vert = nodes[start]
            end_vert = nodes[end]

            beam_nodes = find_beam_nodes(beam, end_vert, start_vert, verts)
            for node in beam_nodes:
                vert = node[0]
                VertData('beamnode').insert_metadata(vert, vert_type_metadata)
                BeamNodeData(start_vert.index, end_vert.index).insert_metadata(
                    vert, vert_beamnode_metadata_keys
                )

            # if no sub nodes are present latest_connected_vert = start_vec
            latest_connected_vert = build_beam_node_chain(
                beam, beam_nodes, nodes, start_vert
            )

            create_edge_with_metadata(
                bm, (latest_connected_vert, end_vert), beam,
                edge_beam_metadata_keys
            )

    def build_beam_node_chain(beam, beam_nodes, nodes, start_vert):
        """
            connect start_vert and all sub_vert chains (if present) first
            vert will be offset to next sub_vert and subsequently set to the
            last sub_vert (or left at the start_vert)
        """
        current_start_vert = start_vert
        for sub_node in beam_nodes:
            vert, id, type, pos = sub_node
            beamnodes[id] = (vert, type)
            nodes[id] = vert
            create_edge_with_metadata(
                bm, (current_start_vert, vert), beam,
                edge_beam_metadata_keys
            )
            current_start_vert = vert
        return current_start_vert

    def find_beam_nodes(beam, end_vert, start_vert, verts):
        # find beam nodes and store as sub_nodes
        beam_nodes = []
        for beam_node in beam.iterfind('beamnode'):
            id = beam_node.attrib['id']
            pos = float(beam_node.attrib['pos'])
            type = int(beam_node.attrib['type'])
            vert = bm.verts.new(start_vert.co.lerp(end_vert.co, pos))
            beam_nodes.append((vert, id, type, pos))
            verts.append(vert)
        # sort by pos value
        beam_nodes.sort(key=lambda x: x[3])
        return beam_nodes

    def add_footer_node(nodes, footer_node, verts: list):
        id = footer_node.attrib['id']
        pos = footer_node.find('pos')
        vert = bm.verts.new(transformation_matrix @ parse_vec(pos.text))

        VertData('footer').insert_metadata(vert, vert_type_metadata)
        FooterData.from_xml_element(footer_node).insert_metadata(
            vert, vert_footer_metadata_keys
        )

        nodes[id] = vert
        verts.append(vert)

    def add_track_connector(nodes, rasc_node, verts):
        track_connector_verts = [
            add_track_connector_node(nodes, sub_node)
            for sub_node in rasc_node.iterfind('subnode')
        ]
        verts.extend(track_connector_verts)

        rasc = RascData.from_xml_element(rasc_node)
        rasc_list.append((rasc, track_connector_verts))

    def add_track_connector_node(nodes, sub_node):
        id = sub_node.attrib['id']
        pos = sub_node.find('pos')
        vert = bm.verts.new(transformation_matrix @ parse_vec(pos.text))
        VertData('rascnode').insert_metadata(vert, vert_type_metadata)

        nodes[id] = vert
        return vert

    def add_free_node(nodes, sub_node, verts):
        id = sub_node.attrib['id']
        pos = sub_node.find('pos')
        vert = bm.verts.new(transformation_matrix @ parse_vec(pos.text))

        freenodes[id] = (vert,)
        nodes[id] = vert
        verts.append(vert)

    def hook_rasc_to_rasc_object(rasc_data: Tuple[RascData, list]):
        rasc: RascData
        rasc, rasc_verts = rasc_data

        rasc_object_name = f'rasc_{rasc.custom_track_index}_' \
                           f'{rasc.center_rails_coord}'
        rasc_object = bpy.data.objects.new(rasc_object_name, None)
        rasc.set_properties(rasc_object.nl2_support.rasc)
        bpy.context.layer_collection.collection.objects.link(rasc_object)
        rasc_object.parent = supports_object

        # interpolate position for now
        position = mathutils.Vector()
        tcv_indices = []
        for v in rasc_verts:
            tcv_indices.append(v.index)
            position += v.co
        position /= len(rasc_verts)

        rasc_object_matrix = rasc_object.matrix_world.Translation(position)
        rasc_object.matrix_local = rasc_object_matrix

        hook_modifier = supports_object.modifiers.new(
            rasc_object_name, 'HOOK'
        )
        hook_modifier.object = rasc_object
        hook_modifier.matrix_inverse = rasc_object_matrix.inverted()
        hook_modifier.vertex_indices_set(tcv_indices)

    new_nodes, verts = create_support_mesh(supports_root)
    nodes.update(new_nodes)
    bm.to_mesh(mesh)
    bpy.context.view_layer.update()

    for rasc in rasc_list:
        hook_rasc_to_rasc_object(rasc)

    bm.free()

    return supports_object


def execute_import(
        context, filepath, atomize_prefabs: bool,
        blender_track_connectors: bool
):
    import_supports(filepath)

    return {'FINISHED'}


class ImportNl2Supports(Operator, ImportHelper):
    """Import Supports exported from No Limits 2 Professional"""
    bl_idname = "import_nl2.supports"  # important since its how
    # bpy.ops.import_nl2.supports is constructed
    bl_label = "Import NL2 Supports"

    # ImportHelper mixin class uses this
    filename_ext = ".xml"

    filter_glob: StringProperty(
        default="*.xml",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    atomized_prefabs: BoolProperty(
        name="Atomized Prefabs",
        description="Uses atomized prefabs on import",
        default=True,
    )

    blender_track_connectors: BoolProperty(
        name="Replicate Track Connectors",
        description="Add Track Connectors to some curve (must be selected on "
                    "import)",
        default=True,
    )

    def execute(self, context):
        return execute_import(
            context, self.filepath, self.atomized_prefabs,
            self.blender_track_connectors
        )


# Only needed if you want to add into a dynamic menu
def menu_func_import(self, context):
    self.layout.operator(
        ImportNl2Supports.bl_idname,
        text="NoLimits 2 Professional Supports (*.xml)"
    )


class RascProperties(bpy.types.PropertyGroup):
    type: bpy.props.IntProperty(
        name="rail support connector type"
    )
    center_rails_coord: bpy.props.FloatProperty(
        name="rail support connector center of rails coordinate"
    )
    custom_track_index: bpy.props.IntProperty(
        name="track index to which the connector is attached"
    )
    # use .is_property_set('size') to check if it is set
    size: bpy.props.FloatProperty(
        name="optional rail support connector size"
    )


class PrefabProperties(bpy.types.PropertyGroup):
    center_rails_coord: bpy.props.FloatProperty(
        name="rail support connector center of rails coordinate"
    )
    custom_track_index: bpy.props.IntProperty(
        name="track index to which the connector is attached"
    )
    path: bpy.props.StringProperty(
        name="internal NoLimits 2 rasc path"
    )


class SupportProperties(bpy.types.PropertyGroup):
    is_support_object: bpy.props.BoolProperty(
        name="indicates if this is some support object"
    )
    rasc: bpy.props.PointerProperty(
        type=RascProperties,
        name="Rail Support Connector Settings"
    )
    prefab: bpy.props.PointerProperty(
        type=PrefabProperties,
        name="Prefab Settings"
    )


def register_properties():
    bpy.utils.register_class(RascProperties)
    bpy.utils.register_class(PrefabProperties)
    bpy.utils.register_class(SupportProperties)
    bpy.types.Object.nl2_support = bpy.props.PointerProperty(
        type=SupportProperties
    )


def unregister_properties():
    bpy.utils.unregister_class(RascProperties)
    bpy.utils.unregister_class(PrefabProperties)
    bpy.utils.unregister_class(SupportProperties)


# Register and add to the "file selector" menu (required to use F3 search
# "Text Import Operator" for quick access)
def register():
    register_properties()
    bpy.utils.register_class(ImportNl2Supports)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    unregister_properties()
    bpy.utils.unregister_class(ImportNl2Supports)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.import_nl2.supports('INVOKE_DEFAULT')
