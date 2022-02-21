import xml.etree.ElementTree
from io_import_supports import BeamData, PrefabData, RascData
import unittest


class TestPrefabData(unittest.TestCase):
    def test_all_attributes(self):
        element = xml.etree.ElementTree.fromstring(
            """
             <prefab center_rails_coord="3.35002" custom_track_index="1" 
             path="intern:data/prefabs/Track Default Double.nl2prefab">
            </prefab>
            """
        )
        prefab = PrefabData.from_xml_element(element)
        self.assertEqual(prefab.center_rails_coord, 3.35002)
        self.assertEqual(prefab.custom_track_index, 1)
        self.assertEqual(
            prefab.path, "intern:data/prefabs/Track Default Double.nl2prefab"
        )

    def test_missing_attribute(self):
        element = xml.etree.ElementTree.fromstring(
            """
             <prefab center_rails_coord="3.35002" 
             custom_track_index="1"></prefab>
            """
        )
        with self.assertRaises(ValueError):
            PrefabData.from_xml_element(element)


class TestFooterData(unittest.TestCase):
    def test_all_attributes(self):
        element = xml.etree.ElementTree.fromstring(
            """
                <rasc type="258" center_rails_coord="1.9685" 
                    custom_track_index="0" size="0.508">
                  <subnode id="0">
                    <pos>2.62297 3.95445 15.5684</pos>
                  </subnode>
                  <subnode id="1">
                    <pos>2.62297 3.45445 15.5684</pos>
                  </subnode>
                </rasc>
             """
        )
        rasc = RascData.from_xml_element(element)
        self.assertEqual(rasc.type, 258)
        self.assertEqual(rasc.center_rails_coord, 1.9685)
        self.assertEqual(rasc.custom_track_index, 0)
        self.assertEqual(rasc.size, 0.508)

    def test_required_attributes(self):
        element = xml.etree.ElementTree.fromstring(
            """
                <rasc type="0" center_rails_coord="2.8249" 
                    custom_track_index="1">
                  <subnode id="2">
                    <pos>0.759448 6.2 15.6851</pos>
                  </subnode>
                </rasc>
             """
        )
        rasc = RascData.from_xml_element(element)
        self.assertEqual(rasc.type, 0)
        self.assertEqual(rasc.center_rails_coord, 2.8249)
        self.assertEqual(rasc.custom_track_index, 1)
        self.assertEqual(rasc.size, None)


class TestBeamData(unittest.TestCase):
    def test_full_example(self):
        element = xml.etree.ElementTree.fromstring(
            """
                 <beam start="4" end="3" type="5" size1="0.219075" size2="0.5">
                   <rotation>0.261799</rotation>
                   <start_extra_length>0.2</start_extra_length>
                   <end_extra_length>0.4</end_extra_length>
                   <offset_rel_x>0.6</offset_rel_x>
                   <offset_abs_y1>0.8</offset_abs_y1>
                   <offset_abs_y2>1</offset_abs_y2>
                   <colormode_custom r="1" g="0.0823529" b="0.0823529"/>
                   <lod>lowest</lod>
                   <open_start_cap/>
                   <open_end_cap/>
                   <open_caps_for_lods/>
                   <dim_tunnel/>
                   <display_bolts/>
                 </beam>
             """
        )
        beam = BeamData.from_xml_element(element)
        self.assertEqual(beam.type, 5)
        self.assertEqual(beam.size1, 0.219075)
        self.assertEqual(beam.size2, 0.5)
        self.assertEqual(beam.rotation, 0.261799)
        self.assertEqual(beam.start_extra_length, 0.2)
        self.assertEqual(beam.end_extra_length, 0.4)
        self.assertEqual(beam.offset_rel_x, 0.6)
        self.assertEqual(beam.offset_abs_y1, 0.8)
        self.assertEqual(beam.offset_abs_y2, 1)
        self.assertEqual(
            beam.colormode_custom.__dict__,
            {'r': 1, 'g': 0.0823529, 'b': 0.0823529}
        )
        self.assertEqual(beam.colormode_handrails, None)
        self.assertEqual(beam.colormode_catwalk, None)
        self.assertEqual(beam.colormode_mainspine, None)
        self.assertEqual(beam.colormode_unpaintedmetal, None)
        self.assertEqual(beam.lod, 'lowest')
        self.assertEqual(beam.open_start_cap, True)
        self.assertEqual(beam.open_end_cap, True)
        self.assertEqual(beam.open_caps_for_lods, True)
        self.assertEqual(beam.dim_tunnel, True)
        self.assertEqual(beam.display_bolts, True)

    def test_missing_attribute(self):
        element = xml.etree.ElementTree.fromstring(
            """
                 <beam start="4" end="3" size1="0.219075" size2="0.5"></beam>
             """
        )
        with self.assertRaises(ValueError):
            BeamData.from_xml_element(element)


if __name__ == '__main__':
    unittest.main()
