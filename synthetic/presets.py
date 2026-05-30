"""Pre-built synthetic scenes for tests and the demo."""

from __future__ import annotations

from scene_extractor.schema import SceneGraph
from synthetic.scene_builder import SyntheticScene


def clevr_like() -> SceneGraph:
    """A CLEVR-style scene: 3 geometric objects with color, size, material."""
    return (
        SyntheticScene()
        .add_object("obj_0", "cube", color="red", size="large", material="metal")
        .add_object("obj_1", "sphere", color="blue", size="small", material="rubber")
        .add_object("obj_2", "cylinder", color="green", size="medium", material="metal")
        .add_relation("obj_0", "left_of", "obj_1")
        .add_relation("obj_1", "left_of", "obj_2")
        .add_relation("obj_2", "above", "obj_0")
        .to_scene_graph()
    )


def kitchen() -> SceneGraph:
    """Everyday-object scene with material and color attributes."""
    return (
        SyntheticScene()
        .add_object("obj_0", "table", color="brown", material="wood")
        .add_object("obj_1", "cup", color="white", material="ceramic")
        .add_object("obj_2", "bottle", color="green", material="glass")
        .add_object("obj_3", "apple", color="red")
        .add_relation("obj_1", "on_top_of", "obj_0")
        .add_relation("obj_2", "on_top_of", "obj_0")
        .add_relation("obj_3", "next_to", "obj_1")
        .add_relation("obj_1", "left_of", "obj_2")
        .to_scene_graph()
    )


def office() -> SceneGraph:
    """Office scene exercising same_color and same_material queries."""
    return (
        SyntheticScene()
        .add_object("obj_0", "chair", color="black", material="metal")
        .add_object("obj_1", "chair", color="black", material="metal")
        .add_object("obj_2", "desk", color="brown", material="wood")
        .add_object("obj_3", "monitor", color="black", material="plastic")
        .add_relation("obj_0", "next_to", "obj_2")
        .add_relation("obj_1", "next_to", "obj_2")
        .add_relation("obj_3", "on_top_of", "obj_2")
        .to_scene_graph()
    )


def single_object() -> SceneGraph:
    """Minimal scene: one red cube. Edge case for existence/count queries."""
    return (
        SyntheticScene()
        .add_object("obj_0", "cube", color="red", size="large", material="metal")
        .to_scene_graph()
    )


def empty_scene() -> SceneGraph:
    """Empty scene. Edge case: every existence query returns False."""
    return SyntheticScene().to_scene_graph()


ALL_PRESETS = {
    "clevr_like": clevr_like,
    "kitchen": kitchen,
    "office": office,
    "single_object": single_object,
    "empty_scene": empty_scene,
}
