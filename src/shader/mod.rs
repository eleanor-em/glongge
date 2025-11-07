pub mod vertex;

#[allow(unused)]
#[derive(Copy, Clone, Debug, Default)]
pub struct ShaderId(u32);

#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct SpriteVertex {
    pub position: [f32; 2],
    pub translation: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub material_id: u32,
    pub blend_col: [f32; 4],
    pub clip_min: [f32; 2],
    pub clip_max: [f32; 2],
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GuiVertex {
    pub a_pos: [f32; 2],
    pub a_tc: [f32; 2],
    pub a_srgba: [u8; 4],
}
