use crate::core::ObjectTypeEnum;
use crate::core::prelude::*;
use crate::core::render::VertexDepth;
use crate::resource::texture::{MaterialId, Texture};
use crate::util::collision::CompoundCollider;
use crate::util::linalg::{Edge2i, Vec2i};
use glongge_derive::{partially_derive_scene_object, register_scene_object};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::rc::Rc;

#[derive(Clone)]
pub struct Tile {
    id: usize,
    collision_id: Option<usize>,
    tex_top_left: Vec2i,
    depth: Rc<RefCell<VertexDepth>>,
}

impl Tile {
    pub fn set_depth(&self, depth: VertexDepth) {
        *self.depth.borrow_mut() = depth;
    }
}

#[derive(Clone)]
struct PolygonBuilder {
    edges: BTreeSet<Edge2i>,
    vertices: BTreeSet<Vec2i>,
    by_src: BTreeMap<Vec2i, Vec2i>,
    by_dest: BTreeMap<Vec2i, Vec2i>,
}

impl PolygonBuilder {
    fn new() -> Self {
        Self {
            edges: BTreeSet::new(),
            vertices: BTreeSet::new(),
            by_src: BTreeMap::new(),
            by_dest: BTreeMap::new(),
        }
    }

    fn add_edge(&mut self, edge: Edge2i) {
        if !self.edges.remove(&edge) && !self.edges.remove(&edge.reverse()) {
            self.edges.insert(edge);
            self.vertices.insert(edge.0);
            self.vertices.insert(edge.1);
            self.by_src.insert(edge.0, edge.1);
            self.by_dest.insert(edge.1, edge.0);
        }
    }

    fn contains_vertex(&self, v: Vec2i) -> bool {
        self.by_src.contains_key(&v) || self.by_dest.contains_key(&v)
    }

    fn is_simple(&self) -> bool {
        let original_src = *self.by_src.first_key_value().unwrap().0;
        let mut src = original_src;
        let mut visited = BTreeSet::new();
        for _i in 0..self.edges.len() {
            if visited.contains(&src) {
                return false;
            }
            visited.insert(src);
            src = *self.by_src.get(&src).unwrap();
        }
        check_eq!(src, original_src);
        true
    }

    fn build(&self) -> Option<Vec<Vec2i>> {
        let mut vertices = Vec::new();
        let original_src = *self.by_src.first_key_value().unwrap().0;
        let mut src = original_src;
        let mut visited = BTreeSet::new();
        for i in 0..self.edges.len() {
            if visited.contains(&src) {
                warn!(
                    "polygon contains hole: reached start at {i}/{} vertices",
                    self.edges.len()
                );
                return None;
            }
            vertices.push(src);
            visited.insert(src);
            src = *self.by_src.get(&src).unwrap();
        }
        check_eq!(src, original_src);
        Some(vertices)
    }
}

pub struct TilesetBuilder {
    name: String,
    filename: String,
    all_tiles: Vec<Tile>,
    all_polygons: Vec<Vec<Vec2i>>,
    collision_sets: Vec<Vec<&'static str>>,
    tile_size: i32,

    tile_map: BTreeMap<usize, BTreeSet<Vec2i>>,
    collision_edges: BTreeMap<usize, Vec<PolygonBuilder>>,
}

impl TilesetBuilder {
    pub fn new(tex_filename: impl AsRef<str>, tile_size: i32) -> TilesetBuilder {
        Self {
            name: "Tileset".to_string(),
            filename: tex_filename.as_ref().to_string(),
            all_tiles: Vec::new(),
            all_polygons: Vec::new(),
            collision_sets: Vec::new(),
            tile_size,
            tile_map: BTreeMap::new(),
            collision_edges: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn named(mut self, name: impl Display) -> Self {
        self.name = format!("Tileset [{name}]");
        self
    }

    pub fn create_tile(&mut self, tex_top_left: impl Into<Vec2i>) -> Tile {
        let id = self.all_tiles.len();
        let tile = Tile {
            id,
            collision_id: None,
            tex_top_left: tex_top_left.into(),
            depth: Rc::new(RefCell::new(VertexDepth::default())),
        };
        self.all_tiles.push(tile.clone());
        tile
    }
    pub fn create_tile_collision(
        &mut self,
        tex_top_left: impl Into<Vec2i>,
        emitting_tags: &Vec<&'static str>,
    ) -> Tile {
        check_false!(emitting_tags.is_empty());
        let id = self.all_tiles.len();
        #[allow(clippy::map_unwrap_or)] // borrowing issues with self.collision_sets
        let collision_id = self
            .collision_sets
            .iter()
            .enumerate()
            .find(|(_, t)| *t == emitting_tags)
            .map(|(i, _)| i)
            .unwrap_or_else(|| {
                let next_id = self.collision_sets.len();
                self.collision_sets.push(emitting_tags.clone());
                next_id
            });
        let tex_top_left = tex_top_left.into();
        let tile = Tile {
            id,
            collision_id: Some(collision_id),
            tex_top_left,
            depth: Rc::new(RefCell::new(VertexDepth::default())),
        };
        self.all_tiles.push(tile.clone());
        // TODO: support other polygons
        self.all_polygons.push(vec![
            Vec2i::zero(),
            Vec2i::right(),
            Vec2i::one(),
            Vec2i::down(),
        ]);
        tile
    }

    pub fn insert(&mut self, tile: &Tile, tile_coord: impl Into<Vec2i>) -> bool {
        let tile_coord = tile_coord.into();
        let rv = self.tile_map.entry(tile.id).or_default().insert(tile_coord);
        if let Some(collision_id) = tile.collision_id {
            let tile_vertices = self.all_polygons[tile.id]
                .iter()
                .map(|v| *v + tile_coord)
                .collect_vec();
            let entry = self.collision_edges.entry(collision_id).or_default();

            for poly in entry
                .iter_mut()
                .filter(|edges| tile_vertices.iter().any(|v| edges.contains_vertex(*v)))
            {
                let mut next_poly = poly.clone();
                for (u, v) in tile_vertices.iter().circular_tuple_windows() {
                    let edge = Edge2i(*u, *v);
                    next_poly.add_edge(edge);
                }
                if next_poly.is_simple() {
                    *poly = next_poly;
                    return rv;
                }
            }

            // No existing polygon found to add this edge to. Create a new one.
            let mut next_poly = PolygonBuilder::new();
            for (u, v) in tile_vertices.iter().circular_tuple_windows() {
                let edge = Edge2i(*u, *v);
                next_poly.add_edge(edge);
            }
            entry.push(next_poly);
        }
        rv
    }

    pub fn build<O: ObjectTypeEnum>(self) -> Container<O> {
        let texture_areas: BTreeMap<_, _> = self
            .all_tiles
            .iter()
            .map(|tile| {
                (
                    tile.id,
                    Rect::from_coords(
                        tile.tex_top_left.into(),
                        (tile.tex_top_left + self.tile_size * Vec2i::one()).into(),
                    ),
                )
            })
            .collect();
        let tiles = self
            .tile_map
            .into_iter()
            .flat_map(|(id, top_lefts)| {
                let tex_area = *texture_areas.get(&id).unwrap();
                let depth = *self.all_tiles[id].depth.borrow();
                let collision_id = self.all_tiles[id].collision_id;
                top_lefts.into_iter().map(move |top_left| {
                    let top_left = self.tile_size * top_left;
                    (
                        collision_id,
                        RenderableTile {
                            tex_area,
                            top_left,
                            depth,
                        },
                    )
                })
            })
            .collect_vec();
        let colliders = self
            .collision_edges
            .into_iter()
            .zip(self.collision_sets)
            .map(|((collision_id, polygons), emitting_tags)| {
                let collider = polygons
                    .into_iter()
                    .filter_map(|poly| {
                        poly.build().map(|vertices| {
                            CompoundCollider::decompose(
                                vertices
                                    .into_iter()
                                    .map(|v| (self.tile_size * v).into())
                                    .collect(),
                            )
                        })
                    })
                    .reduce(CompoundCollider::combined)
                    .map(CompoundCollider::into_generic)
                    .unwrap_or_default();
                GgInternalTileset {
                    tile_size: self.tile_size,
                    filename: self.filename.clone(),
                    texture: Texture::default(),
                    material_id: MaterialId::default(),
                    tiles: tiles
                        .iter()
                        .cloned()
                        .filter_map(|(id, tile)| {
                            if id == Some(collision_id) {
                                Some(tile)
                            } else {
                                None
                            }
                        })
                        .collect_vec(),
                    collider,
                    emitting_tags,
                }
                .into_wrapper()
            })
            .collect_vec();
        Container::new(self.name, colliders)
    }
}

#[derive(Clone)]
struct RenderableTile {
    tex_area: Rect,
    top_left: Vec2i,
    depth: VertexDepth,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct CollidableTile {
    pub pos: Vec2i,
    pub emitting_tags: Vec<&'static str>,
}

#[register_scene_object]
pub struct GgInternalTileset {
    tile_size: i32,
    filename: String,
    texture: Texture,
    material_id: MaterialId,
    tiles: Vec<RenderableTile>,

    collider: GenericCollider,
    emitting_tags: Vec<&'static str>,
}

impl GgInternalTileset {
    pub fn tiles(&self) -> Vec<CollidableTile> {
        self.tiles
            .iter()
            .map(|t| CollidableTile {
                pos: t.top_left / self.tile_size,
                emitting_tags: self.emitting_tags.clone(),
            })
            .collect_vec()
    }
}

#[partially_derive_scene_object]
impl<ObjectType: ObjectTypeEnum> SceneObject<ObjectType> for GgInternalTileset {
    fn type_name(&self) -> String {
        "TilesetSegment".to_string()
    }
    fn gg_type_enum(&self) -> ObjectType {
        ObjectType::gg_tileset()
    }

    fn on_load(
        &mut self,
        object_ctx: &mut ObjectContext<ObjectType>,
        resource_handler: &mut ResourceHandler,
    ) -> Result<Option<RenderItem>> {
        self.texture = resource_handler
            .texture
            .wait_load_file(self.filename.clone())?;
        let mut rv = RenderItem::default();
        for tile in &self.tiles {
            self.material_id = resource_handler
                .texture
                .material_from_texture(&self.texture, &tile.tex_area);
            let vertices = vertex::rectangle(
                (tile.top_left + self.tile_size * Vec2i::one() / 2).into(),
                (self.tile_size * Vec2i::one() / 2).into(),
            );
            rv = rv.concat(RenderItem {
                vertices: vertices.vertices,
                depth: tile.depth,
            });
        }
        object_ctx.add_child(CollisionShape::from_collider(
            self.collider.clone(),
            &self.emitting_tags,
            &[],
        ));
        Ok(Some(rv))
    }

    fn emitting_tags(&self) -> Vec<&'static str> {
        self.emitting_tags.clone()
    }

    fn as_renderable_object(&mut self) -> Option<&mut dyn RenderableObject<ObjectType>> {
        Some(self)
    }
}

impl<ObjectType: ObjectTypeEnum> RenderableObject<ObjectType> for GgInternalTileset {
    fn shader_execs(&self) -> Vec<ShaderExec> {
        vec![ShaderExec {
            shader_id: get_shader(SpriteShader::name()),
            material_id: self.material_id,
            ..Default::default()
        }]
    }
}

use crate::core::builtin::Container;
use crate::shader::{Shader, SpriteShader, get_shader, vertex};
pub use GgInternalTileset as Tileset;
