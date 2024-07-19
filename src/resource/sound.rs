use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use fyrox_resource::io::FsResourceIo;
use fyrox_sound::buffer::{DataSource, SoundBufferResource, SoundBufferResourceExtension};
#[allow(unused_imports)]
use crate::core::prelude::*;

use fyrox_sound::context::SoundContext;
use fyrox_sound::engine::SoundEngine;
use fyrox_sound::pool::Handle;
use fyrox_sound::source::{SoundSource, SoundSourceBuilder, Status};

#[derive(Clone)]
struct SoundInner {
    ctx: SoundContext,
    handle: Handle<SoundSource>,
}
#[derive(Clone, Default)]
pub struct Sound {
    inner: Option<SoundInner>,
}

impl Sound {
    pub fn play(&mut self) {
        let inner = self.inner.clone().unwrap();
        let mut state = inner.ctx.state();
        let source = state.source_mut(inner.handle);
        source.stop()
            .expect("should only be fallible for streaming buffers (see source)");
        source.play();
    }

    pub fn play_loop(&mut self) {
        let inner = self.inner.clone().unwrap();
        let mut state = inner.ctx.state();
        let source = state.source_mut(inner.handle);
        source.set_looping(true);
        source.stop()
            .expect("should only be fallible for streaming buffers (see source)");
        source.play();
    }

    pub fn stop(&mut self) {
        let inner = self.inner.clone().unwrap();
        let mut state = inner.ctx.state();
        let source = state.source_mut(inner.handle);
        source.stop()
            .expect("should only be fallible for streaming buffers (see source)");
    }

    pub fn is_playing(&self) -> bool {
        let inner = self.inner.clone().unwrap();
        let mut state = inner.ctx.state();
        let source = state.source_mut(inner.handle);
        source.status() == Status::Playing
    }
}

struct SoundHandlerInner {
    loaded_files: BTreeMap<String, SoundBufferResource>,
}

#[derive(Clone)]
pub struct SoundHandler {
    _engine: SoundEngine,
    ctx: SoundContext,
    inner: Arc<Mutex<SoundHandlerInner>>,
}

impl SoundHandler {
    pub fn new() -> Result<Self> {
        let engine = SoundEngine::new()
            // SoundEngine::new() returns Box<dyn Error> which is not Send, so expect() here.
            .map_err(|err| anyhow!("fyrox-sound error: SoundEngine::new(): {:?}", err))?;
        let ctx = SoundContext::new();
        engine.state().add_context(ctx.clone());
        Ok(Self {
            _engine: engine,
            ctx,
            inner: Arc::new(Mutex::new(SoundHandlerInner {
                loaded_files: BTreeMap::new(),
            }))
        })
    }

    pub fn wait_load_file(&mut self, filename: String) -> Result<Sound> {
        let sound_buffer = {
            let mut inner = self.inner.lock().unwrap();
            match inner.loaded_files.get(&filename) {
                Some(buffer) => buffer.clone(),
                None => {
                    let buffer = SoundBufferResource::new_generic(fyrox_sound::futures::executor::block_on(
                        DataSource::from_file(&filename, &FsResourceIo))
                        .map_err(|err| anyhow!("fyrox-sound error: DataSource::from_file(): {:?}", err))?
                    ).map_err(|err| anyhow!("fyrox-sound error: SoundBufferResource::new_generic(): {:?}", err))?;
                    inner.loaded_files.insert(filename, buffer.clone());
                    buffer
                }
            }
        };

        let source = SoundSourceBuilder::new()
            .with_buffer(sound_buffer)
            .with_status(Status::Stopped)
            // Ensure that no spatial effects will be applied.
            .with_spatial_blend_factor(0.0)
            .build()?;
        let inner = Some(SoundInner {
            ctx: self.ctx.clone(),
            handle: self.ctx.state().add_source(source),
        });
        Ok(Sound { inner })
    }
}
