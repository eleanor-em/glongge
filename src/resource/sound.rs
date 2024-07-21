use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex}
};

use fyrox_sound::{
    buffer::{
        loader::FsResourceIo,
        DataSource,
        SoundBufferResource,
        SoundBufferResourceExtension
    },
    context::SoundContext,
    engine::SoundEngine,
    pool::Handle,
    source::{SoundSource, SoundSourceBuilder, Status}
};
use rand::{Rng, thread_rng};
use crate::core::linalg;

#[allow(unused_imports)]
use crate::core::prelude::*;

#[derive(Clone)]
struct SoundInner {
    ctx: SoundContext,
    handle: Handle<SoundSource>,
}
#[derive(Clone, Default)]
pub struct Sound {
    inner: Option<SoundInner>,
    is_looping: bool,
}

impl Sound {
    pub fn play_shifted(&mut self, mag: f64) {
        if let Some(inner) = self.inner.as_ref() {
            let mut state = inner.ctx.state();
            let source = state.source_mut(inner.handle);
            source.stop()
                .expect("should only be fallible for streaming buffers (see source)");
            let mut rng = thread_rng();
            source.set_pitch(linalg::eerp(1. - mag, 1. + mag, rng.gen_range(0.0..1.0)));
            source.play();
            source.set_pitch(1.);
        } else {
            warn!("tried to play non-loaded sound");
        }
    }
    pub fn play(&mut self) {
        if let Some(inner) = self.inner.as_ref() {
            let mut state = inner.ctx.state();
            let source = state.source_mut(inner.handle);
            source.stop()
                .expect("should only be fallible for streaming buffers (see source)");
            source.play();
        } else {
            warn!("tried to play non-loaded sound");
        }
    }

    pub fn play_loop(&mut self) {
        if let Some(inner) = self.inner.as_ref() {
            let mut state = inner.ctx.state();
            let source = state.source_mut(inner.handle);
            source.set_looping(true);
            source.stop()
                .expect("should only be fallible for streaming buffers (see source)");
            source.play();
            self.is_looping = true;
        } else {
            warn!("tried to play non-loaded sound");
        }
    }

    pub fn stop(&mut self) {
        if let Some(inner) = self.inner.as_ref() {
            let mut state = inner.ctx.state();
            let source = state.source_mut(inner.handle);
            source.stop()
                .expect("should only be fallible for streaming buffers (see source)");
            self.is_looping = false;
        } else {
            warn!("tried to stop non-loaded sound");
        }
    }

    pub fn is_playing(&self) -> bool {
        self.inner.as_ref()
            .is_some_and(|inner| {
                let mut state = inner.ctx.state();
                let source = state.source_mut(inner.handle);
                source.status() == Status::Playing
            })
    }
}

impl Drop for Sound {
    fn drop(&mut self) {
        if self.is_looping {
            self.stop();
        }
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

    pub fn wait_load_file(&self, filename: String) -> Result<Sound> {
        let sound_buffer = {
            let mut inner = self.inner.lock().unwrap();
            if let Some(buffer) = inner.loaded_files.get(&filename) {
                buffer.clone()
            }  else {
                let buffer = SoundBufferResource::new_generic(fyrox_sound::futures::executor::block_on(
                    DataSource::from_file(&filename, &FsResourceIo))
                    .map_err(|err| anyhow!("fyrox-sound error: DataSource::from_file(): {:?}", err))?
                ).map_err(|err| anyhow!("fyrox-sound error: SoundBufferResource::new_generic(): {:?}", err))?;
                inner.loaded_files.insert(filename, buffer.clone());
                buffer
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
        Ok(Sound { inner, is_looping: false })
    }
}
