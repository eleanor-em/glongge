use crate::core::prelude::*;
use rand::{Rng, rng};
use rodio::{Decoder, OutputStreamBuilder, Sink, Source, buffer::SamplesBuffer, mixer::Mixer};
use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{Arc, Mutex, OnceLock},
    thread::JoinHandle,
};

// OutputStream is not Send on macOS (CoreAudio limitation), so we leak it.
// This OnceLock ensures we only create one OutputStream.
static DID_CREATE_OUTPUT_STREAM: OnceLock<bool> = OnceLock::new();

struct SoundInner {
    buffer: SamplesBuffer,
    sink: Sink,
}
#[derive(Clone, Default)]
pub struct Sound {
    inner: Option<Arc<SoundInner>>,
    is_looping: bool,
}

impl Sound {
    pub fn play_shifted(&mut self, mag: f32) {
        if DISABLE_SOUND {
            return;
        }
        if let Some(inner) = self.inner.as_ref() {
            let mut rng = rng();
            let pitch = linalg::eerp(1.0 - mag, 1.0 + mag, rng.random_range(0.0..1.0));

            inner.sink.stop();
            inner.sink.append(inner.buffer.clone().speed(pitch));
        } else {
            error!("tried to play non-loaded sound");
        }
    }
    pub fn play(&mut self) {
        if DISABLE_SOUND {
            return;
        }
        if let Some(inner) = self.inner.as_ref() {
            inner.sink.stop();
            inner.sink.append(inner.buffer.clone());
        } else {
            error!("tried to play non-loaded sound");
        }
    }

    pub fn play_loop(&mut self) {
        if DISABLE_SOUND {
            return;
        }
        if let Some(inner) = self.inner.as_ref() {
            inner.sink.stop();
            inner.sink.append(inner.buffer.clone().repeat_infinite());
            self.is_looping = true;
        } else {
            error!("tried to play non-loaded sound");
        }
    }

    pub fn stop(&mut self) {
        if DISABLE_SOUND {
            return;
        }
        if let Some(inner) = self.inner.as_ref() {
            inner.sink.stop();
            self.is_looping = false;
        } else {
            error!("tried to stop non-loaded sound");
        }
    }

    pub fn is_playing(&self) -> bool {
        if DISABLE_SOUND {
            return false;
        }
        self.inner.as_ref().is_some_and(|inner| !inner.sink.empty())
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
    loaded_files: BTreeMap<String, SamplesBuffer>,
}

pub struct SoundHandler {
    mixer: Arc<Mixer>,
    inner: Arc<Mutex<SoundHandlerInner>>,
    join_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl SoundHandler {
    pub fn new() -> Result<Arc<Self>> {
        if DID_CREATE_OUTPUT_STREAM.get().is_some() {
            bail!("SoundHandler::new() called more than once");
        }
        DID_CREATE_OUTPUT_STREAM.get_or_init(|| true);

        let stream = OutputStreamBuilder::open_default_stream().map_err(|e| {
            anyhow!("rodio error: OutputStreamBuilder::open_default_stream(): {e:?}")
        })?;
        let mixer = stream.mixer().clone();
        // Leak the stream to keep it alive.
        Box::leak(Box::new(stream));

        Ok(Arc::new(Self {
            mixer: Arc::new(mixer),
            inner: Arc::new(Mutex::new(SoundHandlerInner {
                loaded_files: BTreeMap::new(),
            })),
            join_handles: Arc::new(Mutex::new(Vec::new())),
        }))
    }

    fn load_file_inner(
        inner: &Arc<Mutex<SoundHandlerInner>>,
        mixer: &Arc<Mixer>,
        filename: impl AsRef<str>,
    ) -> Result<Sound> {
        let filename = filename.as_ref().to_string();
        let buffer = {
            if let Some(buffer) = inner.lock().unwrap().loaded_files.get(&filename).cloned() {
                buffer
            } else {
                // Read and decode the file
                let bytes = std::fs::read(&filename)
                    .map_err(|e| anyhow!("failed to read sound file {filename}: {e}"))?;
                let decoder = Decoder::new(Cursor::new(bytes))
                    .map_err(|e| anyhow!("failed to decode sound file {filename}: {e:?}"))?;

                let channels = decoder.channels();
                let sample_rate = decoder.sample_rate();
                let samples: Vec<f32> = decoder.collect();

                let buffer = SamplesBuffer::new(channels, sample_rate, samples.clone());
                inner
                    .lock()
                    .unwrap()
                    .loaded_files
                    .insert(filename, buffer.clone());
                buffer
            }
        };

        let sink = Sink::connect_new(mixer);

        let inner = Some(Arc::new(SoundInner { buffer, sink }));
        Ok(Sound {
            inner,
            is_looping: false,
        })
    }
}

impl Loader<Sound> for SoundHandler {
    fn spawn_load_file(&self, filename: impl AsRef<str>) {
        let inner = self.inner.clone();
        let mixer = self.mixer.clone();
        let filename = filename.as_ref().to_string();
        let handle = std::thread::spawn(move || {
            Self::load_file_inner(&inner, &mixer, filename).unwrap();
        });
        self.join_handles
            .try_lock()
            .expect("join_handles should only be locked after calling wait()")
            .push(handle);
    }

    fn wait_load_file(&self, filename: impl AsRef<str>) -> Result<Sound> {
        Self::load_file_inner(&self.inner, &self.mixer, filename)
    }

    fn wait(&self) -> Result<()> {
        let handles = self
            .join_handles
            .try_lock()
            .expect("join_handles should not be locked when calling wait()")
            .drain(..)
            .collect_vec();
        for handle in handles {
            handle
                .join()
                // XXX: not sure why this is needed.
                .map_err(|e| anyhow!("join error: {e:?}"))?;
        }
        Ok(())
    }
}
