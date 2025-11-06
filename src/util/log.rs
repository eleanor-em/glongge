use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

pub static LAST_LOG: LazyLock<Mutex<HashMap<String, Instant>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// TODO: info_every_frames
#[macro_export]
macro_rules! info_every_seconds {
    ($seconds:expr, $($args:expr),+) => {
        let loc = $crate::util::assert::current_location!().to_string();
        {
            let mut last_log = $crate::util::log::LAST_LOG.lock().unwrap();
            if last_log.get(&loc).map_or(true, |then| then.elapsed().as_secs() >= $seconds) {
                $crate::core::prelude::info!($($args),+);
                last_log.insert(loc.clone(), std::time::Instant::now());
            }
        }
    }
}
#[macro_export]
macro_rules! info_every_millis {
    ($millis:expr, $($args:expr),+) => {
        let loc = $crate::util::assert::current_location!().to_string();
        {
            let mut last_log = $crate::util::log::LAST_LOG.lock().unwrap();
            if last_log.get(&loc).map_or(true, |then| then.elapsed().as_millis() >= $millis) {
                $crate::core::prelude::info!($($args),+);
                last_log.insert(loc.clone(), std::time::Instant::now());
            }
        }
    }
}
#[macro_export]
macro_rules! warn_every_seconds {
    ($seconds:expr, $($args:expr),+) => {
        let loc = $crate::util::assert::current_location!().to_string();
        {
            let mut last_log = $crate::util::log::LAST_LOG.lock().unwrap();
            if last_log.get(&loc).map_or(true, |then| then.elapsed().as_secs() >= $seconds) {
                $crate::core::prelude::warn!($($args),+);
                last_log.insert(loc.clone(), std::time::Instant::now());
            }
        }
    }
}
#[macro_export]
macro_rules! warn_every_millis {
    ($millis:expr, $($args:expr),+) => {
        let loc = $crate::util::assert::current_location!().to_string();
        {
            let mut last_log = $crate::util::log::LAST_LOG.lock().unwrap();
            if last_log.get(&loc).map_or(true, |then| then.elapsed().as_millis() >= $millis) {
                $crate::core::prelude::warn!($($args),+);
                last_log.insert(loc.clone(), std::time::Instant::now());
            }
        }
    }
}

#[macro_export]
macro_rules! error_every_seconds {
    ($seconds:expr, $($args:expr),+) => {
        let loc = $crate::util::assert::current_location!().to_string();
        {
            let mut last_log = $crate::util::log::LAST_LOG.lock().unwrap();
            if last_log.get(&loc).map_or(true, |then| then.elapsed().as_secs() >= $seconds) {
                $crate::core::prelude::error!($($args),+);
                last_log.insert(loc.clone(), std::time::Instant::now());
            }
        }
    }
}
#[macro_export]
macro_rules! error_every_millis {
    ($millis:expr, $($args:expr),+) => {
        let loc = $crate::util::assert::current_location!().to_string();
        {
            let mut last_log = $crate::util::log::LAST_LOG.lock().unwrap();
            if last_log.get(&loc).map_or(true, |then| then.elapsed().as_millis() >= $millis) {
                $crate::core::prelude::error!($($args),+);
                last_log.insert(loc.clone(), std::time::Instant::now());
            }
        }
    }
}
