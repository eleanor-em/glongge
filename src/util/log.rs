use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

pub static LAST_LOG: LazyLock<Mutex<HashMap<String, Instant>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

#[macro_export]
macro_rules! info_every_seconds {
    ($seconds:expr, $($args:expr),+) => {
        let loc = $crate::util::assert::current_location!().to_string();
        {
            let mut last_log = $crate::util::log::LAST_LOG.lock().unwrap();
            if last_log.get(&loc).map_or(true, |then| then.elapsed().as_secs() >= $seconds) {
                info!($($args),+);
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
                info!($($args),+);
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
                warn!($($args),+);
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
                warn!($($args),+);
                last_log.insert(loc.clone(), std::time::Instant::now());
            }
        }
    }
}
