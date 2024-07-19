#[allow(unused_imports)]
use crate::core::prelude::*;

pub fn assert_ord<T: PartialOrd>(_: &T) {}
pub fn assert_partial_eq<T: PartialEq>(_: &T) {}
pub fn assert_same_type<T, U>(_: &T, _: &U) {}
pub fn assert_type<T>(_: &T) {}

#[allow(unused_macros)]
#[macro_export]
macro_rules! current_location {
    () => {
        format!("{}:{}", file!(), line!())
    };
}
#[allow(unused_imports)]
pub use current_location;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check {
    ($lhs:expr) => {{
        $crate::core::assert::assert_type::<bool>(&$lhs);
        if !$lhs {
            panic!(
                "check failed: {}: {}",
                $crate::core::assert::current_location!(),
                stringify!($lhs),
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_false {
    ($lhs:expr) => {
        $crate::core::assert::assert_type::<bool>(&$lhs);
        if $lhs {
            panic!(
                "check failed: {}: !{}",
                $crate::core::assert::current_location!(),
                stringify!($lhs),
            );
        }
    };
}
#[allow(unused_imports)]
pub use check_false;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_lt {
    ($lhs:expr, $rhs:expr) => {{
        $crate::core::assert::assert_same_type(&$lhs, &$rhs);
        $crate::core::assert::assert_ord(&$lhs);
        if $lhs >= $rhs {
            panic!(
                "check failed: {}: {} < {}: {:?} vs. {:?}",
                $crate::core::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check_lt;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_gt {
    ($lhs:expr, $rhs:expr) => {{
        $crate::core::assert::assert_same_type(&$lhs, &$rhs);
        $crate::core::assert::assert_ord(&$lhs);
        if $lhs <= $rhs {
            panic!(
                "check failed: {}: {} > {}: {:?} vs. {:?}",
                $crate::core::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check_gt;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_le {
    ($lhs:expr, $rhs:expr) => {{
        $crate::core::assert::assert_same_type(&$lhs, &$rhs);
        $crate::core::assert::assert_ord(&$lhs);
        if !($lhs <= $rhs) {
            panic!(
                "check failed: {}: {} <= {}: {:?} vs. {:?}",
                $crate::core::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check_le;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_ge {
    ($lhs:expr, $rhs:expr) => {{
        $crate::core::assert::assert_same_type(&$lhs, &$rhs);
        $crate::core::assert::assert_ord(&$lhs);
        if !($lhs <= $rhs) {
            panic!(
                "check failed: {}: {} >= {}: {:?} vs. {:?}",
                $crate::core::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check_ge;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_eq {
    ($lhs:expr, $rhs:expr) => {{
        $crate::core::assert::assert_same_type(&$lhs, &$rhs);
        $crate::core::assert::assert_partial_eq(&$lhs);
        if !($lhs == $rhs) {
            panic!(
                "check failed: {}: {} == {}: {:?} vs. {:?}",
                $crate::core::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check_eq;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_almost_eq {
    ($lhs:expr, $rhs:expr) => {{
        $crate::core::assert::assert_same_type(&$lhs, &$rhs);
        if !($lhs.almost_eq($rhs)) {
            panic!(
                "check failed: {}: {} ~= {}: {:?} vs. {:?}",
                $crate::core::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check_almost_eq;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_ne {
    ($lhs:expr, $rhs:expr) => {{
        $crate::core::assert::assert_same_type(&$lhs, &$rhs);
        $crate::core::assert::assert_partial_eq(&$lhs);
        if !($lhs != $rhs) {
            panic!(
                "check failed: {}: {} != {}: {:?} vs. {:?}",
                $crate::core::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check_ne;
