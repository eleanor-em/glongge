pub fn assert_partial_ord<T: PartialOrd>(_: &T) {}
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
        let value = $lhs;
        $crate::core::util::assert::assert_type::<bool>(&value);
        if !value {
            panic!(
                "check failed: {}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
            );
        }
    }};
    ($lhs:expr, $extra:expr) => {{
        let value = $lhs;
        $crate::core::util::assert::assert_type::<bool>(&value);
        if !value {
            panic!(
                "check failed: {}: {}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                $extra
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
        let value = $lhs;
        $crate::core::util::assert::assert_type::<bool>(&value);
        if value {
            panic!(
                "check failed: {}: !{}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
            );
        }
    };
    ($lhs:expr, $extra:expr) => {
        let value = $lhs;
        $crate::core::util::assert::assert_type::<bool>(&value);
        if value {
            panic!(
                "check failed: {}: !{}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                $extra
            );
        }
    };
}
#[allow(unused_imports)]
pub use check_false;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_is_some {
    ($lhs:expr) => {{
        let value = $lhs;
        $crate::core::util::assert::assert_type::<Option<_>>(&value);
        if value.is_none() {
            panic!(
                "check failed: {}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
            );
        }
    }};
    ($lhs:expr, $extra:expr) => {{
        let value = $lhs;
        $crate::core::util::assert::assert_type::<Option<_>>(&value);
        if value.is_none() {
            panic!(
                "check failed: {}: {}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                $extra
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check_is_some;

#[allow(unused_macros)]
#[macro_export]
macro_rules! check_lt {
    ($lhs:expr, $rhs:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_ord(&lhs);
        if lhs >= rhs {
            panic!(
                "check failed: {}: {} < {}: {:?} vs. {:?}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs
            );
        }
    }};
    ($lhs:expr, $rhs:expr, $extra:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_ord(&lhs);
        if lhs >= rhs {
            panic!(
                "check failed: {}: {} < {}: {:?} vs. {:?}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs,
                $extra
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
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_ord(&lhs);
        if lhs <= rhs {
            panic!(
                "check failed: {}: {} > {}: {:?} vs. {:?}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs
            );
        }
    }};
    ($lhs:expr, $rhs:expr, $extra:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_ord(&lhs);
        if lhs <= rhs {
            panic!(
                "check failed: {}: {} > {}: {:?} vs. {:?}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs,
                $extra
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
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_ord(&lhs);
        if lhs > rhs {
            panic!(
                "check failed: {}: {} <= {}: {:?} vs. {:?}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs
            );
        }
    }};
    ($lhs:expr, $rhs:expr, $extra:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_ord(&lhs);
        if lhs > rhs {
            panic!(
                "check failed: {}: {} <= {}: {:?} vs. {:?}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs,
                $extra
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
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_ord(&lhs);
        if lhs < rhs {
            panic!(
                "check failed: {}: {} >= {}: {:?} vs. {:?}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs
            );
        }
    }};
    ($lhs:expr, $rhs:expr, $extra:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_ord(&lhs);
        if lhs < rhs {
            panic!(
                "check failed: {}: {} >= {}: {:?} vs. {:?}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs,
                $extra
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
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_eq(&lhs);
        #[allow(clippy::float_cmp)]
        if lhs != rhs {
            panic!(
                "check failed: {}: {} == {}: {:?} vs. {:?}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs
            );
        }
    }};
    ($lhs:expr, $rhs:expr, $extra:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_eq(&lhs);
        if lhs != rhs {
            panic!(
                "check failed: {}: {} == {}: {:?} vs. {:?}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs,
                $extra
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
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        if !lhs.almost_eq(rhs) {
            panic!(
                "check failed: {}: {} ~= {}: {:?} vs. {:?}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs
            );
        }
    }};
    ($lhs:expr, $rhs:expr, $extra:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        if !lhs.almost_eq(rhs) {
            panic!(
                "check failed: {}: {} ~= {}: {:?} vs. {:?}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs,
                $extra
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
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_eq(&lhs);
        if lhs == rhs {
            panic!(
                "check failed: {}: {} != {}: {:?} vs. {:?}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs
            );
        }
    }};
    ($lhs:expr, $rhs:expr, $extra:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        $crate::core::util::assert::assert_same_type(&lhs, &rhs);
        $crate::core::util::assert::assert_partial_eq(&lhs);
        if lhs == rhs {
            panic!(
                "check failed: {}: {} != {}: {:?} vs. {:?}: {}",
                $crate::core::util::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                lhs,
                rhs,
                $extra
            );
        }
    }};
}
#[allow(unused_imports)]
pub use check_ne;
