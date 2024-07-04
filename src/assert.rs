pub fn assert_ord<T: PartialOrd>(_: &T) {}
pub fn assert_partial_eq<T: PartialEq>(_: &T) {}
pub fn assert_same_type<T, U>(_: &T, _: &U) {}
pub fn assert_type<T>(_: &T) {}

macro_rules! current_location {
    () => {
        format!("{}:{}", file!(), line!())
    };
}
pub(crate) use current_location;

macro_rules! check {
    ($lhs:expr) => {{
        $crate::assert::assert_type::<bool>(&$lhs);
        if !$lhs {
            panic!(
                "check failed: {}: {}",
                $crate::assert::current_location!(),
                stringify!($lhs),
            );
        }
    }};
}
pub(crate) use check;

macro_rules! check_false {
    ($lhs:expr) => {
        $crate::assert::assert_type::<bool>(&$lhs);
        if $lhs {
            panic!(
                "check failed: {}: !{}",
                $crate::assert::current_location!(),
                stringify!($lhs),
            );
        }
    };
}
pub(crate) use check_false;

macro_rules! check_lt {
    ($lhs:expr, $rhs:expr) => {{
        $crate::assert::assert_same_type(&$lhs, &$rhs);
        $crate::assert::assert_ord(&$lhs);
        if $lhs >= $rhs {
            panic!(
                "check failed: {}: {} < {}: {:?} vs. {:?}",
                $crate::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
pub(crate) use check_lt;

macro_rules! check_gt {
    ($lhs:expr, $rhs:expr) => {{
        $crate::assert::assert_same_type(&$lhs, &$rhs);
        $crate::assert::assert_ord(&$lhs);
        if $lhs <= $rhs {
            panic!(
                "check failed: {}: {} > {}: {:?} vs. {:?}",
                $crate::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
pub(crate) use check_gt;

macro_rules! check_le {
    ($lhs:expr, $rhs:expr) => {{
        $crate::assert::assert_same_type(&$lhs, &$rhs);
        $crate::assert::assert_ord(&$lhs);
        if !($lhs <= $rhs) {
            panic!(
                "check failed: {}: {} <= {}: {:?} vs. {:?}",
                $crate::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
pub(crate) use check_le;

macro_rules! check_ge {
    ($lhs:expr, $rhs:expr) => {{
        $crate::assert::assert_same_type(&$lhs, &$rhs);
        $crate::assert::assert_ord(&$lhs);
        if !($lhs <= $rhs) {
            panic!(
                "check failed: {}: {} >= {}: {:?} vs. {:?}",
                $crate::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
pub(crate) use check_ge;

macro_rules! check_eq {
    ($lhs:expr, $rhs:expr) => {{
        $crate::assert::assert_same_type(&$lhs, &$rhs);
        $crate::assert::assert_partial_eq(&$lhs);
        if !($lhs == $rhs) {
            panic!(
                "check failed: {}: {} == {}: {:?} vs. {:?}",
                $crate::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
pub(crate) use check_eq;

macro_rules! check_ne {
    ($lhs:expr, $rhs:expr) => {{
        $crate::assert::assert_same_type(&$lhs, &$rhs);
        $crate::assert::assert_partial_eq(&$lhs);
        if !($lhs != $rhs) {
            panic!(
                "check failed: {}: {} != {}: {:?} vs. {:?}",
                $crate::assert::current_location!(),
                stringify!($lhs),
                stringify!($rhs),
                $lhs,
                $rhs
            );
        }
    }};
}
pub(crate) use check_ne;
