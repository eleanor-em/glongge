use quote::quote;
use syn::{
    parse_macro_input,
    ItemImpl,
    ImplItemFn,
    ImplItem,
    Type
};

#[proc_macro_attribute]
/// Creates implementations of the following SceneObject trait methods:
/// - [`as_any()`](glongge::core::scene::SceneObject::as_any):
///   Trivial implementation.
/// - [`as_any_mut()`](glongge::core::scene::SceneObject::as_any_mut):
///   Trivial implementation.
/// - [`gg_type_name()`](glongge::core::scene::SceneObject::gg_type_name):
///   The name of the struct.
/// Meant to be used for implementations of SceneObject:
/// ```ignore 
/// use glongge_derive::register_scene_object;
///
/// #[register_scene_object]
/// struct MyObject {};
/// #[partially_derive_scene_object]
/// impl SceneObject for MyObject {
///   // Generated code below:
///   fn as_any(&self) -> &dyn Any {
///       self
///   }
///   fn as_any_mut(&mut self) -> &mut dyn Any {
///       self
///   }
///   fn gg_type_name(&self) -> String {
///       "MyObject".to_string()
///   }
/// }
/// ```
/// See complete examples in src/examples.
pub fn partially_derive_scene_object(_attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut item_impl = parse_macro_input!(item as ItemImpl);
    let struct_name = if let Type::Path(type_path) = &*item_impl.self_ty {
        type_path.path.segments.last().unwrap().ident.clone()
    } else {
        panic!("Unsupported type for impl block");
    }.to_string();

    let has_as_any = item_impl.items.iter().any(|item| {
        if let ImplItem::Fn(ImplItemFn { sig, .. }) = item {
            if sig.ident == "as_any" {
                return true;
            }
        }
        return false;
    });
    if !has_as_any {
        item_impl.items.push(syn::parse_quote! {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        });
    }
    let has_as_any_mut = item_impl.items.iter().any(|item| {
        if let ImplItem::Fn(ImplItemFn { sig, .. }) = item {
            if sig.ident == "as_any_mut" {
                return true;
            }
        }
        return false;
    });
    if !has_as_any_mut {
        item_impl.items.push(syn::parse_quote! {
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        });
    }
    let has_type_name = item_impl.items.iter().any(|item| {
        if let ImplItem::Fn(ImplItemFn { sig, .. }) = item {
            if sig.ident == "gg_type_name" {
                return true;
            }
        }
        return false;
    });
    if !has_type_name {
        item_impl.items.push(syn::parse_quote! {
            fn gg_type_name(&self) -> String {
                #struct_name.to_string()
            }
        });
    }

    proc_macro::TokenStream::from(quote! { #item_impl })
}
