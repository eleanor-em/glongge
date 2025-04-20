use quote::{quote, quote_spanned};
use syn::{
    spanned::Spanned,
    parse_macro_input,
    Data,
    DeriveInput,
    ItemImpl,
    ImplItemFn,
    ImplItem,
    Fields,
    Type
};

#[proc_macro_attribute]
pub fn register_object_type(_args: proc_macro::TokenStream, input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();
    let as_default_code = as_default_impl(&input.data);
    let as_typeid_code = as_typeid_impl(&input.data);
    let all_values_code = all_values_impl(&input.data);

    let expanded = quote! {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        #input

        impl glongge::core::ObjectTypeEnum for #name {
            fn as_default(self) -> glongge::core::SceneObjectWrapper<Self> { #as_default_code }
            fn as_typeid(self) -> std::any::TypeId { #as_typeid_code }
            fn all_values() -> Vec<Self> { #all_values_code }
            fn gg_sprite() -> Self { Self::GgInternalSprite }
            fn gg_collider() -> Self { Self::GgInternalCollisionShape }
            fn gg_canvas() -> Self { Self::GgInternalCanvas }
            fn gg_container() -> Self { Self::GgInternalContainer }
            fn gg_static_sprite() -> Self { Self::GgInternalStaticSprite }
            fn gg_colliding_sprite() -> Self { Self::GgInternalCollidingSprite }
            fn gg_tileset() -> Self { Self::GgInternalTileset }
            fn gg_interactive_spline() -> Self { Self::GgInternalInteractiveSpline }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

fn get_initializer_for(field_name: proc_macro2::Ident, ty: Type) -> proc_macro2::TokenStream {
    if let Type::Path(type_path) = ty {
        match type_path.path.segments.last().unwrap().ident.to_string().as_str() {
            "Instant" => return quote! {
                #field_name: Instant::now()
            },
            _ => {},
        }
    }
    quote! {
        #field_name: Default::default()
    }
}

#[proc_macro_attribute]
pub fn register_scene_object(_args: proc_macro::TokenStream, input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let struct_name = input.ident.clone();
    let maybe_template = if input.generics.lt_token.is_some() {
        quote! {
            <ObjectType>
        }
    } else {
        quote! {}
    };

    let mut default_initializers = Vec::new();
    if let syn::Data::Struct(data_struct) = input.data.clone() {
        if let Fields::Named(fields_named) = data_struct.fields {
            for field in fields_named.named {
                let field_name = field.ident.unwrap();
                default_initializers.push(get_initializer_for(field_name, field.ty));
            }
        } else {
            panic!("no named fields");
        }
    } else {
        panic!("not a struct");
    }

    let expanded = quote! {
        #input

        impl #maybe_template Default for #struct_name #maybe_template {
            fn default() -> Self {
                Self {
                    #(#default_initializers),*
                }
            }
        }
    };
    proc_macro::TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn partially_derive_scene_object(_attr: proc_macro::TokenStream, item: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut item_impl = parse_macro_input!(item as ItemImpl);
    let struct_name = if let syn::Type::Path(type_path) = &*item_impl.self_ty {
        type_path.path.segments.last().unwrap().ident.clone()
    } else {
        panic!("Unsupported type for impl block");
    };

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
    let has_gg_type_enum = item_impl.items.iter().any(|item| {
        if let ImplItem::Fn(ImplItemFn { sig, .. }) = item {
            if sig.ident == "gg_type_enum" {
                return true;
            }
        }
        return false;
    });
    if !has_gg_type_enum {
        item_impl.items.push(syn::parse_quote! {
            fn gg_type_enum(&self) -> ObjectType {
                ObjectType::#struct_name
            }
        });
    }

    proc_macro::TokenStream::from(quote! { #item_impl })
}

fn has_object_type_param(name: &proc_macro2::Ident) -> bool {
    match name.to_string().as_str() {
        "GgInternalContainer" => true,
        _ => false,
    }
}

fn as_default_impl(data: &Data) -> proc_macro2::TokenStream {
    match *data {
        Data::Enum(ref data) => {
            let recurse = data.variants.iter().map(|variant| {
                let name = &variant.ident;
                if has_object_type_param(name) {
                    quote_spanned! {variant.span()=>
                        Self::#name => #name::<Self>::default().into_wrapper()
                    }
                } else {
                    quote_spanned! {variant.span()=>
                        Self::#name => #name::default().into_wrapper()
                    }
                }
            });
            quote! {
                match self {
                    #(#recurse,)*
                }
            }
        }
        _ => panic!("unimplemented (as_default_impl, {data:?})")
    }
}
fn as_typeid_impl(data: &Data) -> proc_macro2::TokenStream {
    match *data {
        Data::Enum(ref data) => {
            let recurse = data.variants.iter().map(|variant| {
                let name = &variant.ident;
                if has_object_type_param(name) {
                    quote_spanned! {variant.span()=>
                        Self::#name => std::any::TypeId::of::<#name::<Self>>()
                    }
                } else {
                    quote_spanned! {variant.span()=>
                        Self::#name => std::any::TypeId::of::<#name>()
                    }
                }
            });
            quote! {
                match self {
                    #(#recurse,)*
                }
            }
        }
        _ => panic!("unimplemented (as_typeid_impl, {data:?})")
    }
}

fn all_values_impl(data: &Data) -> proc_macro2::TokenStream {
    match *data {
        Data::Enum(ref data) => {
            let recurse = data.variants.iter().map(|variant| {
                let name = &variant.ident;
                quote_spanned! {variant.span()=>
                    #name
                }
            });
            quote! {
                vec![#(Self::#recurse,)*]
            }
        }
        _ => panic!("unimplemented (all_values_impl, {data:?})")
    }
}
