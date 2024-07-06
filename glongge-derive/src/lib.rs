use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{
    parse_macro_input, Data, DeriveInput
};

#[proc_macro_attribute]
pub fn register_object_type(_args: proc_macro::TokenStream, input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let expanded = quote! {
        use glongge_derive::ObjectTypeEnum;

        #[derive(Clone, Copy, Debug, Eq, PartialEq, ObjectTypeEnum)]
        #input
    };
    proc_macro::TokenStream::from(expanded)
}

#[proc_macro_derive(ObjectTypeEnum)]
pub fn derive_object_type_enum(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let match_exp = as_type_roundtrip_impl(&input.data);
    let vec = all_values_impl(&input.data);

    let expanded = quote! {
        impl gg::ObjectTypeEnum for #name {
            fn as_type_roundtrip(self) -> Self { #match_exp }
            fn all_values() -> Vec<Self> { #vec }
            fn checked_downcast<T: Default + SceneObject<Self>>(obj: &dyn SceneObject<Self>) -> &T {
                let expected = T::default().get_type();
                if obj.get_type() != expected {
                    panic!("attempt to downcast {:?} -> {:?}", obj.get_type(), expected)
                }
                unsafe { & *(obj as *const dyn SceneObject<Self> as *const T) }
            }
            fn checked_downcast_mut<T: Default + SceneObject<Self>>(obj: &mut dyn SceneObject<Self>) -> &mut T {
                let expected = T::default().get_type();
                if obj.get_type() != expected {
                    panic!("attempt to downcast {:?} -> {:?}", obj.get_type(), expected)
                }
                unsafe { &mut *(obj as *mut dyn SceneObject<Self> as *mut T) }
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

fn as_type_roundtrip_impl(data: &Data) -> proc_macro2::TokenStream {
    match *data {
        Data::Enum(ref data) => {
            let recurse = data.variants.iter().map(|variant| {
                let name = &variant.ident;
                quote_spanned! {variant.span()=>
                    Self::#name => #name::default().get_type()
                }
            });
            quote! {
                match self {
                    #(#recurse,)*
                }
            }
        }
        _ => unimplemented!(),
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
        _ => unimplemented!(),
    }
}
