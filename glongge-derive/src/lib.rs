use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{
    parse_macro_input, Data, DeriveInput
};

#[proc_macro_attribute]
pub fn register_object_type(_args: proc_macro::TokenStream, input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let expanded = quote! {
        #[derive(Clone, Copy, Debug, Eq, PartialEq, glongge_derive::ObjectTypeEnum)]
        #input
    };
    proc_macro::TokenStream::from(expanded)
}

#[proc_macro_derive(ObjectTypeEnum)]
pub fn derive_object_type_enum(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let match_exp = as_typeid_impl(&input.data);
    let vec = all_values_impl(&input.data);

    let expanded = quote! {
        impl gg::ObjectTypeEnum for #name {
            fn as_typeid(self) -> std::any::TypeId { #match_exp }
            fn all_values() -> Vec<Self> { #vec }
            fn checked_downcast<T: gg::SceneObject<Self> + 'static>(obj: &dyn gg::SceneObject<Self>) -> &T {
                let actual = obj.get_type().as_typeid();
                let expected = obj.as_any().type_id();
                if actual != expected {
                    for value in Self::all_values() {
                        if value.as_typeid() == actual {
                            panic!("attempt to downcast {:?} -> {:?}", obj.get_type(), value)
                        }
                    }
                    panic!("attempt to downcast {:?}: type missing? {:?}", obj.get_type(), Self::all_values());
                }
                obj.as_any().downcast_ref::<T>().unwrap()
            }
            fn checked_downcast_mut<T: gg::SceneObject<Self> + 'static>(obj: &mut dyn gg::SceneObject<Self>) -> &mut T {
                let actual = obj.get_type().as_typeid();
                let expected = obj.as_any().type_id();
                if actual != expected {
                    for value in Self::all_values() {
                        if value.as_typeid() == actual {
                            panic!("attempt to downcast {:?} -> {:?}", obj.get_type(), value)
                        }
                    }
                    panic!("attempt to downcast {:?}: type missing? {:?}", obj.get_type(), Self::all_values());
                }
                obj.as_any_mut().downcast_mut::<T>().unwrap()
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

fn as_typeid_impl(data: &Data) -> proc_macro2::TokenStream {
    match *data {
        Data::Enum(ref data) => {
            let recurse = data.variants.iter().map(|variant| {
                let name = &variant.ident;
                quote_spanned! {variant.span()=>
                    Self::#name => std::any::TypeId::of::<#name>()
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
