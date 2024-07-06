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

    let match_exp = as_type_roundtrip_exp(&input.data);
    let vec = members_vec(&input.data);

    let expanded = quote! {
        impl gg::ObjectTypeEnum for #name {
            fn as_type_roundtrip(self) -> Self { #match_exp }
            fn all_values() -> Vec<Self> { #vec }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

fn as_type_roundtrip_exp(data: &Data) -> proc_macro2::TokenStream {
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

fn members_vec(data: &Data) -> proc_macro2::TokenStream {
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
