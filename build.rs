#![feature(iterator_try_collect)]
#![feature(os_str_display)]

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, LineWriter, Write};
use std::path::Path;
use walkdir::WalkDir;

use anyhow::Result;

fn main() -> Result<()> {
    let mut imports = Vec::new();
    let mut decls = vec!["#[register_object_type]".to_string(), "pub enum ObjectType {".to_string()];
    let current_dir = env::current_dir()?;
    for entry in WalkDir::new(current_dir.clone()) {
        let entry = entry?;
        let filename = entry.file_name().display().to_string();
        if filename != "build.rs" && filename != "object_type.rs" && filename.ends_with(".rs") {
            let path = entry.path().display().to_string();
            let path = path.replace(&format!("{}/", current_dir.display()), "");
            let parts = path.split("src/").collect::<Vec<_>>();
            if parts.len() == 2 {
                let rest = parts[1].to_string();
                let import = rest.replace(".rs", "").replace("/", "::");
                if import.contains("mod") || import == "main" || import == "lib" {
                    continue;
                }

                let reader = BufReader::new(File::open(entry.path())?);
                let lines = reader.lines().try_collect::<Vec<_>>()?;
                let lines_with_decls = lines.iter()
                    .enumerate()
                    .filter(|(_, line)| line.contains("#[partially_derive_scene_object]"))
                    .map(|(i, _)| lines[i + 1].clone())
                    .collect::<Vec<_>>();
                if !lines_with_decls.is_empty() {
                    if lines.iter().find(|line| line.starts_with("use crate::object_type::ObjectType;"))
                            .is_none() {
                        panic!("file {path}: contains `#[partially_derive_scene_object]`, but does not import `crate::object_type::ObjectType`");
                    }
                }
                for line in lines_with_decls {
                    let parts = line.split("for").collect::<Vec<_>>();
                    if parts.len() < 2 {
                        panic!("could not parse impl line: `{line}`");
                    }
                    let parts = parts[1].trim().split(" ").collect::<Vec<_>>();
                    if parts.is_empty() {
                        panic!("could not parse impl line: `{line}`");
                    }
                    let struct_name = parts[0];
                    imports.push(format!("use crate::{import}::{struct_name};"));
                    decls.push(format!("{struct_name},"));
                }
            } else if parts.len() > 2 {
                panic!("could not parse file path: {}", entry.path().display());
            }
        }
    }
    decls.push("}".to_string());

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("object_type.rs");
    let mut writer = LineWriter::new(File::create(dest_path)?);
    writer.write_all("pub mod object_type {\n".as_bytes())?;
    writer.write_all("use glongge_derive::register_object_type;\n".as_bytes())?;
    for import in imports.into_iter().map(|line| format!("{line}\n")) {
        writer.write_all(import.as_bytes())?;
    }
    writer.write_all("\n".as_bytes())?;
    for decl in decls.into_iter().map(|line| format!("{line}\n")) {
        writer.write_all(decl.as_bytes())?;
    }
    writer.write_all("}\n".as_bytes())?;
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/");
    Ok(())
}
