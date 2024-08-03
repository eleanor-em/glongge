#![feature(iterator_try_collect)]
#![feature(os_str_display)]

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, LineWriter, Write};
use std::path::Path;
use walkdir::WalkDir;

use anyhow::Result;

fn main() -> Result<()> {
    let sep = std::path::MAIN_SEPARATOR;
    let mut imports = Vec::new();
    let mut decls = Vec::new();
    let builtins = vec![
        "GgInternalSprite,".to_string(),
        "GgInternalCollisionShape,".to_string(),
        "GgInternalCanvas,".to_string(),
        "GgInternalCanvasItem,".to_string(),
        "GgInternalContainer,".to_string(),
    ];
    let current_dir = env::current_dir()?;
    for entry in WalkDir::new(current_dir.clone()) {
        let entry = entry?;
        let filename = entry.file_name().display().to_string();
        let is_rust_file = entry.path().extension()
            .map_or(false, |ext| ext.eq_ignore_ascii_case("rs"));
        if is_rust_file && filename != "build.rs" && filename != "object_type.rs" {
            let path = entry.path().display().to_string();
            let path = path.replace(&format!("{}{sep}", current_dir.display()), "");
            let parts = path.split(&format!("src{sep}")).collect::<Vec<_>>();
            if parts.len() == 2 {
                let rest = parts[1].to_string();
                let import = rest.replace(".rs", "").replace(sep, "::");
                if import.contains("mod") || import == "main" || import == "lib" {
                    continue;
                }

                let reader = BufReader::new(File::open(entry.path())?);
                let lines = reader.lines().try_collect::<Vec<_>>()?;
                let lines_with_decls = lines.iter()
                    .enumerate()
                    .filter(|(_, line)| line.starts_with("#[partially_derive_scene_object]"))
                    .map(|(i, _)| lines[i + 1].clone())
                    .collect::<Vec<_>>();
                if lines_with_decls.is_empty() {
                    if lines.iter().any(|line| line.starts_with("#[register_scene_object]")) {
                        println!("cargo::warning=file {path}: contains `#[register_scene_object]` but not `#[partially_derive_scene_object]`");
                    }
                    continue;
                }

                let mut warned = false;
                for line in lines_with_decls {
                    let parts = line.split("for").collect::<Vec<_>>();
                    assert!(parts.len() >= 2, "could not parse impl line: `{line}`");
                    let parts = parts[1].trim().split(' ').collect::<Vec<_>>();
                    let parts = parts[0].trim().split('<').collect::<Vec<_>>();
                    let struct_name = parts[0];
                    if struct_name.starts_with("GgInternal") {
                        imports.push(format!("use glongge::{import}::{struct_name};"));
                    } else {
                        if !warned {
                            if !lines.iter().any(|line| line.starts_with("use crate::object_type::ObjectType;")) {
                                println!("cargo::warning=file {path}: contains `#[partially_derive_scene_object]`, but does not import `crate::object_type::ObjectType`");
                            }
                            if !lines.iter().any(|line| line.starts_with("#[partially_derive_scene_object]")) {
                                assert!(lines.iter().any(|line| line.starts_with("impl SceneObject<ObjectType>")),
                                        "file {path}: contains `#[partially_derive_scene_object]`, but does not contain an implementation of SceneObject<ObjectType>");
                            }
                            warned = true;
                        }
                        imports.push(format!("use crate::{import}::{struct_name};"));
                    }
                    decls.push(format!("{struct_name},"));
                }
            } else {
                assert!(parts.len() == 1, "could not parse file path: {}", entry.path().display());
            }
        }
    }

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("object_type.rs");
    let mut writer = LineWriter::new(File::create(dest_path)?);
    writer.write_all("pub mod object_type {\n".as_bytes())?;
    writer.write_all("use glongge_derive::register_object_type;\n".as_bytes())?;
    for import in imports.into_iter().map(|line| format!("{line}\n")) {
        writer.write_all(import.as_bytes())?;
    }
    writer.write_all("\n".as_bytes())?;
    writer.write_all("#[register_object_type]\n".as_bytes())?;
    writer.write_all("pub enum ObjectType {\n".as_bytes())?;
    for builtin in builtins {
        if !decls.contains(&builtin) {
            decls.push(builtin);
        }
    }
    for decl in decls.into_iter().map(|line| format!("{line}\n")) {
        writer.write_all(decl.as_bytes())?;
    }
    writer.write_all("}\n".as_bytes())?;
    writer.write_all("}\n".as_bytes())?;
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src{sep}");
    Ok(())
}
