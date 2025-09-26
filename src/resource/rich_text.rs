use crate::core::prelude::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub enum FormatInstruction {
    SetColourTo(Colour),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ParseCharInstruction {
    Skip,
    AlreadyHandled,
    Output,
}

#[derive(Clone, Debug)]
pub struct FormattedChars {
    pub(crate) format_instructions: BTreeMap<usize, FormatInstruction>,
    pub(crate) chars: Vec<char>,
}

impl FormattedChars {
    pub fn unformatted(text: impl AsRef<str>) -> Self {
        Self {
            format_instructions: BTreeMap::new(),
            chars: text.as_ref().chars().collect_vec(),
        }
    }
    fn unformatted_chars(chars: Vec<char>) -> Self {
        Self {
            format_instructions: BTreeMap::new(),
            chars,
        }
    }

    pub fn parse(text: impl AsRef<str>) -> Option<Self> {
        let mut rv = FormattedChars {
            format_instructions: BTreeMap::new(),
            chars: Vec::new(),
        };
        let chars = text.as_ref().chars().collect_vec();

        let mut is_escaped = false;
        let mut i = 0;
        let mut out_ix = 0;
        while i < chars.len() {
            check_le!(out_ix, i);
            let c = chars[i];
            let action = match c {
                '\\' => {
                    is_escaped = true;
                    i += 1;
                    ParseCharInstruction::Skip
                }
                '{' => {
                    if is_escaped {
                        is_escaped = false;
                        ParseCharInstruction::Skip
                    } else {
                        let (num_parsed, formatted) =
                            parse_format_instruction(i, out_ix, &chars[..i], &chars[i..])?;
                        for (instr_ix, instr) in formatted.format_instructions {
                            rv.format_instructions.insert(instr_ix, instr);
                        }
                        for c in formatted.chars {
                            rv.chars.push(c);
                            out_ix += 1;
                        }
                        i += num_parsed;
                        ParseCharInstruction::AlreadyHandled
                    }
                }
                _ => {
                    if is_escaped {
                        error!(
                            "rich_text::parse(): unexpectedly is_escaped: at {i}, prefix {:?}, postfix {:?}",
                            chars[..i].iter().collect::<String>(),
                            chars[(i + 1)..].iter().collect::<String>()
                        );
                    }
                    ParseCharInstruction::Output
                }
            };
            match action {
                ParseCharInstruction::Skip | ParseCharInstruction::AlreadyHandled => {}
                ParseCharInstruction::Output => {
                    rv.chars.push(c);
                    out_ix += 1;
                    i += 1;
                }
            }
        }
        Some(rv)
    }
}

fn parse_format_instruction(
    start_ix: usize,
    start_out_ix: usize,
    prefix: &[char],
    chars: &[char],
) -> Option<(usize, FormattedChars)> {
    if let Some(closing_brace_ix) = chars
        .iter()
        .enumerate()
        .tuple_windows()
        .find(|((_, prev), (_, cur))| **prev != '\\' && **cur == '}')
        .map(|(_, (j, _))| j)
    {
        Some((
            closing_brace_ix + 1,
            parse_format_instruction_inner(
                start_ix,
                start_out_ix,
                prefix,
                &chars[1..closing_brace_ix],
            )?,
        ))
    } else {
        error!(
            "unmatched `{{` at {start_ix}: prefix {:?}, chars {:?}",
            prefix.iter().collect::<String>(),
            chars.iter().collect::<String>()
        );
        None
    }
}

fn parse_format_instruction_inner(
    start_ix: usize,
    start_out_ix: usize,
    prefix: &[char],
    chars: &[char],
) -> Option<FormattedChars> {
    let Some((mid, _)) = chars.iter().find_position(|c| **c == ':') else {
        error!(
            "missing `:` in format instruction at {start_ix}: prefix {:?}, chars {:?}",
            prefix.iter().collect::<String>(),
            chars.iter().collect::<String>()
        );
        return None;
    };
    let (instr, text) = chars.split_at(mid);
    let instr_str = instr.iter().collect::<String>();
    let out_chars = text[1..].iter().copied().collect_vec();

    let col_pattern = "col=";
    if instr_str.starts_with(col_pattern) {
        if let Some(colour) = parse_colour(start_ix, prefix, &instr[col_pattern.len()..]) {
            Some(FormattedChars {
                format_instructions: [
                    (start_out_ix, FormatInstruction::SetColourTo(colour)),
                    (
                        start_out_ix + out_chars.len(),
                        FormatInstruction::SetColourTo(Colour::black()),
                    ),
                ]
                .into_iter()
                .collect(),
                chars: out_chars,
            })
        } else {
            Some(FormattedChars::unformatted_chars(out_chars))
        }
    } else {
        error!(
            "unknown format instruction `{instr_str}` at {start_ix}: prefix {:?}, chars {:?}",
            prefix.iter().collect::<String>(),
            chars.iter().collect::<String>()
        );
        Some(FormattedChars::unformatted_chars(out_chars))
    }
}

fn parse_colour(start: usize, prefix: &[char], chars: &[char]) -> Option<Colour> {
    match chars.iter().collect::<String>().as_str() {
        "red" => Some(Colour::red()),
        "white" => Some(Colour::white()),
        unknown => {
            if unknown.starts_with('#')
                && unknown.len() == 7
                && unknown[1..].chars().all(|c| c.is_ascii_hexdigit())
            {
                Some(Colour::from_hex_rgb(&unknown[1..]).unwrap())
            } else {
                error!(
                    "unknown colour `{unknown}` at {start}: prefix {:?}, chars {:?}",
                    prefix.iter().collect::<String>(),
                    chars.iter().collect::<String>()
                );
                None
            }
        }
    }
}
