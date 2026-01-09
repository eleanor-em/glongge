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
    pub fn chars(&self) -> &[char] {
        &self.chars
    }
    pub fn chars_vec(&self) -> Vec<char> {
        self.chars.clone()
    }

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
                        ParseCharInstruction::Output
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
                        // Extracted to local variables due to llvm-cov not tracking
                        // array slice operations inside macro arguments.
                        let prefix_str: String = chars[..i].iter().collect();
                        let postfix_str: String = chars[(i + 1)..].iter().collect();
                        error!(
                            "rich_text::parse(): unexpectedly is_escaped: at {i}, prefix {:?}, postfix {:?}",
                            prefix_str, postfix_str
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
    // Process escape sequences: \} becomes }
    let text_slice = &text[1..];
    let mut out_chars = Vec::with_capacity(text_slice.len());
    let mut j = 0;
    while j < text_slice.len() {
        if text_slice[j] == '\\' && j + 1 < text_slice.len() && text_slice[j + 1] == '}' {
            j += 1; // skip the backslash, will push the } on next iteration
        }
        out_chars.push(text_slice[j]);
        j += 1;
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unformatted_basic() {
        let text = FormattedChars::unformatted("hello world");
        assert_eq!(
            text.chars(),
            &['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
        );
        assert!(text.format_instructions.is_empty());
    }

    #[test]
    fn test_unformatted_empty() {
        let text = FormattedChars::unformatted("");
        assert!(text.chars().is_empty());
        assert!(text.format_instructions.is_empty());
    }

    #[test]
    fn test_chars_vec() {
        let text = FormattedChars::unformatted("abc");
        assert_eq!(text.chars_vec(), vec!['a', 'b', 'c']);
    }

    #[test]
    fn test_parse_plain_text() {
        let result = FormattedChars::parse("hello").unwrap();
        assert_eq!(result.chars(), &['h', 'e', 'l', 'l', 'o']);
        assert!(result.format_instructions.is_empty());
    }

    #[test]
    fn test_parse_empty() {
        let result = FormattedChars::parse("").unwrap();
        assert!(result.chars().is_empty());
        assert!(result.format_instructions.is_empty());
    }

    #[test]
    fn test_parse_escaped_brace() {
        let result = FormattedChars::parse(r"hello \{world\}").unwrap();
        assert_eq!(
            result.chars(),
            &[
                'h', 'e', 'l', 'l', 'o', ' ', '{', 'w', 'o', 'r', 'l', 'd', '}'
            ]
        );
        assert!(result.format_instructions.is_empty());
    }

    #[test]
    fn test_parse_colour_red() {
        let result = FormattedChars::parse("hello {col=red:world}!").unwrap();
        assert_eq!(
            result.chars(),
            &['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!']
        );
        assert_eq!(result.format_instructions.len(), 2);
        // Check that colour is set at index 6 (start of "world")
        assert!(matches!(
            result.format_instructions.get(&6),
            Some(FormatInstruction::SetColourTo(c)) if *c == Colour::red()
        ));
        // Check that colour is reset at index 11 (after "world")
        assert!(matches!(
            result.format_instructions.get(&11),
            Some(FormatInstruction::SetColourTo(c)) if *c == Colour::black()
        ));
    }

    #[test]
    fn test_parse_colour_white() {
        let result = FormattedChars::parse("{col=white:test}").unwrap();
        assert_eq!(result.chars(), &['t', 'e', 's', 't']);
        assert!(matches!(
            result.format_instructions.get(&0),
            Some(FormatInstruction::SetColourTo(c)) if *c == Colour::white()
        ));
    }

    #[test]
    fn test_parse_colour_hex() {
        let result = FormattedChars::parse("{col=#FF0000:red text}").unwrap();
        assert_eq!(result.chars_vec().iter().collect::<String>(), "red text");
        assert!(matches!(
            result.format_instructions.get(&0),
            Some(FormatInstruction::SetColourTo(c)) if *c == Colour::from_hex_rgb("FF0000").unwrap()
        ));
    }

    #[test]
    fn test_parse_multiple_colours() {
        let result = FormattedChars::parse("{col=red:a}{col=white:b}").unwrap();
        assert_eq!(result.chars(), &['a', 'b']);
        // First colour at 0, reset at 1; second colour at 1, reset at 2
        // Adjacent colours share index 1, so white overwrites the black reset (3 not 4)
        assert_eq!(result.format_instructions.len(), 3);
        assert!(matches!(
            result.format_instructions.get(&0),
            Some(FormatInstruction::SetColourTo(c)) if *c == Colour::red()
        ));
        assert!(matches!(
            result.format_instructions.get(&1),
            Some(FormatInstruction::SetColourTo(c)) if *c == Colour::white()
        ));
        assert!(matches!(
            result.format_instructions.get(&2),
            Some(FormatInstruction::SetColourTo(c)) if *c == Colour::black()
        ));
    }

    #[test]
    fn test_parse_unmatched_brace_returns_none() {
        let _ = crate::util::setup_log();
        let result = FormattedChars::parse("hello {world");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_missing_colon_returns_none() {
        let _ = crate::util::setup_log();
        let result = FormattedChars::parse("hello {col=red}");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_unknown_instruction_preserves_text() {
        let _ = crate::util::setup_log();
        let result = FormattedChars::parse("{unknown=value:text}").unwrap();
        assert_eq!(result.chars(), &['t', 'e', 'x', 't']);
        // Unknown instruction falls back to unformatted
        assert!(result.format_instructions.is_empty());
    }

    #[test]
    fn test_parse_unknown_colour_preserves_text() {
        let _ = crate::util::setup_log();
        let result = FormattedChars::parse("{col=purple:text}").unwrap();
        assert_eq!(result.chars(), &['t', 'e', 'x', 't']);
        // Unknown colour falls back to unformatted
        assert!(result.format_instructions.is_empty());
    }

    #[test]
    fn test_parse_mixed_content() {
        let result = FormattedChars::parse("plain {col=red:colored} plain").unwrap();
        assert_eq!(
            result.chars_vec().iter().collect::<String>(),
            "plain colored plain"
        );
    }

    #[test]
    fn test_parse_escaped_brace_inside_format() {
        // Escaped closing brace inside format instruction
        let result = FormattedChars::parse(r"{col=red:a\}b}").unwrap();
        assert_eq!(result.chars(), &['a', '}', 'b']);
    }

    #[test]
    fn test_parse_hex_colour_lowercase() {
        let result = FormattedChars::parse("{col=#00ff00:green}").unwrap();
        assert_eq!(result.chars(), &['g', 'r', 'e', 'e', 'n']);
        assert!(matches!(
            result.format_instructions.get(&0),
            Some(FormatInstruction::SetColourTo(c)) if *c == Colour::from_hex_rgb("00ff00").unwrap()
        ));
    }

    #[test]
    fn test_parse_invalid_hex_colour() {
        let _ = crate::util::setup_log();
        // Too short hex
        let result = FormattedChars::parse("{col=#FFF:text}").unwrap();
        assert_eq!(result.chars(), &['t', 'e', 'x', 't']);
        assert!(result.format_instructions.is_empty());

        // Invalid hex chars
        let result = FormattedChars::parse("{col=#GGGGGG:text}").unwrap();
        assert_eq!(result.chars(), &['t', 'e', 'x', 't']);
        assert!(result.format_instructions.is_empty());
    }
}
