use imgui::Condition;
use itertools::Itertools;
use num_traits::Zero;
use crate::core::prelude::Vec2;

#[derive(Clone, Default)]
pub struct ImGuiCommandChain {
    inner: Vec<ImGuiCommand>,
}

impl ImGuiCommandChain {
    #[must_use]
    pub fn new() -> Self { Self { inner: Vec::new() } }
    #[must_use]
    pub fn window<F>(
        mut self,
        name: impl AsRef<str>,
        f: F,
        then: ImGuiCommandChain
    ) -> Self
    where
        F: FnOnce(ImGuiWindowCommand) -> ImGuiWindowCommand
    {
        let cmd = f(ImGuiWindowCommand::new(name.as_ref().to_string(), then)).into_command();
        self.inner.push(cmd);
        self
    }
    #[must_use]
    pub fn separator(mut self) -> Self {
        let cmd = ImGuiCommand::Separator;
        self.inner.push(cmd);
        self
    }
    #[must_use]
    pub fn text(mut self, text: impl AsRef<str>) -> Self {
        let cmd = ImGuiCommand::Text(ImGuiTextCommand::new(text.as_ref().to_string()));
        self.inner.push(cmd);
        self
    }
    #[must_use]
    pub fn text_wrapped(mut self, text: impl AsRef<str>) -> Self {
        let cmd = ImGuiCommand::TextWrapped(ImGuiTextCommand::new(text.as_ref().to_string()));
        self.inner.push(cmd);
        self
    }

    #[must_use]
    pub fn get_mouse_pos(mut self, f: fn(Vec2) -> ImGuiCommandChain) -> Self {
        let cmd = ImGuiCommand::GetMousePos(ImGuiGetMousePosCommand::new(f));
        self.inner.push(cmd);
        self
    }

    pub fn build(self, ui: &imgui::Ui) {
        ImGuiCommand::Chain(self.inner).build(ui);
    }

    pub fn extend(&mut self, other: Self) {
        self.inner.extend(other.inner);
    }

    pub(crate) fn drain(&mut self) -> Self {
        Self { inner: self.inner.drain(..).collect_vec() }
    }
}

#[derive(Clone)]
enum ImGuiCommand {
    Separator,
    Text(ImGuiTextCommand),
    TextWrapped(ImGuiTextCommand),
    Window(ImGuiWindowCommand),

    GetMousePos(ImGuiGetMousePosCommand),

    Chain(Vec<ImGuiCommand>),
}

impl ImGuiCommand {
    pub(crate) fn build(self, ui: &imgui::Ui) {
        match self {
            ImGuiCommand::Separator => ui.separator(),
            ImGuiCommand::Text(cmd) | ImGuiCommand::TextWrapped(cmd) => {
                cmd.build(ui);
            },
            ImGuiCommand::Window(cmd) => { cmd.build(ui); },
            ImGuiCommand::GetMousePos(cmd) => {
                for cmd in (cmd.f)(ui.io().mouse_pos.into()).inner {
                    cmd.build(ui);
                }
            }
            ImGuiCommand::Chain(chain) => {
                for cmd in chain {
                    cmd.build(ui);
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct ImGuiWindowCommand {
    name: String,
    size: Vec2,
    size_cond: Condition,
    then: ImGuiCommandChain,
}

impl ImGuiWindowCommand {
    fn into_command(self) -> ImGuiCommand {
        ImGuiCommand::Window(self)
    }

    fn new(name: String, then: ImGuiCommandChain) -> Self {
        Self {
            name,
            size: Vec2::zero(),
            size_cond: Condition::Never,
            then
        }
    }

    #[must_use]
    pub fn size(mut self, size: impl Into<Vec2>, condition: Condition) -> Self {
        self.size = size.into();
        self.size_cond = condition;
        self
    }

    fn build(self, ui: &imgui::Ui) -> bool {
        ui.window(self.name)
            .size(self.size.as_f32_lossy(), self.size_cond)
            .build(|| {
                for cmd in self.then.inner {
                    cmd.build(ui);
                }
            })
            .is_some()
    }
}

#[derive(Clone)]
struct ImGuiTextCommand {
    text: String,
}

impl ImGuiTextCommand {
    fn new(text: String) -> Self {
        Self { text }
    }

    fn build(self, ui: &imgui::Ui) {
        ui.text(self.text);
    }
}

#[derive(Clone)]
struct ImGuiGetMousePosCommand {
    f: fn(Vec2) -> ImGuiCommandChain,
}

impl ImGuiGetMousePosCommand {
    fn new(f: fn(Vec2) -> ImGuiCommandChain) -> Self { Self { f } }
}
