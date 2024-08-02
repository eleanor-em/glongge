use imgui::Condition;
use crate::core::ObjectTypeEnum;
use crate::core::update::ObjectHandler;
use crate::gui::command::ImGuiCommandChain;

pub fn build<ObjectType: ObjectTypeEnum>(
    object_handler: &ObjectHandler<ObjectType>
) -> ImGuiCommandChain {
    let mut cmd = ImGuiCommandChain::new()
        .window(
            "Object Tree",
            |win| {
                win.size([300., 110.], Condition::FirstUseEver)
            },
            ImGuiCommandChain::new()
        )
        .window(
            "Collision",
            |win| {
                win.size([300, 110], Condition::FirstUseEver)
                    .position([200, 0], Condition::FirstUseEver)
            },
            ImGuiCommandChain::new()
        );

    object_handler.depth_first_with(
        |parent, depth: usize| {
            let name = "\t".repeat(depth - 1) + &format!("{:?}", parent.borrow().get_type());
            cmd = cmd.clone().window_default(
                "Object Tree",
                ImGuiCommandChain::new()
                    .text(name));
        },
        |_, depth| depth + 1
    );
    cmd
}
