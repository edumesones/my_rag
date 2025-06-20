import gradio as gr
from app.gradio_ui.zero_shot_ui import zero_shot
from app.gradio_ui.compiled_ui import compiled
from app.gradio_ui.optimize_pipeline import compile_pipeline
from app.gradio_ui.retrieve_only_ui import retrieve_only_compilation


gradio_iface = gr.TabbedInterface(
    [retrieve_only_compilation,zero_shot, compiled, compile_pipeline],
    ["Retrieve only","Zero Shot Query", "Compiled Query", "Optimize Pipeline"],
    title="DSPy",
)
