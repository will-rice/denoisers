"""Gradio demo for denoisers, deployed to the wrice/audio_denoiser Space."""

import tempfile
from pathlib import Path

import gradio as gr

from denoisers import UNet1DModel, WaveUNetModel
from denoisers.inference import denoise_file

MODELS = [
    ("UNet1D · 48 kHz", "wrice/unet1d-vctk-48khz"),
    ("WaveUNet · 48 kHz", "wrice/waveunet-vctk-48khz"),
    ("WaveUNet · 24 kHz", "wrice/waveunet-vctk-24khz"),
]

DESCRIPTION = """
# Audio Denoiser

Upload a noisy speech recording, pick a model, and listen to or download the
denoised result. Models are from
[denoisers](https://github.com/will-rice/denoisers), trained on VCTK.
"""


def denoise(model_name: str, audio_path: str | None) -> str | None:
    """Denoise the uploaded audio file and return the denoised file."""
    if audio_path is None:
        return None

    model_class = UNet1DModel if "unet1d" in model_name else WaveUNetModel
    model = model_class.from_pretrained(model_name)

    # A directory per request so concurrent users don't overwrite each other.
    output_path = Path(tempfile.mkdtemp()) / f"{Path(audio_path).stem}_denoised.wav"
    denoise_file(model, audio_path, output_path)
    return str(output_path)


with gr.Blocks(title="Audio Denoiser") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row(equal_height=True):
        with gr.Column():
            model_name = gr.Dropdown(MODELS, value=MODELS[0][1], label="Model")
            noisy = gr.Audio(sources=["upload"], type="filepath", label="Noisy audio")
            submit = gr.Button("Denoise", variant="primary")
        with gr.Column():
            denoised = gr.Audio(
                type="filepath",
                buttons=["download"],
                interactive=False,
                label="Denoised audio",
            )
    submit.click(denoise, inputs=[model_name, noisy], outputs=denoised)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="green"))
