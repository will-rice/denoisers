"""Gradio demo for denoisers, deployed to the wrice/audio_denoiser Space."""

import tempfile

import gradio as gr

from denoisers import UNet1DModel, WaveUNetModel
from denoisers.inference import denoise_file

MODELS = [
    "wrice/unet1d-vctk-48khz",
    "wrice/waveunet-vctk-48khz",
    "wrice/waveunet-vctk-24khz",
]


def denoise(model_name: str, audio_path: str | None) -> str | None:
    """Denoise the uploaded audio with the selected model."""
    if audio_path is None:
        return None

    model_class = UNet1DModel if "unet1d" in model_name else WaveUNetModel
    model = model_class.from_pretrained(model_name)

    # A file per request so concurrent users don't overwrite each other.
    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    denoise_file(model, audio_path, output_path)
    return output_path


demo = gr.Interface(
    fn=denoise,
    inputs=[gr.Dropdown(choices=MODELS, value=MODELS[0]), gr.Audio(type="filepath")],
    outputs=gr.Audio(type="filepath"),
)

if __name__ == "__main__":
    demo.launch()
