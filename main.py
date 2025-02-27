import spaces
import torch
import gradio as gr
import time
import numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperTokenizer,
    pipeline,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
MODEL_NAME = "openai/whisper-large-v3-turbo"


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
)
model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME)

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=10,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)


@spaces.GPU
def stream_transcribe(stream, new_chunk, output, prev_output):
    start_time = time.time()
    try:
        sr, y = new_chunk

        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)

        # skip silent audio input
        cleanedChunk = y[y != 0]
        if len(cleanedChunk) == 0:
            return stream, output, prev_output, f"{0:.2f}"

        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        if stream is None:
            stream = []
        if len(stream)>5:
            stream = []
            prev_output +=  output+"\n" 

        stream.append(y)

        transcription = pipe({"sampling_rate": sr, "raw": np.concatenate(stream)})[
            "text"
        ]
        end_time = time.time()
        latency = end_time - start_time

        return stream, transcription, prev_output, f"{latency:.2f}"

    except Exception as e:
        print(f"Error during Transcription: {e}")
        return stream, e, "Error"



def clear():
    return ""


def clear_state():
    return None


with gr.Blocks() as microphone:
    with gr.Column():
        gr.Markdown(
            f"# Realtime Whisper Large V3 Turbo"
        )
        with gr.Row():
            input_audio_microphone = gr.Audio(streaming=True)
            prev_output = gr.Textbox(label="log", value="")
            output = gr.Textbox(label="Transcription", value="")
            latency_textbox = gr.Textbox(
                label="Latency (seconds)", value="0.0", scale=0
            )
        with gr.Row():
            clear_button = gr.Button("Clear Output")
        state = gr.State()
        input_audio_microphone.stream(
            stream_transcribe,
            [state, input_audio_microphone, output, prev_output],
            [state, output,prev_output, latency_textbox],
            time_limit=30,
            stream_every=2,
            concurrency_limit=None,
        )
        clear_button.click(clear_state, outputs=[state]).then(clear, outputs=[output]).then(clear, outputs=[prev_output])


with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.TabbedInterface([microphone], ["Microphone"])

demo.launch()
