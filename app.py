import os
import tempfile
import gradio as gr
import whisper
from TTS.api import TTS
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# モデルロード
asr_model = whisper.load_model("base")
tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def speech_to_text(audio_path):
    sound = AudioSegment.from_file(audio_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sound.export(tmp.name, format="wav")
        result = asr_model.transcribe(tmp.name)
    os.remove(tmp.name)
    return result["text"]

def generate_reply(history):
    query = history[-1][0]
    inputs = tokenizer.encode(query + tokenizer.eos_token, return_tensors="pt")
    outputs = chat_model.generate(inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    history[-1][1] = response
    return history, response

def synthesize_speech(text):
    tts_model.tts_to_file(text=text, file_path="output.wav")
    return "output.wav"

def handle_audio(audio_path, chat_history):
    transcribed = speech_to_text(audio_path)
    chat_history.append([transcribed, ""])  # 返信前に空文字を追加
    chat_history, reply = generate_reply(chat_history)
    audio_response = synthesize_speech(reply)
    return chat_history, audio_response

with gr.Blocks() as demo:
    gr.Markdown("## 🎤 英会話Bot (APIキー不要)")
    chatbot = gr.Chatbot(label="💬 会話ログ")
    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ 話しかけてね")
    audio_output = gr.Audio(label="🔊 AIの音声返答", interactive=False)
    state = gr.State([])

    audio_input.change(  # 明示的にイベントを設定
        fn=handle_audio,
        inputs=[audio_input, state],
        outputs=[chatbot, audio_output]
    )

demo.launch()


