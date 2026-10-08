import hashlib
import os
import uuid

import httpx
import streamlit as st

from src.voice.playback import VoicePlaybackStore


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(60.0, connect=5.0)

st.set_page_config(page_title="Customer Support", page_icon="🎧", layout="centered")
st.title("🎧 Customer Support")
st.caption("Grounded answers + multi-turn tickets + optional microphone input and speaker playback")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""
if "voice_audio_hash" not in st.session_state:
    st.session_state.voice_audio_hash = None
if "voice_transcribing" not in st.session_state:
    st.session_state.voice_transcribing = False
if "tts_store" not in st.session_state:
    st.session_state.tts_store = VoicePlaybackStore()
if "tts_in_progress" not in st.session_state:
    st.session_state.tts_in_progress = None

with st.sidebar:
    st.subheader("Conversation")
    st.caption(f"Session: {st.session_state.session_id}")
    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.voice_transcript = ""
        st.session_state.voice_audio_hash = None
        st.session_state.tts_store = VoicePlaybackStore()
        st.rerun()


def send_chat_message(prompt: str) -> None:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        st.warning("Please enter a message before submitting.")
        return

    user_message = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": clean_prompt,
    }
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.write(clean_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Support agent is working…"):
                response = httpx.post(
                    f"{API_BASE_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": clean_prompt},
                    timeout=TIMEOUT,
                )

            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", "Unknown backend error")
                except ValueError:
                    detail = response.text or "Unknown backend error"
                if response.status_code == 503:
                    st.error("A required support component is currently unavailable. Please start/check the backend and model service.")
                elif response.status_code == 422:
                    st.error(f"Input validation failed: {detail}")
                else:
                    st.error(f"Support request failed: {detail}")
                return

            try:
                payload = response.json()
            except ValueError:
                st.error("The backend returned an invalid response. Please try again.")
                return

            assistant_text = payload.get("response", "").strip()
            if not assistant_text:
                st.error("The backend returned an empty support response.")
                return

            assistant_message = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": assistant_text,
                "sources": payload.get("sources", []) or [],
                "ticket_id": payload.get("ticket_id"),
            }
            st.session_state.messages.append(assistant_message)
            render_assistant(assistant_message)
        except httpx.ConnectError:
            st.error("Could not connect to FastAPI. Start the backend and try again.")
        except httpx.TimeoutException:
            st.error("The support backend timed out. Please try again.")
        except httpx.HTTPError:
            st.error("A network error occurred while contacting the support backend.")
        except Exception:
            st.error("Something unexpected happened while processing the request.")


def render_assistant(message: dict) -> None:
    """Render one assistant response with a per-message, on-demand speaker control."""
    st.write(message["content"])
    if message.get("sources"):
        st.caption("Sources: " + ", ".join(message["sources"]))
    if message.get("ticket_id"):
        st.success(f"Ticket created: {message['ticket_id']}")

    message_id = message["id"]
    cached_audio = st.session_state.tts_store.get(message_id)
    button_disabled = st.session_state.tts_in_progress == message_id
    if cached_audio:
        st.audio(cached_audio, format="audio/mpeg")
        if st.button("🔊 Play again", key=f"tts_{message_id}", disabled=button_disabled):
            st.audio(cached_audio, format="audio/mpeg")
        return

    if st.button("🔊 Speak response", key=f"tts_{message_id}", disabled=button_disabled):
        st.session_state.tts_in_progress = message_id
        try:
            with st.spinner("Generating response audio…"):
                tts_response = httpx.post(
                    f"{API_BASE_URL}/voice/synthesize",
                    json={"text": message["content"]},
                    timeout=TIMEOUT,
                )
            if tts_response.status_code >= 400:
                try:
                    detail = tts_response.json().get("detail", "Text-to-speech failed")
                except ValueError:
                    detail = "Text-to-speech failed"
                st.session_state.tts_store.set_error(message_id, detail)
                st.error(f"Voice playback unavailable: {detail}")
            else:
                st.session_state.tts_store.put(message_id, tts_response.content)
                st.audio(tts_response.content, format="audio/mpeg")
        except httpx.HTTPError:
            st.session_state.tts_store.set_error(message_id, "Could not contact the voice synthesis endpoint.")
            st.error("Voice playback failed. The text response is still available.")
        finally:
            st.session_state.tts_in_progress = None
    elif st.session_state.tts_store.get_error(message_id):
        st.caption(f"Previous playback attempt failed: {st.session_state.tts_store.get_error(message_id)}")


# Existing text chat history remains unchanged and voice is additive.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant(message)
        else:
            st.write(message["content"])

# Microphone input sits immediately above the standard text composer. It only
# creates a transcript; no /chat call happens until the user explicitly submits it.
st.divider()
st.subheader("🎙️ Voice input")
st.caption("Record → review/edit transcript → submit to the same text agent")
recorded_audio = st.audio_input("Record a support message", key="voice_recorder")

if recorded_audio is not None:
    audio_bytes = recorded_audio.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if audio_hash != st.session_state.voice_audio_hash:
        st.session_state.voice_audio_hash = audio_hash
        st.session_state.voice_transcript = ""
        st.session_state.pop("voice_transcript_editor", None)
        with st.spinner("Transcribing microphone input…"):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/voice/transcribe",
                    files={
                        "file": (
                            recorded_audio.name or "microphone.wav",
                            audio_bytes,
                            recorded_audio.type or "audio/wav",
                        )
                    },
                    timeout=TIMEOUT,
                )
                if response.status_code >= 400:
                    try:
                        detail = response.json().get("detail", "Speech transcription failed")
                    except ValueError:
                        detail = "Speech transcription failed"
                    st.error(f"Voice input could not be transcribed: {detail}")
                else:
                    payload = response.json()
                    st.session_state.voice_transcript = payload.get("transcript", "").strip()
                    st.session_state.voice_transcript_editor = st.session_state.voice_transcript
                    if not st.session_state.voice_transcript:
                        st.warning("No intelligible speech was detected. Please record again.")
            except httpx.HTTPError:
                st.error("Could not reach the voice transcription endpoint.")
            except Exception:
                st.error("Voice transcription failed. Typed chat is still available.")

if st.session_state.voice_transcript:
    edited_transcript = st.text_area(
        "Editable transcript — correct anything before sending",
        value=st.session_state.voice_transcript,
        key="voice_transcript_editor",
        height=110,
    )
    submit_col, clear_col = st.columns(2)
    with submit_col:
        if st.button("Submit transcript", type="primary", use_container_width=True):
            st.session_state.voice_transcript = edited_transcript
            send_chat_message(edited_transcript)
            st.session_state.voice_audio_hash = None
            st.session_state.voice_transcript = ""
            st.session_state.pop("voice_transcript_editor", None)
            st.rerun()
    with clear_col:
        if st.button("Discard", use_container_width=True):
            st.session_state.voice_audio_hash = None
            st.session_state.voice_transcript = ""
            st.session_state.pop("voice_transcript_editor", None)
            st.rerun()

if prompt := st.chat_input("How can we help?"):
    send_chat_message(prompt)
