# Voice Mid-Session Integration

The voice enhancement is integrated into the existing Customer Support Ticket Agent. It does not create a second conversational application and does not bypass the existing `/chat` workflow.

## Architecture

```text
Browser microphone
       |
       v
Streamlit st.audio_input
       |  POST /voice/transcribe
       v
FastAPI VoicePipeline
       |
       v
Faster-Whisper STT adapter
       |
       v
Editable transcript
       |  explicit Submit
       v
Existing POST /chat
       |
       v
Existing LangGraph + RAG + ticket workflow
       |
       v
Displayed assistant response
       |  explicit speaker click
       v
POST /voice/synthesize
       |
       v
Edge TTS adapter -> MP3 -> selected chat message
```

## STT choice

The project uses **faster-whisper** as an individual local STT component. It produces text only; the transcript is returned to Streamlit for editing and is submitted through the existing `/chat` endpoint.

Configuration: `STT_MODEL`, `STT_DEVICE`, and `STT_COMPUTE_TYPE`. The model is loaded lazily on the first transcription request.

## TTS choice

The project uses **Edge TTS** as an individual TTS component. It returns `audio/mpeg` bytes and does not require an application API key. The exact displayed assistant response is sent to `/voice/synthesize` only when its speaker control is clicked.

## UI behavior

1. User records with the browser/system microphone.
2. Audio is sent to `/voice/transcribe`.
3. The transcript is displayed in an editable text area.
4. User explicitly clicks **Submit transcript**.
5. The confirmed transcript is sent through the existing `/chat` request with the same `session_id`.
6. Every assistant response displays a `🔊 Speak response` control.
7. TTS is never autoplayed.
8. Generated audio is cached in the current Streamlit session by the response message ID.
9. TTS failures leave the text response and ticket details visible.

## API

### `POST /voice/transcribe`
Multipart form field: `file`.

Example response:

```json
{
  "success": true,
  "transcript": "My payment was charged twice",
  "processing_time_ms": 820
}
```

### `POST /voice/synthesize`
JSON:

```json
{
  "text": "Ticket created successfully."
}
```

Response: MP3 audio bytes with `audio/mpeg` media type. A processing-time header is returned as `X-Processing-Time-Ms`.

Validation and downstream voice failures are returned as clear JSON HTTP errors.

## Error behavior

- Empty audio -> HTTP 400.
- Unsupported audio MIME type -> HTTP 415.
- Unintelligible/silent audio -> HTTP 502 with a recoverable message.
- STT provider/decoder failure -> HTTP 502.
- TTS provider failure -> HTTP 502; the text response remains intact in Streamlit.

## Testing

The voice tests cover fake-adapter transcription, empty audio, STT failure, fake synthesis, TTS failure, timing/media type, and response-ID-to-audio association. Existing `/chat` tests remain in place as the typed-chat regression suite.

## Demo requirement

For the final recording, demonstrate one real microphone recording, edit the generated transcript if needed, submit it, receive the normal agent response, then click the speaker icon for that response and play the resulting audio.
