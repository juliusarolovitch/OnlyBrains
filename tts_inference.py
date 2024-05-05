import sys
from elevenlabs import play
from elevenlabs.client import ElevenLabs


def generate_and_play_speech(text):
    client = ElevenLabs(api_key="93e7177e97f99466ec67d94107dc509f")

    audio = client.generate(
        text=text,
        voice="Julius",
        model="eleven_multilingual_v2"
    )

    play(audio)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        transcribed_text = " ".join(sys.argv[1:])
        generate_and_play_speech(transcribed_text)
    else:
        print("Usage: python script.py <transcribed_text>")
