from google import genai
from google.genai import types

def generate():
    client = genai.Client(
        vertexai=True,
        project="mrc-quant-ml",
        location="global",
    )
    
    video_part = types.Part(
                file_data=types.FileData(file_uri='https://www.youtube.com/watch?v=XEzRZ35urlk'),
                video_metadata=types.VideoMetadata(
                    fps=1
                    # start_offset='1250s',
                    # end_offset='1570s'
                )
            ),
    
    model = "gemini-2.0-flash"
    
    contents = [
        types.Content(
            role="user",
            parts=[
                video_part,
                types.Part.from_text(
                    text="Analyze this video from 1-5 seconds and provide a summary"
                )
            ]
        )
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=8192,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", 
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ]
    )
    
    try:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
    except Exception as e:
        print(f"Generation failed: {e}")


def generate2():

    client = genai.Client(
        vertexai=True,
        project="mrc-quant-ml",
        location="global",
    )
    from google.genai.types import VideoMetadata
    from google.genai.types import Part

    def query_video_segment(video_uri, start_time, end_time, fps, user_question):
        """
        Query specific video segments with custom metadata
        
        Args:
            video_uri: Google Cloud Storage URI
            start_time: Start offset in seconds
            end_time: End offset in seconds  
            fps: Frames per second for sampling
            user_question: User's query about the segment
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                Part.from_uri(
                    file_uri=video_uri,
                    mime_type="video/mp4",
                    video_metadata=VideoMetadata(
                        start_offset=f"{start_time}s",
                        end_offset=f"{end_time}s",
                        fps=fps
                    )
                ),
                user_question
            ]
        )
        
        return response.text

    # Example usage
    result = query_video_segment(
        video_uri="gs://my-bucket/conference-recording.mp4",
        start_time=1800,  # 30 minutes
        end_time=2400,    # 40 minutes  
        fps=1,
        user_question="What are the main topics discussed in this segment?"
    )
    print(result)


def generate3():
    client = genai.Client(
        vertexai=True,
        project="mrc-quant-ml",
        location="global",
    )
    from google.genai.types import Part, VideoMetadata, FileData


    response = client.models.count_tokens(
        model="gemini-2.5-flash",
        contents=[
            Part(
                video_metadata=VideoMetadata(fps=1),
                file_data=FileData(
                    file_uri="gs://mrc-quant-ml-video-analysis/test_video_1.mov",
                    mime_type="video/mp4",
                ),
            )
        ],
    )
    print(response)

def generate4():
    client = genai.Client(
        vertexai=True,
        project="mrc-quant-ml",
        location="global",
    )
    from google.genai.types import Part, VideoMetadata, FileData


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            Part(
                video_metadata=VideoMetadata(fps=1, start_offset="1s", end_offset="150s"),
                file_data=FileData(
                    file_uri="gs://mrc-quant-ml-video-analysis/videoplayback.mp4",
                    mime_type="video/mp4",
                ),
            ),
            "Analyze this video and provide a summary."
        ],
    )
    print(response)


if __name__ == "__main__":
    generate4()
