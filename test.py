from youtube_transcript_api import YouTubeTranscriptApi
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import re

# 1. group the transcript by theme and questions or input from users that could lead to that statement being said.
# 2. tag each of them together for the jsonl
# 3. now use this  to  Create JSONL for Chat Completion for finetuning purposes
# it should be in english. the questions.
# system should be along the lines of you are tobbe, an enthusiastic car wash detailer sharing your process for cleaning and restoring vehicles.


client = OpenAI()

load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Function to fetch and save english transcript
def get_english_transcript(video_id, output_file):
    try:
        # Get the transcript for the video
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        # Save the transcript to a file
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in transcript:
                text = entry['text']
                if not re.search(r'\bMusic\b|♪|♫', text, re.IGNORECASE):  # Adjust pattern as needed
                    f.write(f"{entry['start']}: {text}\n")
        
        print(f"english transcript saved to {output_file}.")
    except Exception as e:
        print(f"Error fetching transcript: {e}")


# def preprocess_transcript_to_sentences(input_files, output_file="processed_transcript.jsonl"):
#     combined_text = ""

#     # Step 1: Combine all text from input files into one large chunk
#     for input_file in input_files:
#         with open(input_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if ':' in line:
#                     text = line.split(':', 1)[1].strip()  # Get text after the timestamp
#                     combined_text += " " + text

#     # Step 2: Split the combined text into sentences
#     sentences = re.split(r'(?<=[.!?]) +', combined_text.strip())

#     # Step 4: Save the chunks to a JSONL file
#     with open(output_file, 'w', encoding='utf-8') as out_f:
#         for idx, sentence in enumerate(sentences):
#             json_entry = {"chunk_id": idx + 1, "text": sentence}
#             out_f.write(json.dumps(json_entry) + '\n')

#     print(f"Processed transcript saved to {output_file}")

# Function to preprocess transcript into JSONL format
def preprocess_transcript_to_jsonl(input_files, output_file="combined_transcript.jsonl"):
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for input_file in input_files:
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Split the line at the first occurrence of ':' and take the second part
                        if ':' in line:
                            text = line.split(':', 1)[1].strip()  # Get text after the timestamp
                            # Create a JSON object
                            json_entry = {"prompt": "Respond in this tone:", "completion": text}
                            # Write the JSON object as a line in the output file
                            out_f.write(json.dumps(json_entry) + '\n')

        print("Combined transcript saved to", output_file)

    except Exception as e:
        print(f"Error processing transcript: {e}")

def fine_tune_model(dataset_path):
    print("Uploading and creating fine-tuning job...")
    try:
        # Check if the JSONL file exists and read it
        if not os.path.exists(dataset_path):
            print(f"Dataset file {dataset_path} does not exist.")
            return

        with open(dataset_path, "rb") as f:
            responses = client.files.create(file=f, purpose="fine-tune")

        # Create fine-tuning job
        fine_tune_job = client.fine_tuning.jobs.create(
            training_file=responses.id,
            model="gpt-4o-mini-2024-07-18"  # Replace with your desired model
        )
        print(f"Fine-tune job started. Job ID: {fine_tune_job.id}")

        # Wait for the fine-tuning job to complete
        while True:
            job_status = client.fine_tuning.jobs.retrieve(fine_tune_job.id)
            print(f"Current job status: {job_status}")
            if job_status.status in ['succeeded', 'failed']:
                break
            time.sleep(10)  # Wait for a while before checking again

        if job_status.status == 'succeeded':
            fine_tuned_model = job_status.fine_tuned_model
            print(f"Fine-tuning completed. Fine-tuned model ID: {fine_tuned_model}")
            return fine_tuned_model
        else:
            print("Fine-tuning failed.")
            return None

    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return None

def chat_with_model(fine_tuned_model):
    print("You can start chatting with the fine-tuned model. Type 'quit' or 'q' to exit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "q"]:
            print("Goodbye!")
            break
        if not user_input.strip():  # Check if user input is empty
            print("Please enter a valid message.")
            continue
        try:
            messages = [
                {"role": "system", "content": "Respond in the way you are trained to respond"},
                {"role": "user", "content": user_input}
            ]
            print("Messages being sent to the API:", messages)  # Debugging line
            print(f"Using model: {fine_tuned_model}")  # Debugging model name
            
            completion = client.chat.completions.create(
                model=fine_tuned_model,
                messages=messages
            )
            
            # Print the entire completion response
            print("Completion Response:", completion)
            
            # Check if the response contains choices
            if completion.choices:
                response = completion.choices[0].message.content
                print(f"Model: {response}\n")
            else:
                print("No choices returned in the completion response.")
                
        except Exception as e:
            print(f"Error generating response: {e}")

# Main function to execute the workflow
def main():
    video_urls = [
        # "https://www.youtube.com/watch?v=RTMQrX-g5X4",
        "https://www.youtube.com/watch?v=XrmyWILTesY&ab_channel=tershine",
        # "https://www.youtube.com/watch?v=bMrep-F0fsY&ab_channel=tershine",
        # "https://www.youtube.com/watch?v=k49aBReCTIU&ab_channel=tershine",
        # "https://www.youtube.com/watch?v=bhVKRhBVV5M&ab_channel=tershine"
    ]

    transcript_files = []

    # Fetch transcripts for multiple videos
    for idx, video_url in enumerate(video_urls):
        video_id = video_url.split("v=")[1]
        output_file = f"english_transcript_{idx + 1}.txt"
        get_english_transcript(video_id, output_file)
        transcript_files.append(output_file)

    # Preprocess all transcripts into a single JSONL file
    preprocess_transcript_to_jsonl(transcript_files)

    # Fine-tune the model
    fine_tuned_model = fine_tune_model("combined_transcript.jsonl")

    if fine_tuned_model:
        chat_with_model(fine_tuned_model)
    else:
        print("Fine-tuning was unsuccessful. Exiting.")

if __name__ == "__main__":
    main()