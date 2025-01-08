import os
import json
import requests
import glob
import shutil
import time
import contextlib

from datetime import datetime
from IPython.display import clear_output
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(".env")
should_cleanup: bool = True

# Load the Azure OpenAI API key and version from the environment
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

# Check if the assistant already exists by name
assistant_name = "Insurance Submission Extractor"
assistants = client.beta.assistants.list()
existing_assistant = next((asst for asst in assistants if asst.name == assistant_name), None)

if existing_assistant:
    assistant = existing_assistant
else:
    # Create a new AI assistant for processing insurance submissions
    assistant = client.beta.assistants.create(
        name=assistant_name,
        instructions="You are an AI assistant who is an expert in extracting data from insurance submissions which includes PDF and excel files. You are expected to extract the following information: 1. Named Insured 2. DBA Name 3. Coverage or Peril or Exposure  4. Policy InceptionDate 5. Policy ExpirationDate 6. StreetAddress, City, State, Zip, County details of the Property 7. TotalInsuredValue, OccupancyCode of the property/properties. You will find the StreetAddress, City, State, Zip, County, TotalInsuredValue and Occupancycode attributes in the tab named 'SOV APP' in the excel file. To identify the correct columns from excel files, please examine a larger section of the data (e.g., the first 20 rows) to identify where the meaningful data starts and which columns contain the information we're interested in. For TotalInsuredValue and OccupancyCode attributes, provide unique rows only.",
        model=os.getenv("GPT4_DEPLOYMENT_NAME"),
        tools=[{"type": "file_search"},{"type":"code_interpreter"}]
    )

# Define the directory for submissions and processed data
SUBMISSIONS_DIR = "submissions"
DATA_DIR = "data"
PROCESSED_DIR = "processed"

# Ensure the directories exist
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define the function to monitor the status of the thread run
def monitor_thread_run(client, thread_id, run_id):
    status = "in_progress"
    start_time = time.time()
    while status not in ["completed", "cancelled", "expired", "failed"]:
        time.sleep(5)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id,run_id=run_id)
        print("Elapsed time: {} minutes {} seconds".format(int((time.time() - start_time) // 60), int((time.time() - start_time) % 60)))
        status = run.status
        print(f'Status: {status}')
        clear_output(wait=True)
    return run
    
# Define the function to process the submission for PDF, word files
def process_submission_pdf(submission_folder,assistant_id):
    # Create a new vector store for the submission documents
    vector_store = client.beta.vector_stores.create(name="Submission Documents Vector Store")
    
    # Ready the files for upload to OpenAI
    file_paths = glob.glob(os.path.join(DATA_DIR, "*.pdf")) + glob.glob(os.path.join(DATA_DIR, "*.docx"))
    with contextlib.ExitStack() as stack:
        file_streams = [stack.enter_context(open(path, "rb")) for path in file_paths]
 
        # Use the upload and poll SDK helper to upload the files, add them to the vector store,
        # and poll the status of the file batch for completion.
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
        )
          
    # Print the status and file counts of the batch
    print(file_batch.status)
    print(file_batch.file_counts)
    
    # Update the assistant to use the new vector store
    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    # Create a thread using the vector store file and publish the output
    thread = client.beta.threads.create()
    message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Extract the following attributes from the submission: NamedInsured, DBA Name, RenewalofAccountID, Coverage or Peril or Exposure, InceptionDate, ExpirationDate"
    )
    
    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    )
       
    # Monitor the status of the thread run
    run = monitor_thread_run(client, thread.id, run.id)
    
    # Once the thread run is completed, retrieve the messages
    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )
    print(messages.model_dump_json(indent=2))    

    # Extract the data from the messages
    data = json.loads(messages.model_dump_json(indent=2))  # Load JSON data into a Python object
    output_data = data['data'][0]['content'][0]['text']['value']  # Adjusted to access the correct content

    # Write the 'value' field to a file with extra spaces and JSON formatting
    submission_output_path = os.path.join(DATA_DIR, "submission.txt")
    with open(submission_output_path, "w",encoding="utf-8") as file:
        file.write(json.dumps(output_data, indent=2))  # Convert list to JSON string

    print(f"Data from {submission_folder} written to submission.txt")
    
    # Delete the vector store after processing the submission
    client.beta.vector_stores.delete(vector_store.id)

# Define the function to process a submission
def process_submission_xlsx(submission_folder,assistant_id):
    
    # Ready the files for upload to OpenAI
    xlsx_files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))+ glob.glob(os.path.join(DATA_DIR, "*.xls"))+ glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if not xlsx_files:
        print("No xlsx, xls, or csv files found in the data directory.")
        return

    xlsx_file_path = xlsx_files[0]

    with contextlib.ExitStack() as stack:
        xlsx_file = stack.enter_context(open(xlsx_file_path, "rb"))
        message_file = client.files.create(
            file=xlsx_file, purpose="assistants"
        )

    # Create a thread and attach the file to the message
    thread = client.beta.threads.create(
    messages=[
    {
    "role": "user",
    "content": "Extract the following property attributes from the SOV and provide the unique rows of the attributes : StreetAddress, City, State, Zip, County",
    # Attach the new file to the message.
    "attachments": [
    { "file_id": message_file.id, "tools": [{"type": "code_interpreter"}] }
        ],
    }
    ]
    )
    
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    
    # Monitor the status of the thread run
    run = monitor_thread_run(client, thread.id, run.id)
    
    # Once the thread run is completed, retrieve the messages
    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )
    print(messages.model_dump_json(indent=2))    

    # Extract the data from the messages
    data = json.loads(messages.model_dump_json(indent=2))  # Load JSON data into a Python object
    output_data = data['data'][0]['content'][0]['text']['value']  # Adjusted to access the correct content
    
    # Write the 'value' field to a file with extra spaces and JSON formatting
    submission_output_path = os.path.join(DATA_DIR, "submission.txt")
    with open(submission_output_path, "a", encoding="utf-8") as file:
        file.write("\n\n------------------------\n\n" + json.dumps(output_data, indent=2) + "\n\n---------------------------------")  # Convert list to JSON string

    print(f"Data from {submission_folder} written to submission.txt")

    # Create a new message to extract the TotalInsuredValue and OccupancyCode
    message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Extract the following attributes from the submission: TotalInsuredValue, OccupancyCode"
    )

    run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    )
    
    # Monitor the status of the thread run
    run = monitor_thread_run(client, thread.id, run.id)

    # Once the thread run is completed, retrieve the messages
    messages = client.beta.threads.messages.list(
    thread_id=thread.id
    )
    print(messages.model_dump_json(indent=2))    

    # Extract the data from the messages
    data = json.loads(messages.model_dump_json(indent=2))  # Load JSON data into a Python object
    output_data = data['data'][0]['content'][0]['text']['value']  # Adjusted to access the correct content
    
    # Write the 'value' field to a file with extra spaces and JSON formatting
    submission_output_path = os.path.join(DATA_DIR, "submission.txt")
    with open(submission_output_path, "a", encoding="utf-8") as file:
        file.write("\n\n" + json.dumps(output_data, indent=2) + "\n\n")  # Convert list to JSON string

    print(f"Data from {submission_folder} written to submission.txt")


def main():
    assistant_id = assistant.id

    # Loop through all submission folders in the SUBMISSIONS_DIR
    for submission_folder in os.listdir(SUBMISSIONS_DIR):
        submission_path = os.path.join(SUBMISSIONS_DIR, submission_folder)
        
        # Check if the path is a directory
        if os.path.isdir(submission_path):
            # Clear the DATA_DIR before copying new files
            for file_name in os.listdir(DATA_DIR):
                file_path = os.path.join(DATA_DIR, file_name)
                os.remove(file_path)
            
            # Copy the submission files to the DATA_DIR
            for file_name in os.listdir(submission_path):
                file_path = os.path.join(submission_path, file_name)
                if file_name.endswith((".pdf", ".docx", ".xlsx")):
                    shutil.copy(file_path, DATA_DIR)

            
            # Process the submission
            process_submission_pdf(submission_folder, assistant_id)
            process_submission_xlsx(submission_folder, assistant_id)

            # Read the content of submission.txt
            submission_output_path = os.path.join(DATA_DIR, "submission.txt")
            with open(submission_output_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Replace Unicode characters and newlines
            text = text.replace('\\u3010', '[')
            text = text.replace('\\u3011', ']')
            text = text.replace('\\u2020', '†')
            text = text.replace('\\n', '\n')

            # Write the cleaned content back to submission.txt
            with open(submission_output_path, "w", encoding="utf-8") as file:
                file.write(text)

            # Move the processed submission folder to the PROCESSED_DIR
            processed_submission_path = os.path.join(PROCESSED_DIR, submission_folder)
            shutil.move(submission_path, processed_submission_path)
            
            # Generate a unique name for the submission.txt file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_submission_file = f"submission_{timestamp}.txt"
            
            # Move the submission.txt file to the processed directory with a unique name
            shutil.move(os.path.join(DATA_DIR, "submission.txt"), os.path.join(processed_submission_path, unique_submission_file))
           
        # Prompt the user to continue to the next submission
            input("Press Enter to process the next submission...")
    print("All submissions have been processed.")
    
    # # Delete the assistant after processing all submissions
    # client.beta.assistants.delete(assistant.id)

if __name__ == "__main__":
    main()
