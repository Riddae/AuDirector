#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Assuming your core logic is saved as workflow.py
from workflow import AudioBookWorkflow, WorkflowResult

def main():
    # 1. Configure Command Line Argument Parser
    parser = argparse.ArgumentParser(description="AuDirector: Automated Audiobook Generation Workflow")
    parser.add_argument("input", type=str, help="The story description or prompt for the audiobook")
    parser.add_argument("--output", "-o", type=str, default="projects", help="Root directory for all project outputs")
    parser.add_argument("--name", "-n", type=str, default="my_audiobook", help="Project name prefix")
    
    args = parser.parse_args()

    # 2. Generate Unique Output Path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_folder_name = f"{args.name}_{timestamp}"
    output_base_dir = Path(args.output) / project_folder_name
    
    # Ensure the directory exists
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 60)
    print(f"Initializing AuDirector Workflow")
    print(f"Output Directory: {output_base_dir}")
    print(f"User Input: {args.input}")
    print("-" * 60)

    # 3. Initialize and Execute the Workflow
    try:
        workflow = AudioBookWorkflow(output_base_dir=str(output_base_dir))
        
        print("Processing... This may take a few minutes (LLM generation and voice synthesis in progress).")
        result: WorkflowResult = workflow.run_complete_workflow(args.input)

        # 4. Handle Execution Result
        if result.success:
            print("\nGeneration Successful")
            print(f"Total Execution Time: {result.execution_time:.2f} seconds")
            print(f"Final Master Audio: {result.output_files.get('final_mix')}")
            print("\nResources saved at:")
            print(f" - Speech Clips: {result.output_files.get('speech_dir')}")
            print(f" - BGM Tracks: {result.output_files.get('bgm_dir')}")
            print(f" - SFX Tracks: {result.output_files.get('sfx_dir')}")
        else:
            print("\nWorkflow Failed")
            print(f"Error Message: {result.message}")
            sys.exit(1)

    except Exception as e:
        print(f"\nUnexpected Exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()