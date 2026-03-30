import asyncio
import json
import time
import pandas as pd
import os
import sys

# Ensure the project root is in the path so we can import app.agents
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.agents.single_agent import run_single_agent_pipeline
from app.agents.multi_agent import run_multi_agent_pipeline

async def run_benchmark():
    """
    Orchestrates the benchmark run for both single-agent and multi-agent pipelines.
    """
    # 1. Load tasks from the root directory
    tasks_file = os.path.join(project_root, 'tasks.json')
    with open(tasks_file, 'r') as f:
        tasks = json.load(f)
    
    results = []
    
    # 2. Iterate through tasks
    # We only run a subset for the demonstration to save time, e.g., first 3 tasks.
    for item in tasks[:3]:
        task_id = item['id']
        task_text = item['task']
        
        # --- Run Single Agent ---
        print(f"\n[BENCHMARK] Task {task_id}: Running Single-Agent Pipeline...")
        start_time = time.time()
        try:
            # We need to exhaust the generator
            async for _ in run_single_agent_pipeline(task_text):
                pass
            latency_single = time.time() - start_time
            status_single = "SUCCESS"
        except Exception as e:
            print(f"Single-Agent Error: {e}")
            latency_single = time.time() - start_time
            status_single = f"ERROR: {e}"
        
        # --- Run Multi Agent ---
        print(f"\n[BENCHMARK] Task {task_id}: Running Multi-Agent Pipeline...")
        start_time = time.time()
        try:
            # We expect a generator, so we must iterate through it
            async for _ in run_multi_agent_pipeline(task_text):
                pass
            latency_multi = time.time() - start_time
            status_multi = "SUCCESS"
        except Exception as e:
            print(f"Multi-Agent Error: {e}")
            latency_multi = time.time() - start_time
            status_multi = f"ERROR: {e}"
            
        results.append({
            "task_id": task_id,
            "single_latency": latency_single,
            "single_status": status_single,
            "multi_latency": latency_multi,
            "multi_status": status_multi
        })
        
    # 3. Save results to CSV in the root directory
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(project_root, 'benchmark_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n[BENCHMARK] Completed! Results saved to '{results_csv}'.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
