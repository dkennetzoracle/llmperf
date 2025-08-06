import threading
import argparse
from collections.abc import Iterable
import itertools
import json
import os
from pathlib import Path
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple
import urllib3
import sys

import pandas as pd
import ray

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    randomly_sample_sonnet_lines_prompt,
    LLMPerfResults,
    sample_random_positive_int,
)
from tqdm import tqdm

from transformers import LlamaTokenizerFast
import mlflow

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    num_warmup_requests: int = 0,
    test_timeout_s=90,
    llm_api="openai",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        num_warmup_requests: The number of warmup requests to send before starting the benchmark.
            These requests are not included in the final statistics.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(11111)

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )
    get_token_length = lambda text: len(tokenizer.encode(text))
    
    if not additional_sampling_params:
        additional_sampling_params = {}

    completed_requests_lock = threading.Lock()
    completed_requests = []
    num_completed_requests = 0
    
    # make up prompts outside of send loop for faster benchmarking loop
    # Include warmup requests in total prompt generation
    total_requests = max_num_completed_requests + num_warmup_requests
    num_output_tokens_list = []
    prompts = []
    for i in range(total_requests):
        num_output_tokens = (sample_random_positive_int(
            mean_output_tokens, stddev_output_tokens
        ))
        num_output_tokens_list.append(num_output_tokens)

        prompts.append(randomly_sample_sonnet_lines_prompt(
            prompt_tokens_mean=mean_input_tokens,
            prompt_tokens_stddev=stddev_input_tokens,
            expect_output_tokens=num_output_tokens,
            tokenizer=tokenizer
        ))
    
    # Run warmup requests first if specified
    if num_warmup_requests > 0:
        print(f"Running {num_warmup_requests} warmup requests...")
        warmup_completed = 0
        warmup_lock = threading.Lock()
        warmup_pbar = tqdm(total=num_warmup_requests, desc="Warmup")
        
        def warmup_request(thread_index):
            nonlocal warmup_completed
            clients = construct_clients(llm_api=llm_api, num_clients=1)
            req_launcher = RequestsLauncher(clients)
            request_index = thread_index % num_warmup_requests
            
            while warmup_completed < num_warmup_requests:
                default_sampling_params = {"max_tokens": num_output_tokens_list[request_index]}
                default_sampling_params.update(additional_sampling_params)
                request_config = RequestConfig(
                    model=model,
                    prompt=prompts[request_index],
                    sampling_params=default_sampling_params,
                    llm_api=llm_api,
                )
                req_launcher.launch_requests(request_config)
                
                outs = req_launcher.get_next_ready()
                for out in outs:
                    with warmup_lock:
                        if warmup_completed < num_warmup_requests:
                            warmup_completed += 1
                            warmup_pbar.update(1)
                            request_index = (request_index + num_concurrent_requests) % num_warmup_requests
        
        # Launch warmup threads
        warmup_threads = []
        for i in range(num_concurrent_requests):
            thread = threading.Thread(target=warmup_request, args=(i,))
            warmup_threads.append(thread)
            thread.start()
        
        # Wait for warmup completion
        for thread in warmup_threads:
            thread.join()
        
        warmup_pbar.close()
        print("Warmup completed. Starting benchmark...")
    
    # Reset for actual benchmark
    start_time = time.monotonic()
    pbar = tqdm(total=max_num_completed_requests, desc="Benchmark")

    def launch_request(thread_index):
        nonlocal num_completed_requests
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
        # Start indexing after warmup prompts
        request_index = num_warmup_requests + (thread_index % max_num_completed_requests)

        while (
            time.monotonic() - start_time < test_timeout_s
            and num_completed_requests < max_num_completed_requests
        ):

            default_sampling_params = {"max_tokens": num_output_tokens_list[request_index] }
            default_sampling_params.update(additional_sampling_params)
            request_config = RequestConfig(
                model=model,
                prompt=prompts[request_index],
                sampling_params=default_sampling_params,
                llm_api=llm_api,
            )
            req_launcher.launch_requests(request_config)

            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                num_output_tokens = get_token_length(gen_text)
                with completed_requests_lock:
                    if num_completed_requests < max_num_completed_requests:
                        if num_output_tokens:
                            request_metrics[common_metrics.INTER_TOKEN_LAT] /= request_metrics[common_metrics.NUM_OUTPUT_TOKENS]
                        else:
                            request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                        request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                        request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                        request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = num_output_tokens / request_metrics[common_metrics.E2E_LAT]
                        all_metrics.append(request_metrics)
                        completed_requests.extend(all_metrics)
                        pbar.update(len(all_metrics))
                        num_completed_requests += len(all_metrics)
                        request_index = num_warmup_requests + ((request_index - num_warmup_requests + num_concurrent_requests) % max_num_completed_requests)

    threads = []
    for i in range(num_concurrent_requests):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    # check one last time that there are no remaining results to collect.
    clients = construct_clients(llm_api=llm_api, num_clients=1)
    req_launcher = RequestsLauncher(clients)
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        num_output_tokens = get_token_length(gen_text)
        with completed_requests_lock:
            if num_completed_requests < max_num_completed_requests:
                if num_output_tokens:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
                else:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = num_output_tokens / request_metrics[common_metrics.E2E_LAT]
                completed_requests.extend(request_metrics)

    print(f"Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret
        
    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]
    
    for key in [
        common_metrics.INTER_TOKEN_LAT,
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS
    ]:
        print(key)
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

    overall_output_throughput = df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum() / (end_time - start_time)

    print(f"Overall Output Throughput: {overall_output_throughput}")
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min
    
    return ret


def log_to_mlflow(
    mlflow_uri: str,
    model: str,
    mean_input_tokens: int,
    mean_output_tokens: int,
    num_concurrent_requests: int,
    summary: Dict[str, Any],
    user_metadata: Dict[str, Any],
    tensor_parallel_size: int,
    gpu_name: str
):
    """Log benchmark results to MLflow.
    
    Args:
        mlflow_uri: The MLflow tracking URI
        model: The model name
        mean_input_tokens: Mean input tokens for this test
        mean_output_tokens: Mean output tokens for this test  
        num_concurrent_requests: Number of concurrent requests
        summary: The results summary dictionary
        user_metadata: Additional user metadata
        tensor_parallel_size: The number of tensor parallel processes model is running on.
        gpu_name: The name of the GPU to use for this load test - ex "mi300x" or "h100".
    """
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Create experiment name based on model
        experiment_name = f"llmperf-{model}"
        try:
            mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            pass
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        run_name = f"{model}_input_{mean_input_tokens}_output_{mean_output_tokens}_concurrent_{num_concurrent_requests}_tp_{tensor_parallel_size}_gpu_{gpu_name}"
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("model", model)
            mlflow.log_param("mean_input_tokens", mean_input_tokens)
            mlflow.log_param("stddev_input_tokens", summary.get("stddev_input_tokens", "N/A"))
            mlflow.log_param("mean_output_tokens", mean_output_tokens)
            mlflow.log_param("stddev_output_tokens", summary.get("stddev_output_tokens", "N/A"))
            mlflow.log_param("num_concurrent_requests", num_concurrent_requests)
            mlflow.log_param("tensor_parallel_size", tensor_parallel_size)
            mlflow.log_param("additional_sampling_params", str(summary.get("additional_sampling_params", {})))
            
            # Log user metadata as parameters
            for key, value in user_metadata.items():
                mlflow.log_param(f"user_{key}", value)
            
            # Log results metrics
            results = summary.get("results", {})
            
            # Log main performance metrics
            if common_metrics.OUTPUT_THROUGHPUT in results:
                mlflow.log_metric("output_throughput_tokens_per_s", results[common_metrics.OUTPUT_THROUGHPUT])
            
            if common_metrics.NUM_COMPLETED_REQUESTS in results:
                mlflow.log_metric("completed_requests", results[common_metrics.NUM_COMPLETED_REQUESTS])
                
            if common_metrics.COMPLETED_REQUESTS_PER_MIN in results:
                mlflow.log_metric("completed_requests_per_min", results[common_metrics.COMPLETED_REQUESTS_PER_MIN])
                
            if common_metrics.ERROR_RATE in results:
                mlflow.log_metric("error_rate", results[common_metrics.ERROR_RATE])
                
            if common_metrics.NUM_ERRORS in results:
                mlflow.log_metric("num_errors", results[common_metrics.NUM_ERRORS])
            
            # Log detailed metrics for key performance indicators
            # Group related metrics together using hierarchical naming
            metrics_to_log = [
                common_metrics.INTER_TOKEN_LAT,
                common_metrics.TTFT,
                common_metrics.E2E_LAT,
                common_metrics.REQ_OUTPUT_THROUGHPUT,
                common_metrics.NUM_INPUT_TOKENS,
                common_metrics.NUM_OUTPUT_TOKENS
            ]
            
            for metric in metrics_to_log:
                if metric in results:
                    metric_data = results[metric]
                    if isinstance(metric_data, dict):
                        # Create a cleaner metric name for grouping
                        clean_metric_name = metric.replace("_", " ").title()
                        
                        # Log quantiles as a grouped metric using percentile values as steps
                        # Note: MLflow will show these as steps on x-axis, representing percentile values
                        if "quantiles" in metric_data:
                            quantile_mapping = {"p25": 25, "p50": 50, "p75": 75, "p90": 90, "p95": 95, "p99": 99}
                            for quantile, step_value in quantile_mapping.items():
                                if quantile in metric_data["quantiles"]:
                                    mlflow.log_metric(f"{clean_metric_name} Percentiles", 
                                                    metric_data["quantiles"][quantile], step=step_value)
                        
                        # Log statistical measures as individual metrics (cleaner than artificial grouping)
                        stats_to_log = ["min", "mean", "max", "stddev"]
                        for stat in stats_to_log:
                            if stat in metric_data:
                                mlflow.log_metric(f"{clean_metric_name} {stat.title()}", metric_data[stat])
            
            print(f"Results logged to MLflow experiment '{experiment_name}' with run name '{run_name}'")
            
    except Exception as e:
        print(f"Failed to log to MLflow: {e}")


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    num_warmup_requests: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
    mlflow_uri: str = "",
    tensor_parallel_size: int = 0,
    gpu_name: str = "gpu"
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
        tensor_parallel_size: The number of tensor parallel processes model is running on.
        gpu_name: The name of the GPU to use for this load test - ex "mi300x" or "h100".
    """
    if mean_input_tokens < 40:
        print(
            "the minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now"
        )

    summary, individual_responses = get_token_throughput_latencies(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        mean_input_tokens=mean_input_tokens,
        stddev_input_tokens=stddev_input_tokens,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=stddev_output_tokens,
        num_concurrent_requests=num_concurrent_requests,
        num_warmup_requests=num_warmup_requests,
        additional_sampling_params=json.loads(additional_sampling_params),
    )

    if results_dir:
        filename = f"{model}_{mean_input_tokens}_{mean_output_tokens}_{num_concurrent_requests}"
        filename = re.sub(r"[^\w\d-]+", "-", filename)
        filename = re.sub(r"-{2,}", "-", filename)
        summary_filename = f"{filename}_summary"
        individual_responses_filename = f"{filename}_individual_responses"

        # Update to metadata.
        summary.update(user_metadata)

        results = LLMPerfResults(name=summary_filename, metadata=summary)
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        try:
            with open(results_dir / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
                json.dump(individual_responses, f, indent=4)
        except Exception as e:
            print(individual_responses)
            raise e
    
    # Log to MLflow if URI is provided
    if mlflow_uri:
        log_to_mlflow(
            mlflow_uri=mlflow_uri,
            model=model,
            mean_input_tokens=mean_input_tokens,
            mean_output_tokens=mean_output_tokens,
            num_concurrent_requests=num_concurrent_requests,
            summary=summary,
            user_metadata=user_metadata,
            tensor_parallel_size=tensor_parallel_size,
            gpu_name=gpu_name,
        )


args = argparse.ArgumentParser(
    description="Run a token throughput and latency benchmark."
)

args.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)
args.add_argument(
    "--mean-input-tokens",
    type=int,
    nargs='+',
    default=[550],
    help=(
        "The mean number of tokens to send in the prompt for the request. "
        "Can specify multiple values to run a test matrix. (default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-input-tokens",
    type=int,
    default=150,
    help=(
        "The standard deviation of number of tokens to send in the prompt for the request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--mean-output-tokens",
    type=int,
    nargs='+',
    default=[150],
    help=(
        "The mean number of tokens to generate from each llm request. This is the max_tokens param "
        "for the completions API. Note that this is not always the number of tokens returned. "
        "Can specify multiple values to run a test matrix. (default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-output-tokens",
    type=int,
    default=80,
    help=(
        "The stdandard deviation on the number of tokens to generate per llm request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--num-concurrent-requests",
    type=int,
    nargs='+',
    default=[10],
    help=("The number of concurrent requests to send. Can specify multiple values to run a test matrix. (default: %(default)s)"),
)
args.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
args.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=10,
    help=(
        "The base number of requests to complete before finishing the test. "
        "This will be multiplied by the concurrency level for each test "
        "(e.g., base=100, concurrency=5 -> 500 total requests). Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
args.add_argument(
    "--num-warmup-requests",
    type=int,
    default=0,
    help=(
        "The base number of warmup requests to send before starting the benchmark. "
        "This will be multiplied by the concurrency level for each test "
        "(e.g., base=10, concurrency=5 -> 50 warmup requests). (default: %(default)s)"
    ),
)
args.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
args.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
args.add_argument(
    "--llm-api",
    type=str,
    default="openai",
    help=(
        f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--metadata",
    type=str,
    default="",
    help=(
        "A comma separated list of metadata to include in the results, e.g. "
        "name=foo,bar=1. These will be added to the metadata field of the results. "
    ),
)
args.add_argument(
    "--mlflow-uri",
    type=str,
    default="",
    help=(
        "MLflow tracking URI to log results to (e.g., http://localhost:5000). "
        "If not provided, results will not be logged to MLflow."
    ),
)

args.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=0,
    help="The number of tensor parallel processes to use. (default: %(default)s)",
)

args.add_argument(
    "--gpu-name",
    type=str,
    default="gpu",
    help="The name of the GPU to use for this load test. (default: %(default)s)",
)

if __name__ == "__main__":
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})
    args = args.parse_args()

    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    # Validate that input and output token lists have the same length for pairing
    if len(args.mean_input_tokens) != len(args.mean_output_tokens):
        raise ValueError(
            f"Input tokens list length ({len(args.mean_input_tokens)}) must match "
            f"output tokens list length ({len(args.mean_output_tokens)}) for pairing. "
            f"Got input: {args.mean_input_tokens}, output: {args.mean_output_tokens}"
        )
    
    # Generate test combinations: each input/output pair runs at each concurrency level
    token_pairs = list(zip(args.mean_input_tokens, args.mean_output_tokens))
    test_combinations = []
    for mean_input, mean_output in token_pairs:
        for concurrency in args.num_concurrent_requests:
            max_requests = args.max_num_completed_requests * concurrency
            warmup_requests = args.num_warmup_requests * concurrency
            test_combinations.append((mean_input, mean_output, concurrency, max_requests, warmup_requests))
    
    print(f"Running {len(test_combinations)} test combinations:")
    for i, (mean_input, mean_output, concurrency, max_requests, warmup_requests) in enumerate(test_combinations):
        print(f"  {i+1}/{len(test_combinations)}: input_tokens={mean_input}, output_tokens={mean_output}, concurrency={concurrency}, max_requests={max_requests}, warmup_requests={warmup_requests}")
    print()
    
    for i, (mean_input, mean_output, concurrency, max_requests, warmup_requests) in enumerate(test_combinations):
        print(f"\n{'='*80}")
        print(f"Running test {i+1}/{len(test_combinations)}: input_tokens={mean_input}, output_tokens={mean_output}, concurrency={concurrency}, max_requests={max_requests}, warmup_requests={warmup_requests}")
        print(f"{'='*80}")
        
        run_token_benchmark(
            llm_api=args.llm_api,
            model=args.model,
            test_timeout_s=args.timeout,
            max_num_completed_requests=max_requests,
            mean_input_tokens=mean_input,
            stddev_input_tokens=args.stddev_input_tokens,
            mean_output_tokens=mean_output,
            stddev_output_tokens=args.stddev_output_tokens,
            num_concurrent_requests=concurrency,
            num_warmup_requests=warmup_requests,
            additional_sampling_params=args.additional_sampling_params,
            results_dir=args.results_dir,
            user_metadata=user_metadata,
            mlflow_uri=args.mlflow_uri,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_name=args.gpu_name,
        )
