#!/usr/bin/env python3
"""
Enhanced Inference Demo for Llama-3 Function Calling Fine-tuning

This demo compares the base model vs fine-tuned model on function calling tasks,
demonstrating the improvement achieved through LoRA fine-tuning.

Features:
- Side-by-side base vs fine-tuned comparison
- Diverse business-relevant test cases
- JSON validity checking
- Success rate metrics
- Clean visual output for presentations
"""

import torch
import json
import re
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
BASE_MODEL_PATH = "/mnt/data/models/llama3-8b-instruct"
CHECKPOINT_DIR = "/mnt/data/checkpoints/llama3-function-calling-ray"

def get_checkpoint_path():
    """Auto-detect latest checkpoint or use explicit path."""
    import glob
    import re

    # Check if final model exists (training completed)
    final_path = f"{CHECKPOINT_DIR}/final"
    if os.path.exists(final_path):
        print(f"Using final model: {final_path}")
        return final_path

    # Find latest checkpoint-N directory
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}/checkpoint-*")
    if checkpoints:
        # Extract step numbers and sort
        checkpoint_steps = []
        for ckpt in checkpoints:
            match = re.search(r'checkpoint-(\d+)', ckpt)
            if match:
                checkpoint_steps.append((int(match.group(1)), ckpt))

        if checkpoint_steps:
            # Sort by step number and get the latest
            latest = sorted(checkpoint_steps, key=lambda x: x[0])[-1]
            print(f"Auto-detected latest checkpoint: {latest[1]} (step {latest[0]})")
            return latest[1]

    raise ValueError(f"No checkpoints found in {CHECKPOINT_DIR}")

LORA_ADAPTER_PATH = get_checkpoint_path()

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}{Colors.END}\n")

def print_subheader(text: str):
    """Print a subheader."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}{Colors.END}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

@dataclass
class TestCase:
    """Represents a function calling test case."""
    name: str
    category: str
    system_prompt: str
    user_query: str
    expected_function: str
    expected_args: List[str]  # Arguments that should be present

# Define comprehensive test cases
TEST_CASES = [
    # Weather & Location
    TestCase(
        name="Weather Query",
        category="API Calls",
        system_prompt="""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
        },
        "required": ["location"]
    }
}""",
        user_query="What's the weather like in San Francisco?",
        expected_function="get_weather",
        expected_args=["location"]
    ),

    # Calculator / Math
    TestCase(
        name="Mathematical Calculation",
        category="Computation",
        system_prompt="""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "calculate",
    "description": "Evaluate a mathematical expression and return the result",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The mathematical expression to evaluate"}
        },
        "required": ["expression"]
    }
}""",
        user_query="What is 15% of 850?",
        expected_function="calculate",
        expected_args=["expression"]
    ),

    # Database Query
    TestCase(
        name="Database Search",
        category="Data Operations",
        system_prompt="""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "search_database",
    "description": "Search the customer database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "table": {"type": "string", "description": "Table to search in"},
            "limit": {"type": "integer", "description": "Maximum results to return"}
        },
        "required": ["query", "table"]
    }
}""",
        user_query="Find all customers from New York in the users table",
        expected_function="search_database",
        expected_args=["query", "table"]
    ),

    # Email/Communication
    TestCase(
        name="Send Email",
        category="Communication",
        system_prompt="""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "send_email",
    "description": "Send an email to a recipient",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body content"}
        },
        "required": ["to", "subject", "body"]
    }
}""",
        user_query="Send an email to john@example.com with subject 'Meeting Tomorrow' saying we need to discuss the Q4 report",
        expected_function="send_email",
        expected_args=["to", "subject", "body"]
    ),

    # Calendar/Scheduling
    TestCase(
        name="Schedule Meeting",
        category="Scheduling",
        system_prompt="""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "create_calendar_event",
    "description": "Create a new calendar event",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title"},
            "start_time": {"type": "string", "description": "Start time in ISO format"},
            "duration_minutes": {"type": "integer", "description": "Duration in minutes"},
            "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee emails"}
        },
        "required": ["title", "start_time", "duration_minutes"]
    }
}""",
        user_query="Schedule a 30-minute team standup tomorrow at 9 AM",
        expected_function="create_calendar_event",
        expected_args=["title", "start_time", "duration_minutes"]
    ),

    # File Operations
    TestCase(
        name="File Search",
        category="File Operations",
        system_prompt="""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "search_files",
    "description": "Search for files in the system",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "File name pattern (supports wildcards)"},
            "directory": {"type": "string", "description": "Directory to search in"},
            "file_type": {"type": "string", "description": "Filter by file extension"}
        },
        "required": ["pattern"]
    }
}""",
        user_query="Find all PDF files in the documents folder",
        expected_function="search_files",
        expected_args=["pattern"]
    ),

    # Currency Conversion
    TestCase(
        name="Currency Conversion",
        category="Financial",
        system_prompt="""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "convert_currency",
    "description": "Convert an amount from one currency to another",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_currency": {"type": "string", "description": "Source currency code (e.g., USD)"},
            "to_currency": {"type": "string", "description": "Target currency code (e.g., EUR)"}
        },
        "required": ["amount", "from_currency", "to_currency"]
    }
}""",
        user_query="Convert 500 US dollars to Japanese yen",
        expected_function="convert_currency",
        expected_args=["amount", "from_currency", "to_currency"]
    ),

    # User Management
    TestCase(
        name="Create User Account",
        category="User Management",
        system_prompt="""You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "create_user",
    "description": "Create a new user account",
    "parameters": {
        "type": "object",
        "properties": {
            "username": {"type": "string", "description": "Username for the new account"},
            "email": {"type": "string", "description": "User's email address"},
            "role": {"type": "string", "enum": ["admin", "user", "guest"], "description": "User role"}
        },
        "required": ["username", "email"]
    }
}""",
        user_query="Create a new user account for Alice with email alice@company.com as an admin",
        expected_function="create_user",
        expected_args=["username", "email", "role"]
    ),
]


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from model response, handling various formats."""
    # Try to find JSON in response
    patterns = [
        r'\{[^{}]*"name"[^{}]*\}',  # Simple JSON with "name"
        r'\{[^{}]*"function"[^{}]*\}',  # JSON with "function"
        r'```json\s*(.*?)\s*```',  # Markdown code block
        r'```\s*(.*?)\s*```',  # Generic code block
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                # Clean up the match
                json_str = match.strip()
                if not json_str.startswith('{'):
                    json_str = '{' + json_str + '}'
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue

    # Try to parse the whole response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    return None


def evaluate_response(response: str, test_case: TestCase) -> Tuple[bool, str, Optional[Dict]]:
    """
    Evaluate if the response is a valid function call.
    Returns: (is_valid, evaluation_message, parsed_json)
    """
    parsed = extract_json_from_response(response)

    if parsed is None:
        return False, "No valid JSON found in response", None

    # Check if it has the expected function name
    func_name = parsed.get("name") or parsed.get("function")
    if not func_name:
        return False, "JSON missing function name", parsed

    if func_name != test_case.expected_function:
        return False, f"Wrong function: got '{func_name}', expected '{test_case.expected_function}'", parsed

    # Check for required arguments
    args = parsed.get("arguments") or parsed.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            return False, "Arguments not valid JSON", parsed

    missing_args = [arg for arg in test_case.expected_args if arg not in args]
    if missing_args:
        return False, f"Missing arguments: {missing_args}", parsed

    return True, "Valid function call with correct arguments", parsed


class ModelTester:
    """Handles model loading and inference."""

    def __init__(self, use_quantization: bool = True):
        self.tokenizer = None
        self.base_model = None
        self.finetuned_model = None
        self.use_quantization = use_quantization

    def load_tokenizer(self):
        """Load the tokenizer."""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        print_success("Tokenizer loaded")

    def load_base_model(self):
        """Load the base model (without LoRA)."""
        print("\nLoading base model (this may take a moment)...")

        if self.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.base_model.eval()
        print_success("Base model loaded")

    def load_finetuned_model(self):
        """Load the fine-tuned model with LoRA adapter."""
        print("\nLoading fine-tuned model with LoRA adapter...")

        if self.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        self.finetuned_model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
        self.finetuned_model.eval()
        print_success("Fine-tuned model loaded")

    def generate(self, model, system_prompt: str, user_query: str, max_tokens: int = 256) -> str:
        """Generate a response from the model."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()


def run_comparison_demo(tester: ModelTester, test_cases: List[TestCase], compare_base: bool = True):
    """Run the full comparison demo."""

    results = {
        "base": {"passed": 0, "failed": 0, "details": []},
        "finetuned": {"passed": 0, "failed": 0, "details": []}
    }

    for i, test in enumerate(test_cases, 1):
        print_subheader(f"Test {i}/{len(test_cases)}: {test.name} ({test.category})")

        print(f"\n{Colors.BOLD}User Query:{Colors.END}")
        print(f"  \"{test.user_query}\"")

        print(f"\n{Colors.BOLD}Expected:{Colors.END}")
        print(f"  Function: {test.expected_function}")
        print(f"  Required args: {test.expected_args}")

        # Test fine-tuned model
        print(f"\n{Colors.BOLD}{Colors.GREEN}Fine-tuned Model Response:{Colors.END}")
        ft_response = tester.generate(tester.finetuned_model, test.system_prompt, test.user_query)
        print(f"  {ft_response[:200]}{'...' if len(ft_response) > 200 else ''}")

        ft_valid, ft_msg, ft_parsed = evaluate_response(ft_response, test)
        if ft_valid:
            print_success(f"  {ft_msg}")
            results["finetuned"]["passed"] += 1
        else:
            print_error(f"  {ft_msg}")
            results["finetuned"]["failed"] += 1
        results["finetuned"]["details"].append({
            "test": test.name,
            "valid": ft_valid,
            "message": ft_msg
        })

        # Test base model (if requested)
        if compare_base:
            print(f"\n{Colors.BOLD}{Colors.BLUE}Base Model Response:{Colors.END}")
            base_response = tester.generate(tester.base_model, test.system_prompt, test.user_query)
            print(f"  {base_response[:200]}{'...' if len(base_response) > 200 else ''}")

            base_valid, base_msg, base_parsed = evaluate_response(base_response, test)
            if base_valid:
                print_success(f"  {base_msg}")
                results["base"]["passed"] += 1
            else:
                print_warning(f"  {base_msg}")
                results["base"]["failed"] += 1
            results["base"]["details"].append({
                "test": test.name,
                "valid": base_valid,
                "message": base_msg
            })

    return results


def print_summary(results: Dict, compare_base: bool):
    """Print the summary of results."""
    print_header("EVALUATION SUMMARY", "═")

    total = results["finetuned"]["passed"] + results["finetuned"]["failed"]
    ft_rate = (results["finetuned"]["passed"] / total * 100) if total > 0 else 0

    print(f"{Colors.BOLD}Fine-tuned Model:{Colors.END}")
    print(f"  ✓ Passed: {results['finetuned']['passed']}/{total}")
    print(f"  ✗ Failed: {results['finetuned']['failed']}/{total}")
    print(f"  {Colors.GREEN}Success Rate: {ft_rate:.1f}%{Colors.END}")

    if compare_base:
        base_rate = (results["base"]["passed"] / total * 100) if total > 0 else 0
        improvement = ft_rate - base_rate

        print(f"\n{Colors.BOLD}Base Model:{Colors.END}")
        print(f"  ✓ Passed: {results['base']['passed']}/{total}")
        print(f"  ✗ Failed: {results['base']['failed']}/{total}")
        print(f"  Success Rate: {base_rate:.1f}%")

        print(f"\n{Colors.BOLD}{Colors.CYAN}Improvement: +{improvement:.1f}%{Colors.END}")

    # Detailed breakdown
    print(f"\n{Colors.BOLD}Test Results by Category:{Colors.END}")
    categories = {}
    for detail in results["finetuned"]["details"]:
        test = next(t for t in TEST_CASES if t.name == detail["test"])
        if test.category not in categories:
            categories[test.category] = {"passed": 0, "total": 0}
        categories[test.category]["total"] += 1
        if detail["valid"]:
            categories[test.category]["passed"] += 1

    for cat, stats in categories.items():
        rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        status = "✓" if rate == 100 else "○" if rate >= 50 else "✗"
        print(f"  {status} {cat}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Llama-3 Function Calling Inference Demo")
    parser.add_argument("--no-base", action="store_true", help="Skip base model comparison")
    parser.add_argument("--quick", action="store_true", help="Run only 3 test cases")
    parser.add_argument("--no-quantization", action="store_true", help="Load models without 4-bit quantization")
    args = parser.parse_args()

    print_header("LLAMA-3 FUNCTION CALLING FINE-TUNING DEMO")

    print(f"{Colors.BOLD}Configuration:{Colors.END}")
    print(f"  Base Model: {BASE_MODEL_PATH}")
    print(f"  LoRA Adapter: {LORA_ADAPTER_PATH}")
    print(f"  Quantization: {'Disabled' if args.no_quantization else '4-bit (NF4)'}")
    print(f"  Compare Base: {'No' if args.no_base else 'Yes'}")

    # Initialize tester
    tester = ModelTester(use_quantization=not args.no_quantization)

    print_header("LOADING MODELS")

    tester.load_tokenizer()
    tester.load_finetuned_model()

    if not args.no_base:
        tester.load_base_model()

    print_header("RUNNING FUNCTION CALLING TESTS")

    # Select test cases
    test_cases = TEST_CASES[:3] if args.quick else TEST_CASES
    print(f"Running {len(test_cases)} test cases...\n")

    # Run tests
    results = run_comparison_demo(tester, test_cases, compare_base=not args.no_base)

    # Print summary
    print_summary(results, compare_base=not args.no_base)

    print_header("DEMO COMPLETE")

    # Return exit code based on results
    ft_rate = results["finetuned"]["passed"] / (results["finetuned"]["passed"] + results["finetuned"]["failed"])
    return 0 if ft_rate >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
