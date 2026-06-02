#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CriticAgent: Script Quality Evaluation

Evaluates generated shooting scripts based on:
1. Format Compliance (0-5)
2. Shot Division Rationality (0-5)
3. Content Completeness (0-5)
4. Narrative Coherence (0-5)

Input: 
- infer_script_result.jsonl: Generated scripts (key: "response")
- test.json: Original dialogues (key: "input")

Dependencies: requests
"""

import argparse
import base64
import datetime
import hashlib
import hmac
import json
import logging
import os
import re
import signal
import uuid
from typing import Dict, List, Optional, Any, Union

try:
    import requests
except ImportError:
    requests = None


LOGGER = logging.getLogger(__name__)


# ==========================================
# DistillInterface for Gemini 2.5 Pro
# ==========================================

class DistillInterface:
    """Interface for calling Gemini 2.5 Pro via internal platform."""
    
    def __init__(self, user_id: str, key: str, base_url: str = "/api/v1/data_eval"):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        self.HOST = "http://trpc-gpt-eval.production.polaris:8080"
        self.API_VERSION = "v2.03"
        self.base_url = self.HOST + base_url
        
        self.MODELS = {
            "gemini-2.5-pro": "api_google_gemini-2.5-pro",
            "gemini-2.5-flash": "api_google_gemini-2.5-flash",
        }
        
        self.user = user_id
        self.apikey = key
        self.timeout = 300
    
    def handle_timeout(self, signum, frame):
        raise TimeoutError("Request timeout exceeded")
    
    def get_simple_auth(self, source: str, app_id: str, app_key: str):
        date_time = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        auth = (
            'hmac id="' + app_id + 
            '", algorithm="hmac-sha1", headers="date source", signature="'
        )
        sign_str = "date: " + date_time + "\n" + "source: " + source
        sign = hmac.new(app_key.encode(), sign_str.encode(), hashlib.sha1).digest()
        sign = base64.b64encode(sign).decode()
        sign = auth + sign + '"'
        return sign, date_time
    
    def _get_header(self):
        source = "neo"
        sign, date_time = self.get_simple_auth(source, self.user, self.apikey)
        headers = {
            "Apiversion": self.API_VERSION,
            "Authorization": sign,
            "Date": date_time,
            "Source": source,
            "Content-Type": "application/json"
        }
        return headers
    
    def request(
        self, 
        model: str, 
        content_payload: Union[str, List[Dict]], 
        temperature: float = 0.6
    ) -> Optional[str]:
        """Send request to Gemini API."""
        base_url = self.base_url
        
        if model not in self.MODELS:
            LOGGER.error(f"Model {model} not found in configuration.")
            return None
        
        model_id = self.MODELS[model]
        
        # Convert string to proper format (API expects content to be an array)
        if isinstance(content_payload, str):
            # For text-only, wrap in array format: [{"type": "text", "value": "..."}]
            messages = [{
                "role": "user", 
                "content": [{"type": "text", "value": content_payload}]
            }]
            LOGGER.debug(f"Content payload is string, length={len(content_payload)}, converted to array format")
        else:
            # Already in correct format: list of content parts
            messages = [{"role": "user", "content": content_payload}]
            LOGGER.debug(f"Content payload is list/object with {len(content_payload)} parts")
        
        data = {
            "request_id": str(uuid.uuid4()),
            "model_marker": model_id,
            "system": "",
            "messages": messages,
            "params": {
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": 8192,
                    "topP": 0.95,
                }
            },
            "timeout": self.timeout * 3,
            "temperature": temperature,
        }
        
        headers = dict(self._get_header())
        
        # Log request info for debugging
        LOGGER.debug(f"Request URL: {base_url}")
        LOGGER.debug(f"Request model_marker: {model_id}")
        if isinstance(content_payload, str):
            payload_preview = content_payload[:300] + "..." if len(content_payload) > 300 else content_payload
            LOGGER.debug(f"Request text payload preview: {payload_preview}")
        
        try:
            signal.alarm(self.timeout)
            LOGGER.info(f"Sending request to {model}...")
            
            response = requests.post(
                url=base_url,
                headers=headers,
                json=data,
                timeout=self.timeout * 2,
            )
            signal.alarm(0)
            
            LOGGER.debug(f"Response status code: {response.status_code}")
            
        except requests.exceptions.Timeout as e:
            signal.alarm(0)
            LOGGER.error(f"Request timeout after {self.timeout * 2}s: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            signal.alarm(0)
            LOGGER.error(f"Connection error: {e}")
            return None
        except Exception as e:
            signal.alarm(0)
            LOGGER.error(f"Request failed with exception: {type(e).__name__}: {e}")
            return None
        
        if response.status_code != 200:
            LOGGER.error(
                f"Non-200 response | status={response.status_code} | "
                f"reason={response.reason}"
            )
            try:
                error_body = response.json()
                LOGGER.error(f"Error response body: {json.dumps(error_body, indent=2, ensure_ascii=False)}")
            except:
                LOGGER.error(f"Error response text: {response.text[:1000]}")
            
            # Log request details for debugging
            if isinstance(content_payload, str):
                LOGGER.debug(f"Request payload length: {len(content_payload)} chars")
                LOGGER.debug(f"Request payload preview: {content_payload[:500]}...")
            
            return None
        
        try:
            ret = response.json()
            LOGGER.debug(f"Response JSON keys: {list(ret.keys())}")
        except ValueError as e:
            LOGGER.error(f"JSON decode error: {e}")
            return None
        
        try:
            reasoning = ret.get("answer")
            
            if reasoning is None:
                choices = ret.get("choices", [])
                if choices:
                    reasoning = choices[0].get("message", {}).get("content")
                else:
                    LOGGER.error(f"No 'answer' field found. Available keys: {list(ret.keys())}")
                    reasoning = None
            elif isinstance(reasoning, str):
                pass
            elif isinstance(reasoning, list) and reasoning:
                first_elem = reasoning[0]
                if isinstance(first_elem, dict):
                    reasoning = first_elem.get("value") or first_elem.get("content") or first_elem.get("text")
                elif isinstance(first_elem, str):
                    reasoning = first_elem
            elif isinstance(reasoning, dict):
                reasoning = reasoning.get("value") or reasoning.get("content") or reasoning.get("text")
            
            if not reasoning:
                LOGGER.error(f"Reasoning is None after extraction")
                return None
                
        except Exception as e:
            LOGGER.error(f"Parsing response error: {type(e).__name__}: {e}")
            reasoning = None
        
        return reasoning


SCRIPT_EVALUATION_PROMPT = """You are a professional film director and script supervisor. Your task is to evaluate the quality of a generated shooting script based on the provided coarse-grained dialogue and context.

**Input Data:**
- **Source Dialogue:** {source_dialogue}
- **Generated Script:** {generated_script}

**IMPORTANT CONTEXT:**
The generated script is in a structured screenplay format with sections marked by 【】brackets (Chinese-style format). This is a VALID and PROFESSIONAL format commonly used in Chinese film production. Scripts that contain all required sections (DIALOGUE with detailed shot breakdowns, CHARACTER PROFILES, SCENE DESCRIPTION, and BLOCKING) should be considered high-quality and well-formatted.

**Evaluation Criteria:**
Please score the generated script on a scale of 0.0 to 5.0 for each of the following dimensions.  
**IMPORTANT: Scores can be continuous decimal values (e.g., 3.5, 4.2, 2.7). The integer scores (0, 1, 2, 3, 4, 5) are reference points to help you calibrate, but you can use any value between them for more precise evaluation.**

**Scoring Philosophy:**
- Well-structured scripts with all required sections and detailed descriptions should score 4.5+ on Format Compliance
- Professional-quality scripts with comprehensive details should score 4.0+ overall
- Score 5.0 is for truly exceptional, flawless work
- Be fair and recognize quality work appropriately

For each dimension, use the following general guideline (integer scores are reference points):

- Score 0.0: Completely unusable or fails the requirement.
- Score 1.0: Very poor quality; severe issues in most parts.
- Score 2.0: Clearly below acceptable quality; many issues.
- Score 3.0: Acceptable but with noticeable issues. This is the baseline for functional scripts.
- Score 4.0: Good quality with only minor issues. Above average but not exceptional.
- Score 5.0: **Exceptional quality; near-perfect execution with virtually no flaws. Reserve this for truly outstanding work that demonstrates professional-level craftsmanship.**

**You can use decimal values to provide more nuanced scores. For example:**
- 3.2 = Slightly above acceptable (3.0) but still has noticeable issues
- 3.7 = Approaching good quality (4.0) but not quite there
- 4.3 = Good quality (4.0) with some minor positive aspects
- 4.8 = Very close to exceptional (5.0) but has minor imperfections

Then, judge each dimension more concretely as follows:

1. **Format Compliance (0.0–5.0):**  
Does the output follow the required structured screenplay format? The script should contain all essential sections with proper formatting:
- **【DIALOGUE】**: Shot-by-shot descriptions with time codes [Xs-Xs], camera movements (e.g., Arc Shot, Handheld Camera Effect), shot types (e.g., Medium Shot, Close-up), and detailed plot descriptions
- **【CHARACTER PROFILES】**: Physical descriptions (height, build, facial features, hair, clothing) and personality/demeanor for each character
- **【SCENE DESCRIPTION】**: Setting details and atmospheric description
- **【BLOCKING】**: Character positioning and movement choreography

**You can use decimal scores (e.g., 3.5, 4.2) for more precise evaluation.**
   - 0.0: Completely unstructured or missing all required sections.
   - 1.0: Missing 3+ required sections or severely malformed structure.
   - 2.0: Missing 2 required sections or multiple sections with poor formatting.
   - 3.0: All major sections present but with noticeable formatting inconsistencies or incomplete details in 1-2 sections.
   - 4.0: All required sections present with proper structure. Only minor issues like occasional missing time codes or slight formatting variations.
   - 4.5-4.9: Excellent format with all sections well-structured. Very minor imperfections such as one or two missing details.
   - 5.0: **Perfect format with all sections impeccably structured, complete time codes, detailed descriptions, and consistent formatting throughout.**

2. **Shot Division Rationality (0.0–5.0):**  
Is the script segmented into shots reasonably? Do the shot breaks align with narrative beats and emotional shifts without being too fragmented or too long?
**You can use decimal scores (e.g., 3.5, 4.2) for more precise evaluation.**
   - 0.0: No meaningful shot division; essentially a single block or random splitting.
   - 1.0: Very unreasonable segmentation; shots break the flow and ignore story structure.
   - 2.0: Many inappropriate shot boundaries; frequent over- or under-segmentation.
   - 3.0: Basic correspondence to narrative beats, but with several awkward or suboptimal shot splits.
   - 4.0-4.4: Well-aligned with emotional and narrative shifts, with only minor segmentation issues.
   - 4.5-4.9: Excellent shot division that enhances storytelling, with shots clearly aligned to dialogue turns and emotional beats.
   - 5.0: **Perfectly crafted shot division with every break serving a clear narrative purpose.**

3. **Content Completeness (0.0–5.0):**  
Does the script provide rich, actionable details for filming? Does it supplement necessary visual information that was missing in the source dialogue?
**You can use decimal scores (e.g., 3.5, 4.2) for more precise evaluation.**
   - 0.0: Almost no additional visual or staging information beyond the raw dialogue.
   - 1.0: Very sparse detail; crucial information for filming is largely missing.
   - 2.0: Some useful details, but important aspects (scene, actions, camera) remain underspecified.
   - 3.0: Contains enough information to stage the scene, but important visual details are still missing.
   - 4.0-4.4: Rich and specific details covering visual elements, character actions, and camera work.
   - 4.5-4.9: Comprehensive details including character profiles, scene atmosphere, blocking, and shot-by-shot descriptions with time codes.
   - 5.0: **Exceptionally complete with every visual element thoroughly specified and production-ready.**

4. **Narrative Coherence (0.0–5.0):**  
Is the sequence of shots logically connected? Does the visual storytelling flow smoothly and match the context of the dialogue?
**You can use decimal scores (e.g., 3.5, 4.2) for more precise evaluation.**
   - 0.0: Completely incoherent sequence; shots appear random and unrelated to the dialogue.
   - 1.0: Very confusing progression; frequent contradictions or abrupt jumps.
   - 2.0: A rough story is visible, but there are many logical gaps, contradictions, or unnatural transitions.
   - 3.0: Overall story is understandable, but several transitions or details break the narrative flow.
   - 4.0-4.4: Coherent narrative with smooth transitions and logical shot progression.
   - 4.5-4.9: Excellent narrative flow with clear blocking, consistent character positions, and seamless transitions between shots.
   - 5.0: **Perfectly crafted visual narrative with flawless coherence and masterful storytelling.**

**Scoring Guidelines:**
- Be fair, objective, and recognize quality work appropriately.
- Scores can be decimal values (e.g., 3.5, 4.2, 4.7) for more precise evaluation.
- Integer scores (0, 1, 2, 3, 4, 5) are reference points to help you calibrate.
- Score 3.0 is basic/acceptable quality with noticeable issues.
- Score 4.0-4.4 is good quality - well-executed with only minor issues.
- Score 4.5-4.9 is excellent quality - professional-level work with very minor imperfections.
- Score 5.0 is exceptional/perfect - flawless execution in every aspect.
- For Format Compliance: Scripts with all required sections properly formatted should score 4.5+.
- Use decimal values to provide nuanced scores that accurately reflect the quality level.

**Output Format:**  
Return the result in the following JSON format ONLY (no additional text):
Scores should be numbers (can be integers like 3 or decimals like 3.5, 4.2, etc.):
{{
  "Format Compliance": [Score as number, e.g., 3.5 or 4],
  "Shot Division Rationality": [Score as number, e.g., 4.2 or 3],
  "Content Completeness": [Score as number, e.g., 3.7 or 4],
  "Narrative Coherence": [Score as number, e.g., 4.1 or 3],
  "Reasoning": {{
    "Format Compliance": "[Brief explanation]",
    "Shot Division Rationality": "[Brief explanation]",
    "Content Completeness": "[Brief explanation]",
    "Narrative Coherence": "[Brief explanation]"
  }}
}}
"""


class ScriptEvaluator:
    """Evaluates shooting script quality using Gemini 2.5 Pro."""
    
    def __init__(
        self, 
        user_id: str,
        api_key: str,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.3,
    ):
        """Initialize script evaluator.
        
        Args:
            user_id: User ID for Gemini authentication.
            api_key: API key for Gemini authentication.
            model: Gemini model to use for evaluation.
            temperature: Sampling temperature for generation.
        """
        if requests is None:
            raise RuntimeError("requests library not found. Please install it: pip install requests")
        
        self.client = DistillInterface(user_id, api_key)
        self.model = model
        self.temperature = temperature
        
        LOGGER.info(f"Initialized ScriptEvaluator with model: {model}")
    
    def evaluate(
        self,
        source_dialogue: str,
        generated_script: str,
    ) -> Dict[str, Any]:
        """Evaluate a generated script.
        
        Args:
            source_dialogue: Original coarse-grained dialogue text.
            generated_script: Generated shooting script (JSON or text).
            
        Returns:
            Dictionary with evaluation scores and reasoning.
        """
        prompt = SCRIPT_EVALUATION_PROMPT.format(
            source_dialogue=source_dialogue,
            generated_script=generated_script,
        )
        
        LOGGER.info("Sending evaluation request to Gemini API...")
        LOGGER.debug(f"Prompt length: {len(prompt)} chars")
        LOGGER.debug(f"Source dialogue length: {len(source_dialogue)} chars")
        LOGGER.debug(f"Generated script length: {len(generated_script)} chars")
        
        # Check if content is too long (rough estimate: ~4 chars per token, limit ~32k tokens)
        estimated_tokens = len(prompt) / 4
        if estimated_tokens > 30000:
            LOGGER.warning(f"Content may be too long: ~{estimated_tokens:.0f} tokens (limit: ~32000)")
        
        try:
            response_text = self.client.request(
                model=self.model,
                content_payload=prompt,
                temperature=self.temperature
            )
            
            if not response_text:
                LOGGER.error("No response received from API")
                return {
                    "error": "No response received from API"
                }
            
            LOGGER.debug(f"Raw API response: {response_text[:500]}...")
            
            # Try to parse JSON from response
            try:
                result = json.loads(response_text)
                LOGGER.info("Evaluation completed successfully")
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from text
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, response_text, re.DOTALL)
                
                if matches:
                    matches.sort(key=len, reverse=True)
                    for match in matches:
                        try:
                            result = json.loads(match)
                            LOGGER.info("Extracted JSON from embedded text")
                            return result
                        except:
                            continue
                
                # Fallback: try brace matching
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    try:
                        result = json.loads(json_str)
                        LOGGER.info("Extracted JSON using brace matching")
                        return result
                    except:
                        pass
                
                LOGGER.error(f"Could not parse response as JSON: {response_text[:200]}")
                return {
                    "raw_response": response_text[:1000],
                    "error": "Failed to parse response as JSON"
                }
            
        except Exception as e:
            LOGGER.error(f"Evaluation failed: {e}")
            import traceback
            LOGGER.debug(traceback.format_exc())
            return {
                "error": f"Evaluation failed: {str(e)}"
            }
    
    def evaluate_from_files(
        self,
        scripts_jsonl: str,
        dialogues_json: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """Evaluate scripts from paired files.
        
        Args:
            scripts_jsonl: Path to JSONL file with generated scripts (key: "response").
            dialogues_json: Path to JSON file with original dialogues (key: "input").
            output_path: Path to save results.
            
        Returns:
            Summary statistics with average scores.
        """
        LOGGER.info(f"Reading scripts file: {scripts_jsonl}")
        LOGGER.info(f"Reading dialogues file: {dialogues_json}")
        
        # Read generated scripts from JSONL
        scripts = []
        with open(scripts_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    scripts.append(data.get("response", ""))
                except json.JSONDecodeError as e:
                    LOGGER.error(f"Failed to parse line in scripts file: {e}")
                    scripts.append("")
        
        # Read original dialogues from JSON
        with open(dialogues_json, 'r', encoding='utf-8') as f:
            dialogues_data = json.load(f)
        
        dialogues = []
        if isinstance(dialogues_data, list):
            for item in dialogues_data:
                if isinstance(item, dict):
                    dialogues.append(item.get("input", ""))
                else:
                    dialogues.append("")
        else:
            LOGGER.error("Dialogues JSON is not a list")
            return {}
        
        LOGGER.info(f"Found {len(scripts)} scripts and {len(dialogues)} dialogues")
        
        # Check if counts match
        if len(scripts) != len(dialogues):
            LOGGER.warning(
                f"Mismatch in counts: {len(scripts)} scripts vs {len(dialogues)} dialogues. "
                f"Will evaluate min({len(scripts)}, {len(dialogues)}) entries."
            )
        
        results = []
        total_count = min(len(scripts), len(dialogues))
        success_count = 0
        
        # Evaluate each pair
        for idx in range(total_count):
            dialogue = dialogues[idx]
            script = scripts[idx]
            
            if not dialogue or not script:
                LOGGER.warning(f"Entry {idx+1}: Missing dialogue or script, skipping")
                continue
            
            LOGGER.info(f"Evaluating entry {idx+1}/{total_count}...")
            
            try:
                # Evaluate
                result = self.evaluate(dialogue, script)
                result['entry_number'] = idx + 1
                result['dialogue_preview'] = dialogue[:200] + "..." if len(dialogue) > 200 else dialogue
                result['script_preview'] = script[:200] + "..." if len(script) > 200 else script
                results.append(result)
                success_count += 1
                
                # Log evaluation results for each entry
                if "error" not in result:
                    dimensions = [
                        "Format Compliance",
                        "Shot Division Rationality",
                        "Content Completeness",
                        "Narrative Coherence"
                    ]
                    
                    # Build score summary
                    score_summary = []
                    for dimension in dimensions:
                        if dimension in result:
                            score = result[dimension]
                            if isinstance(score, (int, float)):
                                score_summary.append(f"{dimension}: {score:.2f}/5.0")
                            else:
                                score_summary.append(f"{dimension}: {score}")
                        else:
                            score_summary.append(f"{dimension}: Missing")
                    
                    # Log formatted results
                    LOGGER.info(f"[Entry {idx+1}/{total_count}] Evaluation Results:")
                    for score_line in score_summary:
                        LOGGER.info(f"  ✓ {score_line}")
                    
                    # Log reasoning preview if available
                    if "Reasoning" in result and isinstance(result["Reasoning"], dict):
                        LOGGER.debug(f"  Reasoning: Available for all dimensions")
                else:
                    LOGGER.warning(f"[Entry {idx+1}/{total_count}] Evaluation failed: {result.get('error', 'Unknown error')}")
                
                # Save progress every 10 entries
                if success_count % 10 == 0:
                    LOGGER.info(f"Progress: {success_count}/{total_count} completed")
                    # Save intermediate results
                    try:
                        temp_output = output_path + ".temp"
                        with open(temp_output, 'w', encoding='utf-8') as f:
                            json.dump({"partial_results": results}, f, indent=2, ensure_ascii=False)
                        LOGGER.debug(f"Saved intermediate results to {temp_output}")
                    except:
                        pass
                
            except Exception as e:
                LOGGER.error(f"Entry {idx+1}: Evaluation failed - {e}")
                import traceback
                LOGGER.debug(traceback.format_exc())
                continue
        
        # Calculate statistics
        LOGGER.info(f"Evaluation complete: {success_count}/{total_count} successful")
        
        avg_scores = self._calculate_average_scores(results)
        
        # Prepare output
        output_data = {
            "summary": {
                "total_entries": total_count,
                "successful_evaluations": success_count,
                "failed_evaluations": total_count - success_count,
                "average_scores": avg_scores,
            },
            "detailed_results": results,
        }
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        LOGGER.info(f"Results saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("SCRIPT EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total Entries: {total_count}")
        print(f"Successful: {success_count}")
        print(f"Failed: {total_count - success_count}")
        print("\nAverage Scores:")
        for dimension, score in avg_scores.items():
            print(f"  {dimension}: {score:.2f}/5.0")
        print("=" * 70 + "\n")
        
        return output_data
    
    def _calculate_average_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average scores across all evaluations."""
        dimensions = [
            "Format Compliance",
            "Shot Division Rationality",
            "Content Completeness",
            "Narrative Coherence"
        ]
        
        avg_scores = {}
        for dimension in dimensions:
            scores = [
                r[dimension] for r in results 
                if dimension in r and isinstance(r[dimension], (int, float))
            ]
            if scores:
                avg_scores[dimension] = sum(scores) / len(scores)
            else:
                avg_scores[dimension] = 0.0
        
        return avg_scores


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="CriticAgent: Evaluate shooting script quality using Gemini"
    )
    
    parser.add_argument(
        "--scripts_jsonl",
        type=str,
        required=True,
        help="Path to JSONL file with generated scripts (e.g., infer_script_result.jsonl, key: 'response')"
    )
    parser.add_argument(
        "--dialogues_json",
        type=str,
        required=True,
        help="Path to JSON file with original dialogues (e.g., test.json, key: 'input')"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save evaluation results (JSON format)"
    )
    parser.add_argument(
        "--user_id",
        type=str,
        required=True,
        help="User ID for Gemini authentication"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for Gemini authentication"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help="Gemini model to use (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser


def main() -> None:
    """Main entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s][%(levelname)s] %(message)s"
    )
    
    evaluator = ScriptEvaluator(
        user_id=args.user_id,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
    )
    
    # Evaluate from paired files
    evaluator.evaluate_from_files(
        scripts_jsonl=args.scripts_jsonl,
        dialogues_json=args.dialogues_json,
        output_path=args.output_json,
    )


if __name__ == "__main__":
    main()
