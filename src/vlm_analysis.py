import os
import base64
import json
import csv
import time
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# API keys — set as environment variables before running
VLM_CONFIG = {
    'anthropic_api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
    'openai_api_key': os.environ.get('OPENAI_API_KEY', ''),
    'gemini_api_key': os.environ.get('GEMINI_API_KEY', ''),
    # Model names (configurable for research comparison)
    'claude_model': 'claude-sonnet-4-6',
    'openai_model': 'gpt-4o',
    'gemini_model': 'gemini-2.5-flash',
    # Local open-source model via Ollama
    'ollama_model': 'llava:7b-v1.5-q4_K_M',
    'ollama_url': 'http://localhost:11434',
}

REPORT_PROMPT = """You are analyzing thermal infrared drone footage captured from a UAV monitoring highway traffic for incident detection.

IMPORTANT context about thermal imagery:
- Bright/white spots represent heat-emitting objects (vehicles with running engines)
- Darker areas represent cooler surfaces (road pavement)
- The drone is flying along the highway, so there is background motion from the drone movement
- Vehicles appear as small bright rectangular blobs

You are provided with two types of images:
1. RAW THERMAL FRAMES: Direct thermal camera captures showing the highway scene from above
2. TRAJECTORY PLOTS: Vehicle trajectory visualizations where black lines on white background represent individual vehicle movement paths tracked over a 3-second sliding window using optical flow

Based on these images, generate a structured traffic incident report:

- **Incident Type**: What type of incident is observed? (e.g., rear-end collision, stalled vehicle, lane-blocking debris, multi-vehicle pileup, vehicle on shoulder, etc.)
- **Severity**: Low / Medium / High / Critical
- **Traffic Impact**: How is traffic flow affected? Describe any visible congestion, queue formation, or lane blockage patterns.
- **Vehicles Affected**: Estimated number of vehicles directly involved and indirectly affected (queued vehicles)
- **Location on Roadway**: Describe where on the roadway the incident appears (travel lanes, shoulder, median, ramp, etc.)
- **Evidence from Trajectories**: What do the trajectory patterns tell you? (e.g., converging lines = vehicles slowing/stopping, parallel lines = normal flow, erratic lines = lane changes around incident)
- **Recommended Response**: Suggested emergency response actions
- **Confidence Level**: Low / Medium / High — and briefly explain why

Be specific and base your analysis only on what you observe in the provided images."""


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_media_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.bmp': 'image/bmp'}
    return types.get(ext, 'image/png')


def analyze_with_claude(image_paths, prompt=REPORT_PROMPT):
    api_key = VLM_CONFIG['anthropic_api_key']
    if not api_key:
        return {'provider': 'Claude', 'model': VLM_CONFIG['claude_model'], 'error': 'ANTHROPIC_API_KEY not set', 'report': None, 'latency': 0}

    start = time.time()
    content = []
    for img_path in image_paths:
        content.append({
            'type': 'image',
            'source': {
                'type': 'base64',
                'media_type': get_media_type(img_path),
                'data': encode_image_to_base64(img_path),
            }
        })
    content.append({'type': 'text', 'text': prompt})

    model = VLM_CONFIG['claude_model']
    try:
        resp = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json',
            },
            json={
                'model': model,
                'max_tokens': 1500,
                'messages': [{'role': 'user', 'content': content}],
            },
            timeout=90,
        )
        latency = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            usage = result.get('usage', {})
            return {
                'provider': 'Claude',
                'model': model,
                'report': result['content'][0]['text'],
                'latency': round(latency, 2),
                'input_tokens': usage.get('input_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0),
                'error': None,
            }
        else:
            return {
                'provider': 'Claude',
                'model': model,
                'report': None,
                'latency': round(time.time() - start, 2),
                'error': f'HTTP {resp.status_code}: {resp.text[:200]}',
            }
    except Exception as e:
        return {
            'provider': 'Claude',
            'model': model,
            'report': None,
            'latency': round(time.time() - start, 2),
            'error': str(e),
        }


def analyze_with_gpt4o(image_paths, prompt=REPORT_PROMPT):
    api_key = VLM_CONFIG['openai_api_key']
    if not api_key:
        return {'provider': 'GPT-4o', 'model': VLM_CONFIG['openai_model'], 'error': 'OPENAI_API_KEY not set', 'report': None, 'latency': 0}

    start = time.time()
    content = []
    for img_path in image_paths:
        img_b64 = encode_image_to_base64(img_path)
        media_type = get_media_type(img_path)
        content.append({
            'type': 'image_url',
            'image_url': {'url': f'data:{media_type};base64,{img_b64}'},
        })
    content.append({'type': 'text', 'text': prompt})

    try:
        resp = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': VLM_CONFIG['openai_model'],
                'max_tokens': 1500,
                'messages': [{'role': 'user', 'content': content}],
            },
            timeout=90,
        )
        latency = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            usage = result.get('usage', {})
            return {
                'provider': 'GPT-4o',
                'model': VLM_CONFIG['openai_model'],
                'report': result['choices'][0]['message']['content'],
                'latency': round(latency, 2),
                'input_tokens': usage.get('prompt_tokens', 0),
                'output_tokens': usage.get('completion_tokens', 0),
                'error': None,
            }
        else:
            return {
                'provider': 'GPT-4o',
                'model': VLM_CONFIG['openai_model'],
                'report': None,
                'latency': round(time.time() - start, 2),
                'error': f'HTTP {resp.status_code}: {resp.text[:200]}',
            }
    except Exception as e:
        return {
            'provider': 'GPT-4o',
            'model': VLM_CONFIG['openai_model'],
            'report': None,
            'latency': round(time.time() - start, 2),
            'error': str(e),
        }


def analyze_with_gemini(image_paths, prompt=REPORT_PROMPT):
    api_key = VLM_CONFIG['gemini_api_key']
    if not api_key:
        return {'provider': 'Gemini', 'model': VLM_CONFIG['gemini_model'], 'error': 'GEMINI_API_KEY not set', 'report': None, 'latency': 0}

    start = time.time()
    parts = []
    for img_path in image_paths:
        parts.append({
            'inline_data': {
                'mime_type': get_media_type(img_path),
                'data': encode_image_to_base64(img_path),
            }
        })
    parts.append({'text': prompt})

    model_name = VLM_CONFIG['gemini_model']
    try:
        resp = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}',
            headers={'Content-Type': 'application/json'},
            json={
                'contents': [{'parts': parts}],
                'generationConfig': {'maxOutputTokens': 8192},
            },
            timeout=90,
        )
        latency = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            usage = result.get('usageMetadata', {})
            return {
                'provider': 'Gemini',
                'model': model_name,
                'report': result['candidates'][0]['content']['parts'][0]['text'],
                'latency': round(latency, 2),
                'input_tokens': usage.get('promptTokenCount', 0),
                'output_tokens': usage.get('candidatesTokenCount', 0),
                'error': None,
            }
        else:
            return {
                'provider': 'Gemini',
                'model': model_name,
                'report': None,
                'latency': round(time.time() - start, 2),
                'error': f'HTTP {resp.status_code}: {resp.text[:200]}',
            }
    except Exception as e:
        return {
            'provider': 'Gemini',
            'model': model_name,
            'report': None,
            'latency': round(time.time() - start, 2),
            'error': str(e),
        }


def analyze_with_llava(image_paths, prompt=REPORT_PROMPT):
    """Send images to LLaVA via Ollama local server."""
    ollama_url = VLM_CONFIG.get('ollama_url', 'http://localhost:11434')
    model = VLM_CONFIG.get('ollama_model', 'llava:7b-v1.5-q4_K_M')

    # Check if Ollama is running
    try:
        health = requests.get(f'{ollama_url}/api/tags', timeout=5)
        if health.status_code != 200:
            return {'provider': 'LLaVA', 'model': model, 'error': 'Ollama server not responding', 'report': None, 'latency': 0}
    except Exception:
        return {'provider': 'LLaVA', 'model': model, 'error': 'Ollama server not running. Start it with: ollama serve', 'report': None, 'latency': 0}

    start = time.time()

    # LLaVA 7B performs better with fewer images. Select the middle raw frame
    # and the middle trajectory image for the best single-pair representation.
    raw_frames = [p for p in image_paths if 'frame' in os.path.basename(p)]
    traj_images = [p for p in image_paths if 'trajectory' in os.path.basename(p)]
    selected = []
    if raw_frames:
        selected.append(raw_frames[len(raw_frames) // 2])
    if traj_images:
        selected.append(traj_images[len(traj_images) // 2])
    if not selected:
        selected = image_paths[:2]

    images_b64 = []
    for img_path in selected:
        images_b64.append(encode_image_to_base64(img_path))

    try:
        resp = requests.post(
            f'{ollama_url}/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'images': images_b64,
                'stream': False,
                'options': {
                    'num_predict': 1500,
                },
            },
            timeout=300,  # LLaVA on CPU can be slow
        )
        latency = time.time() - start

        if resp.status_code == 200:
            result = resp.json()
            return {
                'provider': 'LLaVA',
                'model': model,
                'report': result.get('response', ''),
                'latency': round(latency, 2),
                'input_tokens': result.get('prompt_eval_count', 0),
                'output_tokens': result.get('eval_count', 0),
                'error': None,
            }
        else:
            return {
                'provider': 'LLaVA',
                'model': model,
                'report': None,
                'latency': round(time.time() - start, 2),
                'error': f'HTTP {resp.status_code}: {resp.text[:200]}',
            }
    except Exception as e:
        return {
            'provider': 'LLaVA',
            'model': model,
            'report': None,
            'latency': round(time.time() - start, 2),
            'error': str(e),
        }


def analyze_incident(image_paths, providers=None):
    """Run VLM analysis across all providers in parallel. Returns list of result dicts."""
    if providers is None:
        providers = ['claude', 'gpt4o', 'gemini', 'llava']

    dispatch = {
        'claude': analyze_with_claude,
        'gpt4o': analyze_with_gpt4o,
        'gemini': analyze_with_gemini,
        'llava': analyze_with_llava,
    }

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for p in providers:
            if p in dispatch:
                futures[executor.submit(dispatch[p], image_paths)] = p
        for future in as_completed(futures):
            results.append(future.result())

    return results


def save_results(results, output_dir, video_name):
    """Save VLM results to JSON (detailed) and CSV (comparison table for paper)."""
    os.makedirs(output_dir, exist_ok=True)

    # Detailed JSON per video
    json_path = os.path.join(output_dir, f'{video_name}_vlm_reports.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Append to comparison CSV (one row per provider per video)
    csv_path = os.path.join(output_dir, 'vlm_comparison.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'timestamp', 'video_name', 'provider', 'model',
                'latency_s', 'input_tokens', 'output_tokens',
                'report_length_chars', 'error',
            ])
        for r in results:
            writer.writerow([
                datetime.now().isoformat(),
                video_name,
                r.get('provider', ''),
                r.get('model', ''),
                r.get('latency', 0),
                r.get('input_tokens', 0),
                r.get('output_tokens', 0),
                len(r.get('report', '') or ''),
                r.get('error', ''),
            ])

    return json_path
