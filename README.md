# PROTOON (Professional Token-Oriented Object Notation)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Optimized For: LLMs](https://img.shields.io/badge/Optimized%20For-LLMs-blueviolet)](https://openai.com/)
[![Token Efficiency: High](https://img.shields.io/badge/Token%20Efficiency-Extreme-green)]()

**PROTOON** is a strict, zero-overhead data serialization format designed specifically for Large Language Models (LLMs) and Byte-Pair Encoding (BPE) tokenizers. 

It is the **"Professional"** successor to the standard TOON format, stripping away all human-centric syntactical sugar (brackets, commas, quotes) to achieve maximum token density and lowest inference latency.

---

## 🚀 Why PROTOON?

JSON, YAML, and even standard TOON waste tokens on delimiters and redundant keys. PROTOON treats the **Context Window as a scarce resource**.

| Feature | JSON | Standard TOON | **PROTOON** |
| :--- | :--- | :--- | :--- |
| **Delimiters** | `"{},[]` (Heavy) | `[]{}` (Moderate) | **None (Whitespace/Pipe)** |
| **Separators** | `,` (Sticky*) | `,` | **`|` (Clean Split)** |
| **Booleans** | `true` (1 token) | `true` | **`+` (1 token)** |
| **Nulls** | `null` (1 token) | `null` | **`~` (1 token)** |
| **Arrays** | Repeated Keys | `items[3]` (Pre-calc required) | **`$` (Streaming Friendly)** |

*> **Sticky Commas:** In BPE tokenizers, `123,` is often a different token than `123`. PROTOON's pipe `|` is universally treated as a distinct separator, preventing integer vocabulary bloat.*

---

## 📋 Specification (v1.0)

### 1. Zero-Bracket Architecture
PROTOON forbids `{}` and `[]`. Structure is defined **strictly by indentation** (2 spaces).

### 2. The Schema Header (`$`)
For lists of objects, define the keys once using the `$` operator.
*   **Syntax:** `$key1|key2|key3`
*   **Reasoning:** Allows the LLM to stream the header immediately without calculating the list length beforehand.

### 3. Atomic Primitives
Use single characters for high-frequency constants.
*   **True:** `+`
*   **False:** `-`
*   **Null:** `~`
*   **Empty String:** `_`

### 4. The Pipe Separator (`|`)
Columns in schema arrays are separated by pipes.
*   **Escaping:** If a string contains `|` or `\n`, wrap it in backticks: `` `Text with | pipe` ``.

---

## 🔍 Examples

### 1. High-Frequency Financial Data
*JSON would repeat "ticker", "price", "vol" for every row. PROTOON does not.*

```text
market:NASDAQ
status:open
stream:
  $ticker|price|change|vol_m
  NVDA|495.50|+2.5|45.2
  TSLA|240.00|-1.2|33.0
  AAPL|190.50|~|12.1
```

### 2. Complex Nested Configuration
*Demonstrates deep nesting without braces.*

```text
app_config:
  version:2.0.1
  features:
    dark_mode:+
    beta_user:-
  logging:
    level:debug
    outputs:
      $type|path|retention
      file|/var/log/app.log|7d
      s3|s3://bucket/logs|30d
      console|~|~
```

### 3. Mixed Text (Chat Logs)
*Using backticks for strings containing special characters.*

```text
session:chat_001
messages:
  $role|content
  user|Explain quantum physics.
  assistant|`Quantum physics is the study of matter and energy at the most fundamental level.`
  user|Thanks.
```

---

## 🤖 Implementation Guide

### System Prompt Injection
Add this to your LLM's system instructions to force PROTOON output.

```text
RESPONSE FORMAT: PROTOON
RULES:
1. USE indentation (2 spaces) for hierarchy.
2. NO braces {}, brackets [], or quotes "".
3. FOR LISTS: Start with a header row "$key1|key2", then value rows "val1|val2".
4. PRIMITIVES: "+"=true, "-"=false, "~"=null.
5. ESCAPING: Wrap strings containing "|" or newlines in backticks (`).
```

### Python Parser (`protoon.py`)
A production-ready deserializer.

```python
import re

def parse_protoon(text):
    lines = [l.rstrip() for l in text.split('\n') if l.strip()]
    root = {}
    # Stack: (current_object, indent_level)
    stack = [(root, -1)] 
    
    # State to track if we are currently filling a list based on a schema
    current_list = None 
    current_schema = None
    
    # Regex to handle pipe splitting while respecting backticks
    # Matches: `quoted content` OR non-pipe content
    row_pattern = re.compile(r'`([^`]+)`|([^|]+)')

    def _cast(val):
        val = val.strip()
        if val == '+': return True
        if val == '-': return False
        if val == '~': return None
        if val == '_': return ""
        if val.replace('.','',1).isdigit():
            return float(val) if '.' in val else int(val)
        return val

    for line in lines:
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        
        # 1. Dedent Logic
        while stack and indent <= stack[-1][1]:
            stack.pop()
            # If we dedent, we leave the current schema/list context
            current_schema = None
            current_list = None

        parent = stack[-1][0]

        # 2. Schema Header ($key|key)
        if content.startswith('$'):
            keys = content[1:].split('|')
            current_schema = keys
            
            # The key for this list is the *last key added* to the parent object
            # We must convert that key's value from {} (placeholder) to []
            last_key = list(parent.keys())[-1]
            parent[last_key] = []
            current_list = parent[last_key]
            
            # We push the list to stack (conceptually at current indent) 
            # so dedenting works later
            stack.append((current_list, indent)) 
            continue

        # 3. Data Row (val|val)
        if current_schema and current_list is not None and '|' in content:
            # Extract values using regex to respect backticks
            matches = row_pattern.findall(content)
            # m[0] is backticked group, m[1] is normal group
            raw_vals = [m[0] if m[0] else m[1] for m in matches]
            
            row_obj = {}
            for k, v in zip(current_schema, raw_vals):
                row_obj[k] = _cast(v)
            current_list.append(row_obj)
            continue

        # 4. Key:Value Pair
        if ':' in content:
            # Reset list context if we encounter a standard key
            current_schema = None
            current_list = None
            
            key, val = content.split(':', 1)
            key = key.strip()
            val = val.strip()
            
            if val == '':
                # Start of a nested object
                new_obj = {}
                parent[key] = new_obj
                stack.append((new_obj, indent))
            else:
                parent[key] = _cast(val)

    return root
```

---

## ⚠️ Limitations

1.  **Stream Hallucination:** If the model hallucinates a pipe `|` inside a plain string without backticks, the parser column count will mismatch. *Mitigation: Use strict system prompts.*
2.  **Human Readability:** PROTOON is dense. Humans must look up the schema header (`$`) to know what the 5th column represents in a data row.
3.  **Root Arrays:** PROTOON v1.0 expects a Root Object (Dictionary). To represent a simple list at the root, wrap it in a `data:` key.

---

## License

MIT License. Free for commercial and personal use.
