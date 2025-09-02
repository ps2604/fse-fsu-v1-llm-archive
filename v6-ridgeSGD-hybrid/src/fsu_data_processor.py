# SPDX-License-Identifier: Apache-2.0
# FSU DATA PROCESSOR – V7 - DEFINITIVE PRODUCTION REWRITE
# Incorporates robust parsing for all datasets, including CommonsenseQA format variations.

import json
import gzip
import os
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union, Iterator
import numpy as np
from multiprocessing import Pool, cpu_count

try:
    from adjoint_core_optimized import FSEField
except ImportError:
    FSEField = Any

logger = logging.getLogger(__name__)

@dataclass
class ConversationData:
    """Standardized container for all processed data."""
    char_sequence: np.ndarray
    target_sequence: np.ndarray
    conversation_id: str
    data_type: str
    metadata: Dict[str, Any]
    sequence_length: int
    original_text: str

class FSUDataProcessor:
    """
    Turns raw corpora into character-level training pairs using robust,
    dataset-specific parsers and multi-core processing for high speed.
    """

    def __init__(self, sequence_length: int = 4096, vocab_size: int = 65536):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.PAD, self.S, self.E, self.UNK = range(4)
        logger.info(f"✅ FSUDataProcessor Initialized: SeqLen={sequence_length}, Vocab={vocab_size}")

    def get_parser_for_filename(self, filename: str):
        """
        [DEFINITIVE UTILITY] Returns the correct processing function based on the
        contents of the filename.
        """
        filename_lower = filename.lower()
        if "oasst" in filename_lower: return self._proc_openassistant
        if "sharegpt" in filename_lower: return self._proc_sharegpt
        if "wizardlm" in filename_lower: return self._proc_wizardlm
        if "gsm8k" in filename_lower: return self._proc_gsm8k
        if "competition_math" in filename_lower: return self._proc_comp_math
        if "commonsense_qa" in filename_lower: return self._proc_csqa
        if "hellaswag" in filename_lower: return self._proc_hellaswag
        if "cot" in filename_lower: return self._proc_cot
        return None
    def _text_to_pairs(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        [CORRECTED] Converts text to correctly aligned input/target pairs
        of the specified sequence_length.
        """
        try:
            if not isinstance(text, str): text = ""
            # The target length for the full sequence is one longer than the model's
            # processing length to allow for the input/target shift.
            target_full_len = self.sequence_length + 1

            bytes_ = text.encode("utf-8", errors="replace")
            # Start with a special start-of-sequence token
            codes = [self.S] + [min(b + 4, self.vocab_size - 1) for b in bytes_]

            # Truncate or pad the sequence
            if len(codes) >= target_full_len:
                # Truncate to the exact length needed
                final_codes = codes[:target_full_len]
                # Ensure the last character is an end-of-sequence token for the target
                final_codes[-1] = self.E
            else:
                # Pad with PAD tokens
                final_codes = codes + [self.E] + ([self.PAD] * (target_full_len - len(codes) - 1))

            # Convert to a numpy array
            arr = np.array(final_codes, dtype=np.int32)

            if arr.shape != (target_full_len,):
                raise ValueError(f"Array shape error during pair creation: got {arr.shape}, expected ({target_full_len},)")

            # Create the correctly aligned input and target sequences
            # Input sequence is the first `sequence_length` characters
            input_arr = arr[:-1]
            # Target sequence is the last `sequence_length` characters (shifted by one)
            target_arr = arr[1:]

            # Final validation
            if input_arr.shape != (self.sequence_length,) or target_arr.shape != (self.sequence_length,):
                raise ValueError(f"Final pair shape error: input {input_arr.shape}, target {target_arr.shape}")

            return input_arr, target_arr

        except Exception as e:
            logger.error(f"FATAL in _text_to_pairs: {e}", exc_info=True)
            # Return validly shaped zero-arrays on failure
            arr = np.full((self.sequence_length,), self.PAD, dtype=np.int32)
            return arr, arr

    def _read_file_to_records(self, file_path: str) -> List[Dict]:
        """Reads a .json or .jsonl file (.gz supported) into a list of records."""
        logger.info(f"  Reading file '{os.path.basename(file_path)}'...")
        open_fn = gzip.open if file_path.endswith('.gz') else open
        records = []
        try:
            with open_fn(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if not content.strip(): return []
                try: 
                    data = json.loads(content)
                    if isinstance(data, list): records.extend(data)
                    else: records.append(data)
                except json.JSONDecodeError:
                    records = [json.loads(line) for line in content.strip().split('\n') if line]
            logger.info(f"  Successfully read {len(records)} records from '{os.path.basename(file_path)}'.")
            return records
        except Exception as e:
            logger.error(f"❌ Could not read or parse file {file_path}: {e}")
            return []

    def discover_and_process_all_datasets(self, data_paths: Dict[str, str]) -> Dict[str, List[ConversationData]]:
        """Main entry point to discover and process all datasets in parallel."""
        logger.info("🚀 Starting dataset discovery and processing...")
        dispatch = {
            "oasst_ready_trees.jsonl.gz": self._proc_openassistant,
            "sharegpt_vicuna_unfiltered.json": self._proc_sharegpt,
            "wizardlm_evol_instruct_v2.json": self._proc_wizardlm,
            "gsm8k_train.json": self._proc_gsm8k, "gsm8k_test.json": self._proc_gsm8k,
            "competition_math_train.json": self._proc_comp_math, "competition_math_test.json": self._proc_comp_math,
            "commonsense_qa_train.json": self._proc_csqa, "commonsense_qa_validation.json": self._proc_csqa,
            "hellaswag_train.json": self._proc_hellaswag, "hellaswag_validation.json": self._proc_hellaswag,
            "cot_collection_kaist.json": self._proc_cot, "cot_data.json": self._proc_cot,
        }
        tasks = [(dispatch[bf], fp, bf) for k, fp in data_paths.items() if (bf := os.path.basename(fp)) in dispatch and os.path.exists(fp)]
        
        all_data: List[ConversationData] = []
        num_cores = max(1, cpu_count() - 2)
        logger.info(f"⚙️ Using {num_cores} cores for parallel processing.")
        
        with Pool(processes=num_cores) as pool:
            results = pool.map(self._process_file_wrapper, tasks)
        
        for result_list in results:
            if result_list: all_data.extend(result_list)
        
        if not all_data:
            logger.error("❌ No data was processed from any source. Returning empty dataset.")
            return {"train": [], "val": []}
            
        logger.info(f"🔀 Shuffling {len(all_data)} total processed items...")
        np.random.shuffle(all_data)
        k = int(len(all_data) * 0.10)
        val_data, train_data = all_data[:k], all_data[k:]
        logger.info(f"📊 Dataset split complete: {len(train_data)} train items, {len(val_data)} val items.")
        return {"train": train_data, "val": val_data}

    @staticmethod
    def _process_file_wrapper(args_tuple):
        """
        Static helper function to be used by multiprocessing.Pool. It now
        correctly unpacks all necessary arguments.
        """
        file_path, args = args_tuple
        # Each process gets its own FSUDataProcessor instance
        processor = FSUDataProcessor(sequence_length=args.sequence_length, vocab_size=args.vocab_size)
        
        base_filename = os.path.basename(file_path)
        parser_func = processor.get_parser_for_filename(base_filename)
        if not parser_func: 
            return []

        records = processor._read_file_to_records(file_path)
        if not records: 
            return []
        
        return list(parser_func(records))
    # =========================================================================
    # DATASET-SPECIFIC PARSERS
    # =========================================================================

    def _proc_openassistant(self, records: List[Dict]) -> Iterator[ConversationData]:
        trees, children_map = {}, {}
        for msg in records:
            if not (tree_id := msg.get("message_tree_id")): continue
            if tree_id not in trees: trees[tree_id], children_map[tree_id] = {}, {}
            trees[tree_id][msg.get("message_id")] = msg
            if parent_id := msg.get("parent_id"):
                if parent_id not in children_map[tree_id]: children_map[tree_id][parent_id] = []
                children_map[tree_id][parent_id].append(msg)
        for tree_id, msgs in trees.items():
            if not (root := next((m for m in msgs.values() if m.get("parent_id") is None), None)): continue
            thread, cur = [root], root
            while True:
                children = children_map[tree_id].get(cur.get("message_id"), [])
                if not children: break
                children.sort(key=lambda x: x.get("rank", 0) or 0, reverse=True)
                thread.append(children[0]); cur = children[0]
            text = "".join(f"{'Human' if m.get('role') == 'prompter' else 'Assistant'}: {m.get('text', '')}\n\n" for m in thread)
            if text.strip():
                yield ConversationData(*self._text_to_pairs(text), tree_id, "conversation", {"source": "openassistant"}, len(text), text[:250])

    def _proc_sharegpt(self, records: List[Dict]) -> Iterator[ConversationData]:
        for i, conv in enumerate(records):
            text = "".join(f"{'Human' if t.get('from','').lower() in ('human','user') else 'Assistant'}: {t.get('value','')}\n\n" for t in conv.get("conversations", []))
            if text.strip():
                yield ConversationData(*self._text_to_pairs(text), f"sharegpt_{i}", "conversation", {"source": "sharegpt"}, len(text), text[:250])

    def _proc_wizardlm(self, records: List[Dict]) -> Iterator[ConversationData]:
        for i, item in enumerate(records):
            text = "".join(f"{'Human' if t.get('from','').lower() == 'human' else 'Assistant'}: {t.get('value','')}\n\n" for t in item.get("conversations", []))
            if text.strip():
                yield ConversationData(*self._text_to_pairs(text), f"wizardlm_{i}", "instruction", {"source": "wizardlm"}, len(text), text[:250])

    def _proc_gsm8k(self, records: List[Dict]) -> Iterator[ConversationData]:
        for i, item in enumerate(records):
            if (q := item.get("question", "")) and (a := item.get("answer", "")):
                text = f"Human: {q}\n\nAssistant: Let's think step by step. {a}\n\n"
                yield ConversationData(*self._text_to_pairs(text), f"gsm8k_{i}", "math_reasoning", {"source": "gsm8k"}, len(text), text[:250])

    def _proc_comp_math(self, records: List[Dict]) -> Iterator[ConversationData]:
        for i, item in enumerate(records):
            if (q := item.get("problem", "")) and (a := item.get("solution", "")):
                text = f"Human: {q}\n\nAssistant: Here is the solution. {a}\n\n"
                yield ConversationData(*self._text_to_pairs(text), f"comp_math_{i}", "advanced_math", {"source": "competition_math"}, len(text), text[:250])
    
    def _proc_csqa(self, records: List[Dict]) -> Iterator[ConversationData]:
        """[DEFINITIVE FIX] Handles multiple known formats for CommonsenseQA records."""
        for i, item in enumerate(records):
            try:
                question_text = item.get("question", "")
                answer_key = item.get("answerKey", "").upper()
                choices_obj = item.get("choices", {})

                if not (question_text and answer_key and choices_obj): continue

                choices = {}
                if isinstance(choices_obj, dict) and "label" in choices_obj and "text" in choices_obj:
                    # Format: {"label": ["A", "B"], "text": ["text1", "text2"]}
                    labels = choices_obj.get("label", [])
                    texts = choices_obj.get("text", [])
                    if len(labels) == len(texts):
                        choices = {label: text for label, text in zip(labels, texts)}
                elif isinstance(choices_obj, list):
                    # Format: [{"label": "A", "text": "text1"}, ...]
                    choices = {c.get("label"): c.get("text") for c in choices_obj if isinstance(c, dict)}

                if not choices or answer_key not in choices: continue
                
                options = "\n".join([f"{label}) {text}" for label, text in choices.items()])
                answer_text = choices[answer_key]
                text = f"Human: Answer the following question.\n{question_text}\n\nOptions:\n{options}\n\nAssistant: The correct answer is {answer_key}. {answer_text}\n\n"
                yield ConversationData(*self._text_to_pairs(text), item.get('id', f"csqa_{i}"), "logical_reasoning", {"source": "commonsense_qa"}, len(text), text[:250])
            except Exception as e:
                logger.warning(f"Skipping record in CSQA due to parsing error: {e}")
                continue

    def _proc_hellaswag(self, records: List[Dict]) -> Iterator[ConversationData]:
        """
        Handles all known HellaSwag formats:

        ① HF/Algolia  : {"ctx": "...", "endings": [...], "label": 2}
        ② AI2 JSONL   : {"ctx_a": "...", "ctx_b": "...", "ending_options": [...], "label": "1"}
        ③ Kaggle      : {"ctx": "...", "answers": [...], "label": "0"}
        """
        for i, item in enumerate(records):
            # -------- context -------------------------------------------------
            ctx = item.get("ctx")
            if not ctx:                       # older split stores a/b segments
                ctx = (item.get("ctx_a", "") + item.get("ctx_b", "")).strip()
            if not ctx:
                continue

            # -------- endings / options --------------------------------------
            endings = (
                item.get("endings")           # HF
                or item.get("ending_options") # AI2 JSONL
                or item.get("answers")        # Kaggle
                or []
            )
            if not isinstance(endings, list) or not endings:
                continue

            # -------- label → int --------------------------------------------
            label_raw = item.get("label", "")
            try:
                label = int(label_raw)
            except (ValueError, TypeError):
                # sometimes "label" is a single letter "A"…"D"
                if isinstance(label_raw, str) and label_raw.isdigit():
                    label = int(label_raw)
                else:
                    label = -1

            if not (0 <= label < len(endings)):
                # fall back to the first ending if label is missing / corrupt
                label = 0

            # -------- build conversation -------------------------------------
            answer = endings[label]
            text = (
                f"Human: Complete the following sentence:\n"
                f"{ctx}\n\n"
                f"Assistant: {answer}\n\n"
            )

            yield ConversationData(
                *self._text_to_pairs(text),
                f"hellaswag_{i}",
                "completion_reasoning",
                {"source": "hellaswag"},
                len(text),
                text[:250],
            )

    def _proc_cot(self, records: List[Dict]) -> Iterator[ConversationData]:
        """[ROBUSTNESS FIX] Handles multiple keys for CoT datasets."""
        for i, item in enumerate(records):
            q = item.get("source") or item.get("instruction") or item.get("input") or item.get("prompt")
            a = item.get("target") or item.get("output") or item.get("response") or item.get("completion")
            if q and a and isinstance(q, str) and isinstance(a, str):
                text = f"Human: {q}\n\nAssistant: {a}\n\n"
                yield ConversationData(*self._text_to_pairs(text), f"cot_{i}", "chain_of_thought", {"source": "cot_collection"}, len(text), text[:250])