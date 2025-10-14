import base64
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import torch
from openai import OpenAI
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)
from vllm import LLM, SamplingParams


@dataclass
class RawInput:
    """Dataclass to store raw input for a function."""

    image_input: Optional[Image.Image]
    text_input: Optional[str]


@dataclass
class IOExamples:
    """Dataclass to store input-output examples for a function."""

    description: str
    inputs: list[RawInput]
    outputs: list[list[Any]]


def img2base64(img):
    buffer = BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")

    # if width or height < 28, resize it keeping aspect ratio
    if img.width < 28 or img.height < 28:
        # make smallest dimension 28
        new_width = 28
        new_height = 28
        if img.width < img.height:
            new_height = int((28 / img.width) * img.height)
        else:
            new_width = int((28 / img.height) * img.width)
        img = img.resize((new_width, new_height))

    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def base642img(base64_str):
    imgdata = base64.b64decode(base64_str)
    return Image.open(BytesIO(imgdata))


class OurLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        if "Llama-3.2" in model_name:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                token=os.environ.get("HF_TOKEN"),
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name, token=os.environ.get("HF_TOKEN")
            )
        elif "QVQ" in model_name:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
                token=os.environ.get("HF_TOKEN"),
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name, token=os.environ.get("HF_TOKEN")
            )
        elif "Qwen" in model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                token=os.environ.get("HF_TOKEN"),
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name, token=os.environ.get("HF_TOKEN")
            )
        elif "OpenGVLab/InternVL2_5-78B-MPO" in model_name:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
                token=os.environ.get("HF_TOKEN"),
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name, token=os.environ.get("HF_TOKEN"), trust_remote_code=True
            )

    def chat(self, prompt, sampling_params, use_tqdm):
        # parse prompt content
        imgs = []
        processed_prompt = []
        for j in range(len(prompt)):
            prompt_content = []
            for i in range(len(prompt[j]["content"])):
                if prompt[j]["content"][i]["type"] == "text":
                    prompt_content.append(prompt[j]["content"][i])
                elif prompt[j]["content"][i]["type"] == "image_url":
                    prompt_content.append({"type": "image"})
                    img_base64 = prompt[j]["content"][i]["image_url"]["url"].split(",")[1]
                    imgs.append(base642img(img_base64))
            processed_prompt.append({"role": prompt[j]["role"], "content": prompt_content})

        # prompt = [{"role": "user", "content": prompt_content}]
        print("prompt:", processed_prompt)

        input_text = self.processor.apply_chat_template(
            processed_prompt, add_generation_prompt=True
        )
        inputs = self.processor(
            imgs if len(imgs) > 0 else None,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda:0")

        print(self.processor.decode(inputs["input_ids"][0]))
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=2500, temperature=0.0, do_sample=False, top_p=1.0
            )
        elapsed = time.time() - start

        print(f"Tokens per second: {(len(outputs[0][len(inputs['input_ids'][0]):])) / elapsed}")

        output_text = self.processor.decode(outputs[0][len(inputs["input_ids"][0]) :][:-1])
        print("output:", output_text)

        class Outputs:
            def __init__(self, outputs):
                self.outputs = outputs

        class Text:
            def __init__(self, text):
                self.text = text

        return [Outputs([Text(output_text)])]


class APIModel:
    def __init__(self, model_name):
        self.model_name = model_name
        if "gemini" in model_name:
            self.client = OpenAI(
                api_key=os.environ.get("GEMINI_API_KEY", ""),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "")
            )

    def chat(self, prompt, sampling_params, use_tqdm):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=0.0,
            max_tokens=2500,
            top_p=1.0,
        )

        class Outputs:
            def __init__(self, outputs):
                self.outputs = outputs

        class Text:
            def __init__(self, text):
                self.text = text

        # print number of tokens
        print("Prompt tokens:", response.usage.prompt_tokens)
        print("Response tokens:", response.usage.completion_tokens)

        return [Outputs([Text(response.choices[0].message.content)])]


class LLMNet:
    def __init__(
        self,
        model: LLM,
        input_desc: str,
        output_desc: str,
        examples: Optional[IOExamples] = None,
        few_shot=True,
        image_before_prompt=False,
    ) -> str:
        self.model = model
        self.input_desc = input_desc
        self.output_desc = output_desc
        self.examples = examples
        self.few_shot = few_shot
        self.image_before_prompt = image_before_prompt

    def forward(self, input: RawInput) -> str:
        prompt = []

        # Adding any the examples to the prompt
        prompt_content = []
        if self.examples is not None:
            for i, (ex_input, ex_output) in enumerate(
                zip(self.examples.inputs, self.examples.outputs)
            ):
                symbol_str = ", ".join([repr(o) for o in ex_output])

                if i == 0:
                    prompt_content.append(
                        {
                            "type": "text",
                            "text": f"After examining the input, determine {self.output_desc}. Here are some examples:",
                        }
                    )

                if not self.image_before_prompt:
                    if self.few_shot:
                        prompt_content.append(
                            {
                                "type": "text",
                                "text": f"\nThe following input is {self.input_desc}. Output just {self.output_desc} after 'FINAL ANSWER:'.",
                            }
                        )
                    else:
                        prompt_content.append(
                            {
                                "type": "text",
                                "text": f"\nThe following is an example of {symbol_str}:",
                            }
                        )
                else:
                    prompt_content.append({"type": "text", "text": f"\nExample {i + 1}:"})

                if ex_input.text_input is not None and ex_input.image_input is None:
                    prompt_content.append(
                        {
                            "type": "text",
                            "text": f"{ex_input.text_input}",
                        }
                    )
                elif ex_input.text_input is None and ex_input.image_input is not None:
                    prompt_content.extend(
                        [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img2base64(ex_input.image_input)}",
                                    "detail": "high",
                                },
                            },
                        ]
                    )
                else:
                    prompt_content.extend(
                        [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img2base64(ex_input.image_input)}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": f"{ex_input.text_input}",
                            },
                        ]
                    )

                if self.image_before_prompt:
                    if self.few_shot:
                        prompt_content.append(
                            {
                                "type": "text",
                                "text": f"The input is {self.input_desc}. Output just {self.output_desc} after 'FINAL ANSWER:'.",
                            }
                        )
                    else:
                        prompt_content.append(
                            {"type": "text", "text": f"This is an example of {symbol_str}."}
                        )

                if self.few_shot:
                    prompt.append({"role": "user", "content": prompt_content})
                    prompt.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": f"FINAL ANSWER: {symbol_str}"}],
                        }
                    )
                    prompt_content = []

        if not self.image_before_prompt:
            prompt_content.append(
                {
                    "type": "text",
                    "text": f"\nThe following input is {self.input_desc}. Examine it and then output just {self.output_desc} after 'FINAL ANSWER:'. If unsure of the answer, try to choose the best option.",
                }
            )
        else:
            prompt_content.append({"type": "text", "text": "\n"})

        # Adding the input to the prompt (text or image)
        if input.text_input is not None and input.image_input is None:
            prompt_content.append({"type": "text", "text": input.text_input})
        elif input.text_input is None and input.image_input is not None:
            prompt_content.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}",
                            "detail": "high",
                        },
                    },
                ]
            )
        else:
            prompt_content.extend(
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img2base64(input.image_input)}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": input.text_input},
                ]
            )

        if self.image_before_prompt:
            prompt_content.append(
                {
                    "type": "text",
                    "text": f"The input is {self.input_desc}. Examine it and then output just {self.output_desc} after 'FINAL ANSWER:'. If unsure of the answer, try to choose the best option.",
                }
            )

        prompt.append({"role": "user", "content": prompt_content})

        sampling_params = SamplingParams(temperature=0.0, max_tokens=5000, top_p=1.0)
        output = (
            self.model.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0]
            .outputs[0]
            .text
        )
        print("out:", output)

        extra_args = [re.DOTALL]
        try:
            if "\\[ \\boxed{" in output:
                res = re.findall(r"\[ \\boxed{(.*)}", output, *extra_args)[-1]
                pred = res.strip()
            elif "**FINAL ANSWER:**" in output:
                res = re.findall(r"\*\*FINAL ANS.*:\*\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*FINAL ANSWER:*" in output:
                res = re.findall(r"\*FINAL ANS.*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "**Final Answer:**" in output:
                res = re.findall(r"\*\*Final Ans.*:\*\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*Final Answer:*" in output:
                res = re.findall(r"\*Final Ans.*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "Final answer:" in output:
                res = re.findall(r"Final ans.*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*Final answer:*" in output:
                res = re.findall(r"\*Final ans.*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "**Final answer:**" in output:
                res = re.findall(r"\*\*Final ans.*:\*\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "**Answer:**" in output:
                res = re.findall(r"\*\*Answer:\*\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*Answer:*" in output:
                res = re.findall(r"\*Answer:\*(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "**Answer**:" in output:
                res = re.findall(r"\*\*Answer\*\*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "*Answer*:" in output:
                res = re.findall(r"\*Answer\*:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            elif "FINAL ANSWER:" in output:
                # print("here", re.findall(r"FINAL ANS.*:(.*)(?:<|$)", output, *extra_args))
                res = re.findall(r"FINAL ANSWER:(.*)(?:<|$)", output, *extra_args)[-1]
                pred = res.strip()
            else:
                pred = output.strip()

            if "```json" in pred:
                pred = re.findall(r"```json(.*?)```", pred, *extra_args)[-1]
            if "```" in pred:
                pred = re.sub(r"```", "", pred).strip()
            if "<|eot_id|>" in pred:
                pred = re.sub(r"<\|eot_id\|>", "", pred).strip()
            if "\\text{" in pred:
                res = re.findall(r"\\text{(.*?)}", pred, *extra_args)[-1]
                pred = res.strip()
            return pred
        except Exception:
            return "None"
