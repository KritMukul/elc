from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct", local_files_only=True
)

# A tiny fake netlist with clear, obvious whitespace to test with
test_spice = "* Testbench for foo\nVDD VDD 0 DC 1.8\nVSS VSS 0 DC 0\n"

messages = [
    {"role": "user", "content": "test input"},
    {"role": "assistant", "content": test_spice}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

print("=== ORIGINAL SPICE TEXT (repr, to show whitespace) ===")
print(repr(test_spice))
print()
print("=== FULL TEMPLATED TEXT (repr) ===")
print(repr(text))
print()
print("=== FULL TEMPLATED TEXT (rendered) ===")
print(text)