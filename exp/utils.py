alphabet_to_int = {}

for i in range(1, 10):
    alphabet_to_int[str(i)] = i
# Add lowercase alphabets
for i in range(26):
    alphabet_to_int[chr(ord("a") + i)] = 9 + i + 1

# Add uppercase alphabets
for i in range(26):
    alphabet_to_int[chr(ord("A") + i)] = 9 + i + 27

alphabet_to_int[""] = 0

int_to_alphabet = {v: k for k, v in alphabet_to_int.items()}
length_window = 100


sensor_fieldnames = [
    "timestamp",
    "A0",
    "A2",
    "label",
]
