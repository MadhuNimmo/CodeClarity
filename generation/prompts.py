PROMPTS = {
    "prompt0": (
        "Please generate a code summary for the given code snippet in {language_name}. "
        "Include the following information: the function's name, a brief description of its purpose, "
        "a list of its arguments and their types, and a summary of its key logic — all in plain, "
        "natural text without formatting or section headings.\n\nThe code is:\n\n{code}\n\n"
    ),
    "prompt1": (
        "Please generate a code summary for the given code in {language_name}. "
        "Your summary must include all of the following components: "
        "1. The function or method name (if applicable), "
        "2. A brief description of its overall purpose, "
        "3. A list of all arguments and their expected types (if applicable), "
        "4. A plain-language explanation of the main logic or steps involved. "
        "Write in natural, complete sentences without using code formatting, bullet points, "
        "or section headings. Be precise and clear. The code is:\n{code}"
    ),
    "prompt2": (
        "What does the following code do? Please describe its purpose and behavior in {language_name}. "
        "The code is:\n{code}"
    ),
    "prompt3": (
        "Generate Code Summary for the given code snippet in {language_name}. The code is:\n{code}"
    ),
}
TARGET_NATURAL_LANGUAGES = ["English", "Chinese", "French", "Spanish", "Portuguese", "Arabic", "Hindi"]
