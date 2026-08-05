from transformers import pipeline


def main():
    print("=" * 60)
    print("        GPT-2 TEXT GENERATION")
    print("=" * 60)

    print("\nLoading GPT-2 model...")

    generator = pipeline(
        "text-generation",
        model="gpt2"
    )

    print("Model loaded successfully!")

    while True:
        print("\n" + "-" * 60)

        prompt = input(
            "Enter your prompt (or type 'exit' to quit): "
        )

        if prompt.lower() == "exit":
            print("\nThank you for using the GPT-2 Text Generator!")
            break

        if not prompt.strip():
            print("Please enter a valid prompt.")
            continue

        print("\nGenerating text...\n")

        result = generator(
            prompt,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

        generated_text = result[0]["generated_text"]

        print("=" * 60)
        print("GENERATED TEXT")
        print("=" * 60)
        print(generated_text)


if __name__ == "__main__":
    main()
