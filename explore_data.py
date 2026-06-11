from datasets import load_dataset


def main():
    print("CNN daily mail dataset")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    print(dataset)
    print(dataset["train"]["article"][0])
    print("\n" + "=" * 60)
    print(dataset["train"]["highlights"][0])


if __name__ == "__main__":
    main()
