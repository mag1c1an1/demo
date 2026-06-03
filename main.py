from datasets import load_dataset


# 先加载全部数据
def main():
    print("Hello from multi!")
    dataset = load_dataset("nlphuji/flickr30k")
    print(dataset)


if __name__ == "__main__":
    main()
