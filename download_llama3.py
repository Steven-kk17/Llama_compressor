from transformers import AutoTokenizer, AutoModelForCausalLM

def download_and_save_model(model_name="meta-llama/Llama-3.2-1B", save_directory="./saved_model"):
    # 下载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 保存模型和分词器
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

if __name__ == "__main__":
    download_and_save_model()