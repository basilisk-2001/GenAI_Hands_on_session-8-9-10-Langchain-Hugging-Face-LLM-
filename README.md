# GenAI_Hands_on_session-8-9-10-Langchain-Hugging-Face-LLM-
This repository contains an intermediate hands-on that can be used to understand the functioning of Langchain, Advanced NLP with Hugging Face and LLMs

### Hands-On 1: Practical Applications of Langchain

#### Part 1: Overview of LLMs and Langchain

1. **Overview of LLMs:**
   - Discuss the architecture, functionality, and breakthroughs in LLMs.

2. **OpenAI Use via Langchain:**
   - Compare and contrast OpenAI and Langchain.

3. **Prompt Templating:**
   - Create a simple prompt template using Langchain.

**Code Example:**
```python
from langchain import PromptTemplate

# Define a simple prompt template
template = "Translate the following English text to French: {text}"
prompt = PromptTemplate(template=template, input_variables=["text"])

# Use the prompt template
text = "Hello, how are you?"
formatted_prompt = prompt.format(text=text)
print(formatted_prompt)
```

#### Part 2: Document Loader and Highlighting

1. **Document Loader:**
   - Load a document using Langchain.

2. **Highlighting Face via Langchain:**
   - Highlight specific parts of a document.

**Code Example:**
```python
from langchain.document_loaders import LocalDocumentLoader

# Load a document
loader = LocalDocumentLoader(filepath="sample.txt")
document = loader.load()

# Print the loaded document
print(document)
```

### Hands-On 2: Advanced NLP with Hugging Face

#### Part 1: Hugging Face via Langchain

1. **Integrate Hugging Face with Langchain:**
   - Use Hugging Face models via Langchain.

2. **Various APIs in Hugging Face:**
   - Explore different APIs available in Hugging Face.

**Code Example:**
```python
from langchain.llms import HuggingFaceLLM

# Load a Hugging Face model
llm = HuggingFaceLLM(model_name="gpt2")

# Use the model to generate text
text = "The future of AI is"
generated_text = llm(text)
print(generated_text)
```

### Hands-On 3: Fine-Tuning Large Language Models (LLMs)

#### Part 1: Fine-Tuning a Pre-Trained Transformer

1. **Introduction to LLMs:**
   - Discuss the basics of large language models.

2. **Fine-Tuning for a Specific Domain:**
   - Fine-tune a pre-trained transformer on a smaller dataset.

**Code Example:**
```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset

# Load a pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a dataset
dataset = load_dataset("imdb")
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True), batched=True)

# Fine-tune the model
training_args = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=8)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset['train'], eval_dataset=tokenized_dataset['test'])

trainer.train()
```

#### Part 2: Applications in Various Industries

1. **Discuss Fine-Tuning Applications:**
   - Explore how fine-tuning can be applied to different industries such as healthcare, finance, and education.

2. **Compare Model Outputs Before and After Fine-Tuning:**
   - Evaluate the performance of the model before and after fine-tuning.

**Details to Keep in Mind:**
- Ensure the dataset used for fine-tuning is relevant to the specific domain.
- Compare model outputs using metrics like accuracy, precision, recall, and F1-score.

**Deliverables:**
- A report comparing model outputs before and after fine-tuning.
- Examples of fine-tuned model outputs.
