FROM python:3.10
RUN pip install --upgrade jax jaxlib
RUN pip install flax
RUN pip3 install accelerate pandas datasets transformers[sentencepiece] pickle-mixin
WORKDIR /
RUN mkdir -p "/input_test"
RUN mkdir -p "/models"
RUN python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2'); tokenizer.save_pretrained('/models/deepset/roberta-base-squad2')"

#COPY validation.jsonl /input_test/
COPY task2_multi.sav /models/
COPY task2_passage.sav /models/
COPY task2_phrase.sav /models/


COPY naive-baseline-task-2.py /

ENTRYPOINT [ "/naive-baseline-task-2.py" ]
