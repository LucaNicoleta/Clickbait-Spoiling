FROM python:3.10

RUN pip3 install pandas scikit-learn pickle-mixin nltk scipy
RUN python3 -c 'import nltk; nltk.download("punkt"); nltk.download("stopwords")'
RUN mkdir -p "/input_test"
RUN mkdir -p "/models"
#COPY validation.jsonl /input_test/
COPY phrase_plus_passage_div_multi.sav /models/
COPY finalized_model.sav /models/

#RUN apt-get update \
#	&& apt-get install -y git-lfs wget \
#	&& wget 'https://raw.githubusercontent.com/tira-io/tira/development/application/src/tira/templates/tira/tira_git_cmd.py' -O '/opt/conda/lib/python3.7/site-packages/tira.py' \
	#&& git clone 'https://huggingface.co/webis/spoiler-type-classification' /model \
	#&& cd /model \
	#&& git lfs install \
	#&& git fetch \
	#&& git checkout --track origin/deberta-all-three-types-concat-1-checkpoint-1000-epoch-10 \
	#&& rm -Rf .git

COPY transformer-baseline-task-1.py /

ENTRYPOINT [ "/transformer-baseline-task-1.py" ]

