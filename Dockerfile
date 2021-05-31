# base image
# FROM allennlp/allennlp:v0.9.0
FROM python:3.7

# exposing default port for streamlit
EXPOSE 8501

# copy over and install packages
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 --no-cache-dir install torchvision 
RUN pip3 --no-cache-dir install allennlp

# copying everything over
COPY . .

# run app
CMD streamlit run src/app_filename_zip.py