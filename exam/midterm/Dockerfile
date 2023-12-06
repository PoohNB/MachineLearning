FROM python:3.11.5-bullseye

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install nbconvert
RUN apt update
RUN apt install -y pandoc
RUN apt install -y texlive-xetex 
RUN apt install -y texlive-fonts-recommended 
RUN apt install -y texlive-plain-generic
RUN rm -rf /var/cache/apt/archives /var/lib/apt/lists/*.


CMD tail -f /dev/null