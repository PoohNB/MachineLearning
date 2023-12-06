# Task 1 (9)

Build a Docker container that uses a base image from `3.11.5-bullseye` along with these pip packages.

1. numpy 1.26.0
2. pandas 2.1.1
3. sklearn 1.3.1
4. matplotlib 3.8.0 

Add this section to get PDF export support.

```Dockerfile
RUN pip3 install nbconvert
RUN apt update
RUN apt install -y pandoc
RUN apt install -y texlive-xetex 
RUN apt install -y texlive-fonts-recommended 
RUN apt install -y texlive-plain-generic
RUN rm -rf /var/cache/apt/archives /var/lib/apt/lists/*.
```

With or without `docker-compose` is fine. 

Map the folder `code` to the `/root/code` of your container.

To submit this task, name the image `ait-ml-2023-midterm` and upload it to your DockerHub.

Proceed next to the `code` folder.