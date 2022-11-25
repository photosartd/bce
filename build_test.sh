#!/bin/bash
imageName=bce:latest
containerName=bce

docker build --build-arg command="-m unittest -v" -t $imageName -f Dockerfile .

echo Delete old container...
docker rm -f $containerName

echo Run new container...
docker run --name $containerName $imageName