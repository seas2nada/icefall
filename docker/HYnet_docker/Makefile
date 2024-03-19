.DEFAULT_GOAL := help

# GPU configuration
GPU?=0,1,2,3
DOCKER=docker

# # Jupyter notebook option
# JUPYTER=notebook
# JUPYTER=lab

# Docker configuration
ARG_IMAGE_NAME?=hynet2/cuda:base
ARG_CONTAINER_NAME?=icefall
# ARG_CONTAINER_HOSTNAME?=$(shell echo $(ARG_CONTAINER_NAME) | tr a-z A-Z)
ARG_CONTAINER_HOSTNAME?=$(ARG_CONTAINER_NAME)

# Container user permission
ARG_PUID?=$(shell id -u)
ARG_PGID?=$(shell id -g)

# Container network preference
ARG_PORT_SSH?=32778
ARG_PORT_TENSORBOARD?=9211

# Container mount point
ARG_WORKSPACE_HOST?=$(realpath ../../../)
ARG_WORKSPACE_CONTAINER?=/home/Workspace

_build:
	docker build --no-cache -t $(ARG_IMAGE_NAME) \
		-f Dockerfile .

_push:
	docker push $(ARG_IMAGE_NAME)

pull:
	docker pull $(ARG_IMAGE_NAME)

run:
	$(DOCKER) run -d --restart=unless-stopped \
		--name $(ARG_CONTAINER_NAME) \
		--ipc=host \
		-h $(ARG_CONTAINER_HOSTNAME) \
		-e PUID=$(ARG_PUID) -e PGID=$(ARG_PGID) \
		-p $(ARG_PORT_SSH):22 \
		-p $(ARG_PORT_TENSORBOARD):6006 \
		-v $(ARG_WORKSPACE_HOST):$(ARG_WORKSPACE_CONTAINER) \
		-v /DB:/DB \
		-v /exps:/exps \
		-v /etc/timezone:/etc/timezone \
		-v /etc/localtime:/etc/localtime \
		--gpus all \
		$(ARG_IMAGE_NAME)

start:
	docker start $(ARG_CONTAINER_NAME)

stop:
	docker stop $(ARG_CONTAINER_NAME)

rm: stop 
	docker rm $(ARG_CONTAINER_NAME)

copyauth:
	$(eval HOST_IP := $(shell hostname -I | grep -o '^\S*'))
	$(eval SSH_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 22/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	scp -P $(SSH_PORT) $(HOME)/.ssh/authorized_keys icefall@$(HOST_IP):/home/Workspace/.ssh

ssh:
	$(eval HOST_IP := $(shell hostname -I | grep -o '^\S*'))
	$(eval SSH_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 22/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	ssh icefall@$(HOST_IP) -p $(SSH_PORT)

sshroot:
	$(eval HOST_IP := $(shell hostname -I | grep -o '^\S*'))
	$(eval SSH_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 22/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	ssh root@$(HOST_IP) -p $(SSH_PORT)

info:
	$(info ----------------------------------------)
	$(info ARG_IMAGE_NAME          = $(ARG_IMAGE_NAME))
	$(info ARG_CONTAINER_NAME      = $(ARG_CONTAINER_NAME))
	$(info ARG_WORKSPACE_HOST      = $(ARG_WORKSPACE_HOST))
	$(info ARG_WORKSPACE_CONTAINER = $(ARG_WORKSPACE_CONTAINER))
	$(info ----------------------------------------)
	$(info ARG_PUID                = $(ARG_PUID))
	$(info ARG_PGID                = $(ARG_PGID))
	$(info ----------------------------------------)
	$(eval TB_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 6006/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	$(info Port(TensorBoard)=$(TB_PORT))
	$(info ----------------------------------------)
	$(eval HOST_IP := $(shell hostname -I | grep -o '^\S*'))
	$(eval SSH_PORT := $(shell docker port $(ARG_CONTAINER_NAME) | grep 22/ | sed 's/.*0.0.0.0://g' | sed 's/->.*//g'))
	$(info HINT: ssh-keygen -t rsa -b 4096 -C "$$(id -un)[$$(id -u)-$$(id -g)]@$$(hostname)")
	$(info HINT: ssh-copy-id -i ~/.ssh/id_rsa.pub icefall@$(HOST_IP) -p $(SSH_PORT))
	$(info ----------------------------------------)

logs:
	docker logs $(ARG_CONTAINER_NAME)

env:
	docker exec -u $(ARG_PUID):$(ARG_PGID) $(ARG_CONTAINER_NAME) env

bash:
	docker exec -it -u $(ARG_PUID):$(ARG_GUID) $(ARG_CONTAINER_NAME) bash

bashroot:
	docker exec -it $(ARG_CONTAINER_NAME) bash

piplist:
	docker exec -u $(ARG_PUID):$(ARG_PGID) $(ARG_CONTAINER_NAME) pip list

cat:
	@cat Makefile

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'
