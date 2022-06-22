# Introduction

This folder contains the configuration files for the Kubernetes cluster. Configuring the cluster will allow the user to distribute the computations over multiple pods and parallelise gridsearch as well as training tasks.

## Build docker containers (ignore this section!)

In our case, **all the containers have already been built.** However, in case you need to set-up your own version of the containers, from the root folder of giotto-deep, you can build them like this:

```
docker build -f kubernetes/dockerfiles/jupyter-lab/Dockerfile -t matteocao/giotto-deep:gdeep-lab .
```
Once you are done checking that the containers are properly running, then push them to make them available for download to your cluster

```
docker push matteocao/giotto-deep:gdeep-lab
```

# To set-up K8

All the config files should work out of the blue, just follow these steps:
 1. Move all the files to the kubernetes master server
 2. Set up a persistent machine for volume (no need if using `minikube` on local for testing):
 
```
gcloud compute disks create --size=200GB my-data-disk --region us-central1 --replica-zones us-central1-a,us-central1-b
```

 3. Create namespace `kubectl create namespace main`
 4. Run the following commands, in this order:
```
kubectl apply -f pv-volume.yaml  # or kubectl apply -f pv-volume-gcp.yaml if on GCP
kubectl apply -f pv-claim.yaml
kubectl apply -f mysql-secret.yaml
kubectl apply -f mysql.yaml
kubectl apply -f tensorboard.yaml
kubectl apply -f gdeep-lab.yaml
kubectl apply -f redis.yaml
kubectl apply -f rq-worker.yaml
kubectl apply -f lab-ingress.yaml 
```
 
 5. Connect to  `<ingress_external_ip>/one:8888` to access jupyter lab. The token is `abcd`
 6. Connect to `<ingress_external_ip>/ssh_one` if you want to access via ssh. For ssh:
 
```
user: matteo
psw: i_like_giotto
```

 7.  (Optional) scale up or down the workers (change the number of replicas):
```
kubectl scale deployment gdeep-rq-worker-deployment --replicas=0
```

## Local testing on Minikube

From a terminal with administrator access (but not logged in as root), run:

```
minikube start --disk-size 30000mb
```
In case you also need to update `kubectl`, simply run:

```
minikube kubectl -- get pods -A
```


Then you can **run the commands of the step 3** after moving to the `kubernetes/` folder. In case you deployed too many services, you can always do, for example,
```
kubectl delete deployment gdeep-lab-deployment-1 --namespace main
```
or

```
kubectl delete service gdeep-lab-service-1 --namespace main
```

Afterwards, the cluster is up and running, except for the ingress, which `minikube` does not support. 
To check the cluster status simply run:

```
kubectl get all --namespace main
```

If you now want to access any of the services, like the Jupyter Lab one, simply move to the corresponding external IP. The `NodePort` in the services exposes such services outside the cluster.

To proceed, access the `gdeep-lab` service at port 8888. You will need to copy the content of the `kubernetes/examples` folder to the volume to then be able to run the `parallel_hpo.ipynb` notebook. This example, if has gone well, will run in parallel the function `run_hpo_parallel`, which contains an example of HPO.

### Debugging on K8/minikube

In case something goes wrong, you can always get the terminal in each pod via
```
kubectl exec -it <pod name> --sh
```

If the pod did not even started, try out something like:

```
kubectl describe pods gdeep-rq-worker-deployment-6bc5bbfcfc-4wjmg -n main
```

### Further useful commands

In case you want to access the `minikube` VM:

```
minikube ssh
```
In case you want to clear up the images from `minikube` VM:
```
docker image prune -a
```

To stop `minikube`:

```
minikube stop --all

```

## Working on different machines
 
You can deploy many more pods with the `gdeep-lab.yaml` configuration. This will allow you to open up many instances of the notebook and launch, on each pod, a different computation. You can add, to each pod, GPUs or other computing hardware.
 
## Parallelising HPO

To run hyperparameters optimisations in parallel, we need the support of a `mysql` database. To connect to such database, you need to run the python script follow the instruction on `parallel_hpo.ipynb` from the jupyter-lab instance. The script connects optuna to the database. You will need to set up:
 - MySQL connection: you need to find the <mysql_internal_ip> and assign it to the constant variable `MYSQL_IP` in the `parallel_hpo.ipynb` notebook
 - Redis connection:  you need to find the <redis_internal_ip> and assign it to the constant variable `REDIS_IP` in the `parallel_hpo.ipynb` notebook

You can find the parameters `<mysql_internal_ip>` and `<redis_internal_ip>` by looking at your cluster with `kubectl get all --namespace main` and checking the `ClusterIP` of the `service/mysql-service` and `service/redis-service`.
Finally, to parallelise the optuna computations, recall to set the parameters in the `HyperParameterOptimisation.__init__`:
 ```
study_name="distributed-example", 
db_url="mysql+mysqldb://root:password@<mysql_internal_ip>:3306/example",
 ```
 
Note that such parameters are already set up in the functions provided: you just need to initialise the variables `USER` and `PSW`
where you have only to change `<mysql_internal_ip>` with the ClusterIP found before.
