# Introduction

This folder contains the configuration files for the Kubernetes cluster. Configuring the cluster will allow the user to distribute the computations over multiple pods and parallelise hyper parameters optimisations.

## Build docker containers (you can most probably ignore this section!)

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
kubectl create -f namespace-main.json
kubectl apply -f pv-volume.yaml  # or kubectl apply -f pv-volume-gcp.yaml if on GCP
kubectl apply -f pv-claim.yaml
kubectl apply -f mysql-secret.yaml
kubectl apply -f mysql.yaml
kubectl apply -f tensorboard.yaml
kubectl apply -f gdeep-lab.yaml
kubectl apply -f redis.yaml
kubectl apply -f rq-worker.yaml  # to run the workers; if you change the location of the parallel_hpo.py file, make sure to change the --path here!
kubectl apply -f lab-ingress.yaml
```

 5. Connect to  `<ingress_external_ip>/one:8888` to access jupyter lab. The token is `abcd`. Alternatively, on minikube, please run `minikube service gdeep-lab-service --url -n main` and you will see the URL at the terminal
 6. Connect to `<ingress_external_ip>/ssh_one` if you want to access via ssh. For ssh:

```
user: matteo
psw: i_like_giotto
```
From `minikube` us the same command as point 5.

 7.  (Optional) scale up or down the workers (*change the number of replicas*):
```
kubectl scale deployment gdeep-rq-worker-deployment --replicas=0 -n main
```

 8. Access the jupyter lab service and in the giotto-deep folder (the volume!) you can `git clone https://github.com/giotto-ai/giotto-deep.git` and also `pip install giotto-deep`.
 9. Run the `giotto-deep/kubernetes/examples/parallel_hpo.ipynb` notebook and see the parallelisation happening!
 10. To check the results, use the tensorboard service: on minikube you can do `minikube service gdeep-tensorboard-service --url -n main`, while on K8 you can simply go to `<ingress_external_ip>/one:8080`

# The general schematics

In this picture we describe the general overview of the cluster:

![img](https://raw.githubusercontent.com/giotto-ai/giotto-deep/master/kubernetes/k8-gdeep.png)

In short, the user have direct access to the `gdeep-lab` and the `gdeep-tensorboard`, for experimenting and visualising. The distribution of HPOs is done in the backend via Redis: the job to be run is pickled and stored in a Redis DB. The workers that are idle will pick-up the job and store teh data on the shared volume. The tensorboard and the jupyter lab will both have access to the share volume.

# Set-up Minikube

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

If you now want to access any of the services, like the Jupyter Lab one, simply move to the corresponding external IP. The `NodePort` type in the services exposes such services outside the cluster.

To proceed, access the `gdeep-lab` service at port 8888. You will need to copy the content of the `kubernetes/examples` folder to the volume to then be able to run the `parallel_hpo.ipynb` notebook. This example, if all has gone well, will distribute the function `run_hpo_parallel`, which contains an example of HPO.

## Debugging on K8/minikube

In case something goes wrong, you can always get the terminal in each pod via
```
kubectl exec -it <pod_name> -n main -- bash
```

If the pod did not even started, try out something like

```
kubectl describe pods gdeep-rq-worker-deployment-6bc5bbfcfc-4wjmg -n main
```
to find out the issues.

If you want to check the outputs/logs of each pod, simply run (get your pod via `kubectl get pods -n main`):
```
kubectl logs gdeep-rq-worker-deployment-c96674448-sbtx6 -n main
```

## Further useful commands for minikube

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

To completely delete the cluster:
```
minikube delete --all
```

## Working on different machines

You can scale-up workers by changing the number of replicas of the RQ workers deployment (see point 7 above). You can add, to each worker, GPUs or other computing hardware.

# Distributing HPO

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
