# terraform instructions

This short readme provide some details on the use of *terraform* to deploy the infrastructure.

## What needs to be done first

The only thing you have to change is the `provider.tf` file with the proper configurations. In our example, the provider must be kubernetes.

Furthermore, depending on the details of the provider you use, **persistent volumes** may work differently and you shall change the resource code in the `infra.tf` file.

## instructions

To fire up terraform:

```
terraform init
```

To see the diff with current infrastructure status:

```
terraform plan
```

To execute the changes to reach the desired state specified in the `infra.tf` file:

```
terraform apply
```

To destroy everything:

```
terraform destroy
```

