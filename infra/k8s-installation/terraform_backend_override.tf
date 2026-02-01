terraform {
  backend "s3" {
    bucket = "tfstate-k8s-training-cbaeb13db6a2f95cc4752a790016b8c4"
    key    = "k8s-training.tfstate"

    endpoints = {
      s3 = "https://storage.eu-north1.nebius.cloud:443"
    }
    region = "eu-north1"

    skip_region_validation      = true
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }
}
