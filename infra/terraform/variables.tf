variable "vm_name" {
  description = "Name of the ML training VM"
  type        = string
  default     = "rideshare-predictor-dev"
}

variable "memory_mb" {
  description = "RAM in MB - increase for larger models"
  type        = number
  default     = 4096
}

variable "vcpu_count" {
  description = "Virtual CPU cores"
  type        = number
  default     = 2
}

variable "base_image_path" {
  description = "Path to Ubuntu 22.04 cloud image on host"
  type        = string
  default     = "/home/ops/vm-images/ubuntu-22.04-base.img"
}

variable "disk_size" {
  description = "VM disk size in bytes (default 20GB)"
  type        = number
  default     = 21474836480
}
