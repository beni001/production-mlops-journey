output "how_to_get_ip" {
  description = "Command to get VM IP address"
  value       = "sudo virsh domifaddr ${var.vm_name}"
}

output "how_to_ssh" {
  description = "SSH command once IP is known"
  value       = "ssh ubuntu@<IP from above> (password: ubuntu)"
}

output "vm_name" {
  description = "Name of the running VM"
  value       = libvirt_domain.ml_vm.name
}
